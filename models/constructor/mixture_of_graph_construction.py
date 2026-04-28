
import json
import os
import threading
from typing import Any, Dict, List
import time
import json_repair
import networkx as nx
import multiprocessing
from config import get_config
from models.utils.logger import logger
from .construction_utils import (ChunkManager, TripleExtractionUtils, ExperimentManager,
                                 DocumentProcessor, GraphIOUtils, CheckpointManager)
from .modules import MixtureOfGraphHubExpertDetector
import concurrent.futures
import random

class MixtureOfGraphConstructor:
    def __init__(self, dataset_name, config, construction_mode="MoGBuild", use_chunks_cache=True):
        self.config = config
        self.graphs_dir = config.output.graphs_dir
        self.dataset_name = dataset_name
        self.node_counter = 0
        self.graph = nx.MultiDiGraph()
        self.entity_to_id = {}
        self.attribute_to_id = {}
        self.token_len = 0
        self.lock = threading.Lock()
        self.all_chunks = {}
        self.construction_mode = construction_mode
        logger.info(f"The pipeline mode: {self.construction_mode}")
        self.use_chunks_cache = use_chunks_cache
        self.datasets_no_chunk = config.construction.datasets_no_chunk

        self.chunk_id2text = {}
        chunk_file = f"{config.output.chunks_dir}/{self.dataset_name}.txt"
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "\t" in line:
                            parts = line.split("\t", 1)
                            if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                                chunk_id = parts[0][4:]
                                chunk_text = parts[1][7:]
                                self.chunk_id2text[chunk_id] = chunk_text
                logger.info(f" Loaded {len(self.chunk_id2text)} chunks from {chunk_file}")
            except Exception as e:
                logger.error(f"Error loading chunks: {e}")

        self._initialize_processors()

    def _initialize_processors(self):
        self.chunk_manager = ChunkManager(self.config, self.dataset_name, self.datasets_no_chunk, self.use_chunks_cache)
        self.graph_IO = GraphIOUtils(self.dataset_name, self.config)
        self.experiment_manager = ExperimentManager(self.dataset_name, self.config)
        self.triple_extraction = TripleExtractionUtils(self)
        self.document_processor = DocumentProcessor(self.config, self.dataset_name)
        self.subGraph_detector = MixtureOfGraphHubExpertDetector(self.config, self.dataset_name, self.chunk_id2text)
        self.checkpoint_manager = CheckpointManager(self.config, self.dataset_name)


    def mog_subGraph_detection(self, name=""):
        logger.warning(f"========{'Mixture-of-Graph':^40}========")
        time_start = time.time()
        self.subGraph_detector._run_mixture_of_graph_algorithm(self.graph, self.construction_mode)
        logger.warning(f"Mixture-of-Graph completed in {time.time() - time_start:.2f} seconds")

        logger.warning("Saving Mixture-of-Graph metadata and graph...")
        self.experiment_manager.save_mog_metadata(
            self.graph, 0
        )
        if name:
            graph_output_path = f"{self.graphs_dir}/{self.dataset_name}_{name}.json"
        else:
            graph_output_path = f"{self.graphs_dir}/{self.dataset_name}.json"
        os.makedirs(self.graphs_dir, exist_ok=True)
        self.graph_IO.save_graph_to_json(self.graph, graph_output_path)

        logger.warning(f"Graph saved:")
        logger.warning(f"  - Graph: {graph_output_path}")
        logger.warning(f"  - Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        logger.warning(f"  - Construction time: {time.time() - time_start}")

    def wrapper_process_all_documents(self,document_processor, *args, **kwargs):
        return document_processor.process_all_documents(*args, **kwargs)

    def preprocess_graph(self, documents):
        time_start = time.time()

        self.document_processor.process_all_documents(
            documents, self.chunk_manager, self.triple_extraction,
            self.graph, self.all_chunks, self.lock, self.dataset_name
        )
        self.document_processor.schema_agent.save_schema()

        logger.warning(f"🚀🚀🚀🚀 {'Processing Level 3 and 4':^20} 🚀🚀🚀🚀")
        logger.warning(f"{'➖' * 20}")
        self.triple_extraction.deduplicate_triples(self.graph)
        self.graph_IO.save_preprocessing_graph(self.graph, time_start)
        logger.warning("Preprocessing graph saved. Skipping subGraph detection.")
        self.chunk_manager.all_chunks.update(self.all_chunks)
        self.chunk_manager.save_chunks_to_file()
        return


    def preprocess_graph_with_cached_chunks(self) -> None:
        logger.warning(f"🚀🚀🚀🚀 {'Processing Level 1 and 2':^20} 🚀🚀🚀🚀")
        logger.warning(f"{'➖' * 20}")
        time_start = time.time()
        cached_chunks = self.chunk_manager.all_chunks
        self.document_processor.process_all_documents_with_cached_chunks(
            cached_chunks, self.triple_extraction,
            self.graph, self.lock, self.dataset_name
        )
        self.document_processor.schema_agent.save_schema()

        logger.warning(f"🚀🚀🚀🚀 {'Processing Level 3 and 4':^20} 🚀🚀🚀🚀")
        logger.warning(f"{'➖' * 20}")

        self.triple_extraction.deduplicate_triples(self.graph)

        self.graph_IO.save_preprocessing_graph(self.graph, time_start)
        logger.warning("Preprocessing graph saved. Skipping subGraph detection.")
        return

    def preprocess_graph_with_batches(self, documents: List[Dict], num_batches: int = 10):
        total_docs = len(documents)
        batch_size = (total_docs + num_batches - 1) // num_batches

        logger.info(f"=" * 60)
        logger.info(f"📦 BATCH PROCESSING MODE")
        logger.info(f"  Total documents: {total_docs}")
        logger.info(f"  Number of batches: {num_batches}")
        logger.info(f"  Batch size: ~{batch_size}")
        logger.info(f"=" * 60)

        can_resume, progress = self.checkpoint_manager.check_resume_capability()

        if can_resume:
            logger.info(f"🔄 RESUMING from previous checkpoint...")
            start_batch = progress["current_batch"]
            if start_batch > 0:
                last_batch = start_batch - 1
                self.graph, self.all_chunks, _ = self.checkpoint_manager.load_checkpoint(
                    last_batch, self.graph_IO
                )
                logger.info(f" Loaded checkpoint from batch {last_batch}")
        else:
            start_batch = 0
            progress = {
                "dataset_name": self.dataset_name,
                "total_documents": total_docs,
                "total_batches": num_batches,
                "current_batch": 0,
                "completed_batches": [],
                "batch_size": batch_size,
                "status": "in_progress",
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_chunks_processed": 0,
                "total_nodes": 0,
                "total_edges": 0
            }

        for batch_id in range(start_batch, num_batches):
            batch_start_time = time.time()

            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, total_docs)
            batch_documents = documents[start_idx:end_idx]

            logger.info(f"\n{'=' * 60}")
            logger.info(f"🚀 Processing Batch {batch_id + 1}/{num_batches}")
            logger.info(f"  Documents range: [{start_idx}:{end_idx}] ({len(batch_documents)} docs)")
            logger.info(f"{'=' * 60}\n")

            batch_chunks = {}

            self.document_processor.process_all_documents(
                batch_documents,
                self.chunk_manager,
                self.triple_extraction,
                self.graph,
                batch_chunks,
                self.lock,
                self.dataset_name
            )

            self.all_chunks.update(batch_chunks)

            batch_processing_time = time.time() - batch_start_time

            logger.info(f"\n💾 Saving checkpoint for batch {batch_id}...")
            self.checkpoint_manager.save_checkpoint(
                batch_id=batch_id,
                graph=self.graph,
                chunks=self.all_chunks,
                batch_info={
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "documents_count": len(batch_documents),
                    "processing_time": batch_processing_time
                },
                graph_io_utils=self.graph_IO
            )

            self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

            progress["current_batch"] = batch_id + 1
            progress["completed_batches"].append(batch_id)
            progress["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            progress["total_chunks_processed"] = len(self.all_chunks)
            progress["total_nodes"] = self.graph.number_of_nodes()
            progress["total_edges"] = self.graph.number_of_edges()

            self.checkpoint_manager.save_progress(progress)

            logger.info(f" Batch {batch_id} completed in {batch_processing_time:.2f}s")
            logger.info(f"  Current totals: {self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges, "
                        f"{len(self.all_chunks)} chunks")


        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎉 ALL BATCHES COMPLETED!")
        logger.info(f"{'=' * 60}\n")

        self.document_processor.schema_agent.save_schema()

        logger.info(f"🚀🚀🚀🚀 {'Processing Level 3 and 4':^20} 🚀🚀🚀🚀")
        logger.info(f"{'➖' * 20}")

        time_start = time.time()
        self.triple_extraction.deduplicate_triples(self.graph)
        self.graph_IO.save_preprocessing_graph(self.graph, time_start)

        self.chunk_manager.all_chunks.update(self.all_chunks)
        self.chunk_manager.save_chunks_to_file()

        progress["status"] = "completed"
        progress["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.checkpoint_manager.save_progress(progress)

        logger.info(" Preprocessing graph saved. Skipping subGraph detection.")
        logger.info(f"📊 Final statistics:")
        logger.info(f"  - Total documents processed: {total_docs}")
        logger.info(f"  - Total chunks: {len(self.all_chunks)}")
        logger.info(f"  - Total nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  - Total edges: {self.graph.number_of_edges()}")




    def build_preprocessed_knowledge_graph(self, corpus):
        logger.info(f"========{'Start Building':^20}========")
        logger.info(f"{'➖' * 30}")

        if "MoGBuild" in self.construction_mode:
            self.graph = self.graph_IO.load_preprocessed_graph()
            logger.warning("Loaded preprocessing graph for experiment")
        else:
            with open(corpus, 'r', encoding='utf-8') as f:
                documents = json_repair.load(f)
            use_batch_processing = self.config.construction.use_batch_processing
            batch_size = self.config.construction.batch_size

            if use_batch_processing:
                logger.warning(f"🔄 Using BATCH PROCESSING mode with batch {batch_size}")
                self.preprocess_graph_with_batches(documents, num_batches=batch_size)
            else:
                logger.warning(f"🔄 Using STANDARD PROCESSING mode")
                if self.use_chunks_cache and self.chunk_manager.check_chunks_cache(corpus):
                    logger.warning("🚀 Found valid chunks cache, loading from file...")
                    if self.chunk_manager.load_chunks_from_cache():
                        logger.warning(" Successfully loaded chunks from cache, skipping chunking step")
                        self.all_chunks = self.chunk_manager.all_chunks.copy()
                        self.preprocess_graph_with_cached_chunks()
                    else:
                        logger.warning("❌ Failed to load chunks from cache, will re-chunk documents")
                        self.preprocess_graph(documents)
                else:
                    if not self.use_chunks_cache:
                        logger.warning("🔄 Chunks cache is disabled by use_chunks_cache=False, will chunk documents from scratch")
                    else:
                        logger.warning("🔄 No valid chunks cache found, will chunk documents and create cache")
                    self.preprocess_graph(documents)

            logger.warning(f"All Process finished, token cost: {self.token_len}")
