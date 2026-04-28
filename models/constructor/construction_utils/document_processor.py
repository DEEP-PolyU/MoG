
import os
import time
from concurrent import futures
from typing import Any, Dict, List

import nanoid
from ...utils import logger
from .schema_agent import SchemaAgent
from .llm_processor_construction import LLMProcessorConstruction

class DocumentProcessor:

    def __init__(self, config, dataset_name, max_workers: int = None):
        self.config = config
        self.schema_agent = SchemaAgent(config, dataset_name)
        self.llm_processor = LLMProcessorConstruction(config)
        self.max_workers = max_workers or min(config.construction.max_workers, (os.cpu_count() or 1) + 4)

    def process_level1_level2(self, chunk: str, chunk_id: str, triple_extraction,
                            graph, dataset_name, lock):
        prompt = self.llm_processor.get_construction_prompt(chunk, dataset_name, "noagent", self.schema_agent.schema)
        llm_response = self.llm_processor.extract_with_llm(prompt)

        parsed_response = self.llm_processor.validate_and_parse_llm_response(prompt, llm_response, None)
        if not parsed_response:
            return

        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        attr_nodes, attr_edges = triple_extraction.process_attributes(extracted_attr, chunk_id, graph, entity_types)
        triple_nodes, triple_edges = triple_extraction.process_triples(extracted_triples, chunk_id, graph, entity_types)

        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges

        with lock:
            for node_id, node_data in all_nodes:
                graph.add_node(node_id, **node_data)
            for u, v, relation in all_edges:
                graph.add_edge(u, v, relation=relation)


    def process_document(self, doc: Dict[str, Any], chunk_manager, triple_extraction,
                        graph, all_chunks, lock, dataset_name):
        try:
            if not doc:
                raise ValueError("Document is empty or None")

            chunks, chunk_id2chunks = chunk_manager.chunk_text(doc, all_chunks, lock)

            if not chunks or not chunk_id2chunks:
                raise ValueError(
                    f"No valid chunks generated from document. Chunks: {len(chunks)}, ID2Chunk: {len(chunk_id2chunks)}")

            for chunk_id, chunk_text in chunk_id2chunks.items():
                self.process_level1_level2(
                    chunk_text, chunk_id, triple_extraction,
                    graph, dataset_name, lock
                )

        except Exception as e:
            error_msg = f"Error processing document: {type(e).__name__}: {str(e)}"
            raise Exception(error_msg) from e



    def process_cached_chunk(self, chunk_id: str, chunk_text: str, triple_extraction,
                           graph, lock, dataset_name: str):
        try:
            self.process_level1_level2(
                chunk_text, chunk_id, triple_extraction,
                graph, dataset_name, lock
            )

        except Exception as e:
            error_msg = f"Error processing cached chunk {chunk_id}: {type(e).__name__}: {str(e)}"
            raise Exception(error_msg) from e

    def process_all_documents(self, documents: List[Dict[str, Any]], chunk_manager,
                            triple_extraction, graph, all_chunks: Dict, lock, dataset_name: str):
        start_construct = time.time()
        total_docs = len(documents)

        logger.info(f"Starting processing {total_docs} documents with {self.max_workers} workers...")

        processed_count = 0
        failed_count = 0

        try:
            with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                all_futures = [
                    executor.submit(
                        self.process_document, doc, chunk_manager, triple_extraction,
                        graph, all_chunks, lock, dataset_name
                    ) for doc in documents
                ]

                for i, future in enumerate(futures.as_completed(all_futures)):
                    try:
                        future.result()
                        processed_count += 1

                        if processed_count % 10 == 0 or processed_count == total_docs:
                            elapsed_time = time.time() - start_construct
                            avg_time_per_doc = elapsed_time / processed_count if processed_count > 0 else 0
                            remaining_docs = total_docs - processed_count
                            estimated_remaining_time = remaining_docs * avg_time_per_doc

                            logger.info(f"Progress: {processed_count}/{total_docs} documents processed "
                                        f"({processed_count / total_docs * 100:.1f}%) "
                                        f"[{failed_count} failed] "
                                        f"ETA: {estimated_remaining_time / 60:.1f} minutes")

                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Document processing failed: {type(e).__name__}: {str(e)}")

        except Exception as e:
            logger.error(f"Critical error in document processing pipeline: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback: {e}")
            raise e

        end_construct = time.time()
        logger.info(f"Construction Time: {end_construct - start_construct}s")
        logger.info(f"Successfully processed: {processed_count}/{total_docs} documents")
        logger.info(f"Failed: {failed_count} documents")


    def process_all_documents_with_cached_chunks(self, cached_chunks: Dict, triple_extraction,
                                               graph, lock, dataset_name: str):
        start_construct = time.time()
        total_chunks = len(cached_chunks)

        logger.warning(f"Starting processing with {total_chunks} cached chunks using {self.max_workers} workers...")

        processed_count = 0
        failed_count = 0

        try:
            with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                all_futures = [
                    executor.submit(
                        self.process_cached_chunk, chunk_id, chunk_text, triple_extraction,
                        graph, lock, dataset_name
                    )
                    for chunk_id, chunk_text in cached_chunks.items()
                ]

                for i, future in enumerate(futures.as_completed(all_futures)):
                    try:
                        future.result()
                        processed_count += 1

                        if processed_count % 10 == 0 or processed_count == total_chunks:
                            elapsed_time = time.time() - start_construct
                            avg_time_per_chunk = elapsed_time / processed_count if processed_count > 0 else 0
                            remaining_chunks = total_chunks - processed_count
                            estimated_remaining_time = remaining_chunks * avg_time_per_chunk

                            logger.warning(f"Progress: {processed_count}/{total_chunks} cached chunks processed "
                                        f"({processed_count / total_chunks * 100:.1f}%) "
                                        f"[{failed_count} failed] "
                                        f"ETA: {estimated_remaining_time / 60:.1f} minutes")

                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Cached chunk processing failed: {type(e).__name__}: {str(e)}")

        except Exception as e:
            logger.error(f"Critical error in cached chunk processing pipeline: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback: {e}")
            raise e

        end_construct = time.time()
        logger.warning(f"Cached Chunk Processing Time: {end_construct - start_construct}s")
        logger.warning(f"Successfully processed: {processed_count}/{total_chunks} cached chunks")
        logger.warning(f"Failed: {failed_count} chunks")

