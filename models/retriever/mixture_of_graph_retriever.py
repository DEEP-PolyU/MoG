

import gc
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import concurrent.futures
from .modules import SubgraphRetriever, HybridExpertActivator, ResultsFusionEngine
from .retrieval_utils import GraphDataLoader, SubGraphDataAdapter
from .faiss_embedding import EntityEmbeddingManager, EmbeddingUtils, ChunkEmbeddingManager

from ..utils import logger

import os

@dataclass
class MoGRetrievalResult:
    final_chunk_ids: List[str]
    final_triples: List[Any]
    final_triple_paths: List[Any]
    activated_experts: List[str]
    total_time: float
    phase_times: Dict[str, float]
    fusion_stats: Dict[str, Any]


class MixtureOfGraphRetriever:

    def __init__(
        self,
        dataset_name: str,
        json_path: str = None,
        cache_dir: str = None,
        chunks_dir: str = None,
        top_k: int = 5,
        schema_path: str = None,
        config = None,
        retrieval_mode=""
    ):
        logger.info("=" * 80)
        logger.info("Initializing MixtureOfGraphRetriever")
        logger.info("=" * 80)

        self.retrieval_mode = retrieval_mode
        self.fullRetrieval = False
        if "fullRetrieval" in retrieval_mode:
            self.fullRetrieval = True
            logger.warning("⚠️  Full Retrieval Mode: Full Retrieval without Subgraph Filtering")


        self.dataset_name = dataset_name
        self.device = EmbeddingUtils.setup_device("cuda")
        self.top_k = top_k
        self.config = config
        self.schema_path = schema_path
        self.cache_dir = cache_dir
        self.chunks_dir = chunks_dir
        self.hub_path_name = self.config.output.hub_path_name

        logger.info("📊 Loading graph data and subGraph info...")
        self.data_loader = GraphDataLoader(dataset_name, json_path, config)
        self._graph, self.metadata = self.data_loader.load()

        model_name = config.embeddings.model_name if config else 'all-MiniLM-L6-v2'
        self.qa_encoder = SentenceTransformer(model_name)

        mog_data = self.metadata.get('mixture_of_graph', {})
        logger.warning(mog_data["subGraphs"].keys())
        self.chunk_id2text = {}
        chunk_file = f"{self.chunks_dir}/{self.dataset_name}.txt"
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
                logger.info(f"📄 Loaded {len(self.chunk_id2text)} chunks from {chunk_file}")
            except Exception as e:
                logger.error(f"Error loading chunks: {e}")

        logger.info("🚀 Initializing EntityEmbeddingManager...")

        dataset_cache_dir = os.path.join(cache_dir, dataset_name, "entity_embeddings")
        self.entity_embedding_manager = EntityEmbeddingManager(
            graph=self._graph,
            qa_encoder=self.qa_encoder,
            chunk_id2text=self.chunk_id2text,
            dataset_cache_dir=dataset_cache_dir
        )

        logger.warning("🚀 Initializing ChunkEmbeddingManager...")

        dataset_cache_dir = os.path.join(cache_dir, dataset_name, "chunk_embeddings")
        self.chunk_embedding_manager = ChunkEmbeddingManager(
            qa_encoder=self.qa_encoder,
            chunk_id2text=self.chunk_id2text,
            cache_dir=dataset_cache_dir
        )
        if self.fullRetrieval:
            logger.info("🔧 Initializing full-graph Retriever...")

            full_graph_cache_dir = os.path.join(self.cache_dir, dataset_name, "full_graph")

            all_subGraphs = mog_data.get('subGraphs', {})

            self.subGraph_adapter = SubGraphDataAdapter(mog_data, all_subGraphs)

            self.shared_subGraphs = self.subGraph_adapter.get_shared_subGraphs()
            self.expert_subGraphs = self.subGraph_adapter.get_expert_subGraphs()
            logger.info(f" Found {len(self.shared_subGraphs)} shared subGraphs")
            logger.info(f" Found {len(self.expert_subGraphs)} expert subGraphs")
            self.full_graph_nodes = self._build_full_graph_nodes()

            logger.info(f" Full graph: {len(self.full_graph_nodes)} nodes")
            self.full_graph_retriever = SubgraphRetriever(
                subgraph_nodes=self.full_graph_nodes,
                full_graph=self._graph,
                entity_embedding_manager=self.entity_embedding_manager,
                chunk_embedding_manager=self.chunk_embedding_manager,
                subgraph_name="full_graph",
                cache_dir=full_graph_cache_dir,
                qa_encoder=self.qa_encoder,
                config=config
            )

            self.fusion_engine = ResultsFusionEngine(config=config, chunk_embedding_manager=self.chunk_embedding_manager,
                entity_embedding_manager=self.entity_embedding_manager)

        else:

            all_subGraphs = mog_data.get('subGraphs', {})

            self.subGraph_adapter = SubGraphDataAdapter(mog_data, all_subGraphs)

            self.shared_subGraphs = self.subGraph_adapter.get_shared_subGraphs()
            self.expert_subGraphs = self.subGraph_adapter.get_expert_subGraphs()
            logger.info(f" Found {len(self.shared_subGraphs)} shared subGraphs")
            logger.info(f" Found {len(self.expert_subGraphs)} expert subGraphs")

            logger.info("🏗️ Building subgraph definitions...")

            self.expert_subgraph_nodes = self._build_expert_subgraph_nodes()
            logger.info(f" Expert subgraphs: {len(self.expert_subgraph_nodes)} subgraphs")

            self.expert_retrievers = {}
            logger.info("🌟 r3Hub mode detected - initializing dual Hub retrievers (r3Hub)")
            self._init_3hub_retrievers()

            logger.warning("🌟 Hub retrievers initialized successfully")
            logger.warning(f"🔧 Initializing {len(self.expert_subgraph_nodes)}  Expert Subgraph Retrievers")
            self._init_expert_retriever()
            logger.warning("🌟 Expert retrievers initialized successfully")
            logger.info("🎯 Initializing Activator and Fusion Engine...")

            max_activated_experts = 5
            logger.warning("Using fixed max_activated_experts=5 due to 'act5experts' mode (act5E)")
            self.activator = HybridExpertActivator(
                config=config,
                expert_subGraphs=self.expert_subGraphs,
                entity_embedding_manager=self.entity_embedding_manager,
                chunk_embedding_manager=self.chunk_embedding_manager,
                chunk_id2text=self.chunk_id2text,
                qa_encoder=self.qa_encoder,
                full_graph=self._graph,
                device=str(self.device),
                max_activated_experts=max_activated_experts
            )

            self.fusion_engine = ResultsFusionEngine(
                config=config,
                chunk_embedding_manager=self.chunk_embedding_manager,
                entity_embedding_manager=self.entity_embedding_manager)

        logger.info("=" * 80)
        logger.info(" MixtureOfGraphRetriever initialization completed")
        logger.info("=" * 80)

    def _build_shared_subgraph_nodes(self) -> Set[str]:
        shared_nodes = set()
        for comm_id, subGraph_info in self.shared_subGraphs.items():
            shared_nodes.update(subGraph_info.members)
        return shared_nodes

    def _build_expert_subgraph_nodes(self) -> Dict[str, Set[str]]:
        expert_subgraphs = {}
        for expert_id, subGraph_info in self.expert_subGraphs.items():
            expert_subgraphs[expert_id] = set(subGraph_info.members)
        return expert_subgraphs

    def _build_full_graph_nodes(self) -> Set[str]:
        full_graph_nodes = set()
        for comm_id, subGraph_info in self.shared_subGraphs.items():
            full_graph_nodes.update(subGraph_info.members)
        for expert_id, subGraph_info in self.expert_subGraphs.items():
            full_graph_nodes.update(subGraph_info.members)
        return full_graph_nodes

    def _init_shared_retriever(self):
        logger.info("🔧 Initializing Shared Subgraph Retriever...")

        self.shared_subgraph_nodes = self._build_shared_subgraph_nodes()
        logger.info(f"  Shared subgraph: {len(self.shared_subgraph_nodes)} nodes")

        shared_cache_dir = os.path.join(self.cache_dir, self.dataset_name, self.hub_path_name, "shared_subgraph")

        self.shared_retriever = SubgraphRetriever(
            subgraph_nodes=self.shared_subgraph_nodes,
            full_graph=self._graph,
            entity_embedding_manager=self.entity_embedding_manager,
            chunk_embedding_manager=self.chunk_embedding_manager,
            subgraph_name="Shared",
            cache_dir=shared_cache_dir,
            qa_encoder=self.qa_encoder,
            config=self.config
        )

        logger.info("  ✅ Shared Retriever initialized")


    def _init_3hub_retrievers(self):
        logger.info("🔧 Initializing DCSR Dual Hub Retrievers...")

        semantic_nodes, structural_nodes, unassigned_nodes = self._build_hub_nodes_for_3hub()

        semantic_cache_dir = os.path.join(self.cache_dir, self.dataset_name, self.hub_path_name, "semantic_hub")

        logger.warning(f"  Semantic Hub: {len(semantic_nodes)} nodes")
        self.semantic_hub_retriever = SubgraphRetriever(
            subgraph_nodes=semantic_nodes,
            full_graph=self._graph,
            entity_embedding_manager=self.entity_embedding_manager,
            chunk_embedding_manager=self.chunk_embedding_manager,
            subgraph_name="Semantic_Hub",
            cache_dir=semantic_cache_dir,
            qa_encoder=self.qa_encoder,
            config=self.config
        )

        logger.warning("  ✅ Semantic Hub Retriever initialized")

        structural_cache_dir = os.path.join(self.cache_dir,  self.dataset_name, self.hub_path_name, "structural_hub")

        logger.warning(f"  Structural Hub: {len(structural_nodes)} nodes")
        self.structural_hub_retriever = SubgraphRetriever(
            subgraph_nodes=structural_nodes,
            full_graph=self._graph,
            entity_embedding_manager=self.entity_embedding_manager,
            chunk_embedding_manager=self.chunk_embedding_manager,
            subgraph_name="Structural_Hub",
            cache_dir=structural_cache_dir,
            qa_encoder=self.qa_encoder,
            config=self.config
        )

        logger.warning("  ✅ Structural Hub Retriever initialized")

        unassigned_cache_dir = os.path.join(self.cache_dir, self.dataset_name, self.hub_path_name, "unassigned_hub")

        logger.warning(f"  Unassigned Hub: {len(unassigned_nodes)} nodes")
        self.unassigned_hub_retriever = SubgraphRetriever(
            subgraph_nodes=unassigned_nodes,
            full_graph=self._graph,
            entity_embedding_manager=self.entity_embedding_manager,
            chunk_embedding_manager=self.chunk_embedding_manager,
            subgraph_name="Unassigned_Hub",
            cache_dir=unassigned_cache_dir,
            qa_encoder=self.qa_encoder,
            config=self.config
        )

        logger.warning("  ✅ Unassigned Hub Retriever initialized")

    def _build_hub_nodes_for_3hub(self) -> tuple:
        semantic_nodes = set()
        structural_nodes = set()
        unassigned_nodes = set()
        for comm_id, comm_data in self.shared_subGraphs.items():
            if hasattr(comm_data, 'members'):
                members = comm_data.members
            elif isinstance(comm_data, dict):
                members = comm_data.get('member_nodes', [])
            elif isinstance(comm_data, list):
                members = comm_data
            else:
                logger.warning(f"Unknown format for {comm_id}: {type(comm_data)}")
                continue
            if 'Semantic' in comm_id:
                semantic_nodes.update(members)
                logger.debug(f"  Added {len(members)} nodes to Semantic Hub from {comm_id}")
            elif 'Structural' in comm_id:
                structural_nodes.update(members)
                logger.debug(f"  Added {len(members)} nodes to Structural Hub from {comm_id}")
            elif 'Unassigned' in comm_id:
                unassigned_nodes.update(members)
                logger.debug(f"  Added {len(members)} nodes to Structural Hub from {comm_id}")
            else:
                logger.warning(f"Non shared subGraph detected: {comm_id}, treating as Unassigned Hub")
                semantic_nodes.update(members)

        logger.info(f"  Built DCSR Hub nodes: Semantic={len(semantic_nodes)}, Structural={len(structural_nodes)}, Unassigned={len(unassigned_nodes)}")

        return semantic_nodes, structural_nodes, unassigned_nodes

    def _get_or_create_expert_retriever(self, expert_id: str) -> SubgraphRetriever:
        if expert_id in self.expert_retrievers:
            retriever = self.expert_retrievers[expert_id]
            return retriever

        logger.info(f"🔧 Creating SubgraphRetriever for activated expert: {expert_id}")
        expert_cache_dir = os.path.join(self.cache_dir, self.dataset_name, self.hub_path_name, "expert_subgraphs")


        expert_retriever = SubgraphRetriever(
            subgraph_nodes=self.expert_subgraph_nodes[expert_id],
            full_graph=self._graph,
            entity_embedding_manager=self.entity_embedding_manager,
            chunk_embedding_manager=self.chunk_embedding_manager,
            subgraph_name=expert_id,
            cache_dir=expert_cache_dir,
            qa_encoder=self.qa_encoder,
            config=self.config
        )

        self.expert_retrievers[expert_id] = expert_retriever

        return expert_retriever

    @property
    def graph(self):
        return self._graph

    def _init_expert_retriever(self):
        if self.fullRetrieval:
            logger.warning("⚠️  Ablation Study with Full Retrieval Mode: Skipping Expert FAISS index building")
            return

        logger.info("=" * 80)
        logger.info("🔨 Building FAISS indices for all Expert subGraphs")
        logger.info("=" * 80)

        logger.info(f" Shared indices already built")

        expert_cache_dir = os.path.join(self.cache_dir, self.dataset_name, self.hub_path_name, "expert_subgraphs")

        experts_to_build = []
        for expert_id in self.expert_subgraph_nodes.keys():
            safe_name = expert_id.replace('/', '_').replace('\\', '_')
            entity_cache_path = os.path.join(expert_cache_dir, f"{safe_name}_entity.faiss")

            if not os.path.exists(entity_cache_path):
                experts_to_build.append(expert_id)

        if not experts_to_build:
            logger.info(f" All {len(self.expert_subgraph_nodes)} Expert indices already cached")
            return

        logger.info(f"📦 Need to build {len(experts_to_build)}/{len(self.expert_subgraph_nodes)} Expert indices")
        logger.info(f"   Strategy: Build one by one, save to cache, release memory")

        for idx, expert_id in enumerate(experts_to_build, 1):
            logger.info(f"")
            logger.warning(f"🔨 [{idx}/{len(experts_to_build)}] Building indices for {expert_id}...")

            _ = self._get_or_create_expert_retriever(expert_id)

            safe_name = expert_id.replace('/', '_').replace('\\', '_')
            entity_cache_path = os.path.join(expert_cache_dir, f"{safe_name}_entity.faiss")

            if os.path.exists(entity_cache_path):
                logger.warning(f"    Saved to cache: {safe_name}_*.faiss")
            else:
                logger.warning(f"    Cache not found after build: {expert_id}")

        logger.info("")
        logger.warning("=" * 80)
        logger.warning(f" All Expert FAISS indices built and cached")
        logger.warning(f"   Cache location: {expert_cache_dir}")
        logger.warning("=" * 80)

    def retrieve_single_expert(self, expert_id, query, query_embedding, subGraph_config):
        logger.info(f"  Retrieving from {expert_id}...")

        expert_retriever = self._get_or_create_expert_retriever(expert_id)
        entity_result, expert_chunk_result = expert_retriever.retrieve_full_pipeline(
            query_text=query,
            query_embedding=query_embedding,
            phase=expert_id,
            config_overrides=subGraph_config,
            retrieval_mode=self.retrieval_mode
        )

        logger.info(f"    → {len(entity_result)} entities from {expert_id}")
        logger.info(f"    → {len(expert_chunk_result)} chunks from {expert_id}")

        return expert_id, entity_result, expert_chunk_result, None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> MoGRetrievalResult:
        logger.info(f"🌟 Starting r3Hub Retrieval (3 Hubs  MoG) (r3Hub)")
        return self.retrieval_r3Hub_pipeline(query, top_k)



    def retrieval_r3Hub_pipeline(self, query: str, top_k: Optional[int] = None) -> (MoGRetrievalResult, [], [], [], []):

        start_time = time.time()

        if top_k is None:
            top_k = self.top_k

        logger.info("=" * 80)
        logger.info(f"🚀 Starting MoGRetrieval")
        logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"Top-K: {top_k}")
        logger.info("=" * 80)

        phase_times = {}

        query_embedding = torch.tensor(
            self.qa_encoder.encode(query),
            dtype=torch.float32,
            device=self.device
        )

        logger.info("")
        logger.info("📚 Phase 1: Shared SubGraph Subgraph Retrieval")
        logger.info("-" * 80)

        phase1_start = time.time()

        subGraph_config = {
            'search_top_k': self.config.retrieval.top_k,
            'enable_1hop': False,
            'max_1hop_expand': 0
        }

        all_subGraph_entity_results = {}
        all_subGraph_chunk_results = {}

        semantic_hub_entity_results, semantic_hub_chunk_results = self.semantic_hub_retriever.retrieve_full_pipeline(
            query_text=query,
            query_embedding=query_embedding,
            phase="Semantic_Hub",
            config_overrides=subGraph_config,
            retrieval_mode=self.retrieval_mode
        )
        all_subGraph_entity_results["Semantic_Hub"] = semantic_hub_entity_results
        all_subGraph_chunk_results["Semantic_Hub"] = semantic_hub_chunk_results
        logger.info(f" {len(semantic_hub_entity_results)} entities")
        logger.info(f" {len(semantic_hub_chunk_results)} chunks")


        structural_hub_entity_results, structural_hub_chunk_results = self.structural_hub_retriever.retrieve_full_pipeline(
            query_text=query,
            query_embedding=query_embedding,
            phase="Structural_Hub",
            config_overrides=subGraph_config,
            retrieval_mode=self.retrieval_mode
        )
        all_subGraph_entity_results["Structural_Hub"] = structural_hub_entity_results
        all_subGraph_chunk_results["Structural_Hub"] = structural_hub_chunk_results

        logger.info(f" {len(structural_hub_entity_results)} entities")
        logger.info(f" {len(structural_hub_chunk_results)} chunks")


        unassigned_hub_entity_results, unassigned_hub_chunk_results = self.unassigned_hub_retriever.retrieve_full_pipeline(
            query_text=query,
            query_embedding=query_embedding,
            phase="Unassigned_Hub",
            config_overrides=subGraph_config,
            retrieval_mode=self.retrieval_mode
        )
        all_subGraph_entity_results["Unassigned_Hub"] = unassigned_hub_entity_results
        all_subGraph_chunk_results["Unassigned_Hub"] = unassigned_hub_chunk_results

        logger.info(f" {len(unassigned_hub_entity_results)} entities")
        logger.info(f" {len(unassigned_hub_chunk_results)} chunks")


        phase_times['phase1_shared_retrieval'] = time.time() - phase1_start
        logger.warning(f" Phase 1 completed in {phase_times['phase1_shared_retrieval']:.2f}s")


        logger.info("")
        logger.info("🎯 Phase 2: Expert SubGraph Activation")
        logger.info("-" * 80)
        phase2_start = time.time()

        logger.warning("Using Default AllE Fusion Strategy (hAallE)")
        topk_shared_chunk_ids, topk_shared_sorted_triple_list, _, topk_shared_entity_ids = self.fusion_engine.fuse_results(
            entity_results=all_subGraph_entity_results,
            chunk_results=all_subGraph_chunk_results,
            query_embedding=query_embedding,
            chunk_id2text=self.chunk_id2text,
            final_top_k=top_k
        )
        logger.info(f"  topK chunks from shared retrieval: {topk_shared_chunk_ids}")

        logger.warning("Using Hub-based Expert Activation Strategy (hubAct)")
        activated_experts, activation_info = self.activator.activate_experts_by_hubs(
            topk_shared_entity_ids=topk_shared_entity_ids,
            retrieval_mode = self.retrieval_mode,
        )
        logger.info(f"   Activation details:")
        logger.info(f"     - Total candidates: {activation_info['activated_count']}")

        phase_times['phase2_expert_activation'] = time.time() - phase2_start
        logger.warning(f" Phase 2 completed: {len(activated_experts)} experts activated in {phase_times['phase2_expert_activation']:.2f}s")


        if len(activated_experts) == 0:
            logger.warning("⚠️ No experts activated, skipping Phase 3 Expert retrieval")
            phase_times['phase3_expert_retrieval'] = 0.0
            phase_times['phase4_fusion'] = 0.0
            final_chunk_ids = topk_shared_chunk_ids
            final_triples = topk_shared_sorted_triple_list
            final_triple_paths = []
            fusion_stats = {}
            total_time = time.time() - start_time

            result = MoGRetrievalResult(
                final_chunk_ids=final_chunk_ids,
                final_triples=final_triples,
                final_triple_paths=final_triple_paths,
                activated_experts=activated_experts,
                total_time=total_time,
                phase_times=phase_times,
                fusion_stats=fusion_stats
            )

            logger.info("")
            logger.info("=" * 80)
            logger.info(f" MoGRETRIEVAL COMPLETED")
            logger.info(f"   Total Time: {total_time:.2f}s")
            logger.info(f"   Phase 1 (Shared): {phase_times['phase1_shared_retrieval']:.2f}s")
            logger.info(f"   Phase 2 (Activation): {phase_times['phase2_expert_activation']:.2f}s")
            logger.info(f"   Phase 3 (Expert): {phase_times['phase3_expert_retrieval']:.2f}s")
            logger.info(f"   Phase 4 (Fusion): {phase_times['phase4_fusion']:.2f}s")
            logger.info(f"   Activated Experts: {activated_experts}")
            logger.info(f"   Final Chunks: {len(final_chunk_ids)}")
            logger.info(f"   Final Triples: {len(final_triples)}")
            logger.info(f"   Final Triple Paths: {len(final_triple_paths)}")
            logger.info("=" * 80)

            return result
        logger.info("")
        logger.info("🔬 Phase 3: Expert SubGraphs Subgraph Retrieval")
        logger.info("-" * 80)


        experts_retrieval_start_time = time.time()

        max_workers = min(
            len(activated_experts),
            os.cpu_count() // 3,
            8
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_expert = {
                executor.submit(self.retrieve_single_expert, expert_id, query, query_embedding, subGraph_config): expert_id
                for expert_id in activated_experts
            }

            for future in concurrent.futures.as_completed(future_to_expert):
                expert_id, expert_entity_result, expert_chunk_result, error = future.result()
                if expert_entity_result is not None:
                    all_subGraph_entity_results[expert_id] = expert_entity_result
                if expert_chunk_result is not None:
                    all_subGraph_chunk_results[expert_id] = expert_chunk_result


        phase_times['phase3_expert_retrieval'] = time.time() - experts_retrieval_start_time
        total_expert_entities_count = sum(len(r) for r in all_subGraph_entity_results.values())
        unique_entity_path3_chunk_ids = set()
        for chunk_list in all_subGraph_chunk_results.values():
            for chunk_item in chunk_list:
                chunk_id = chunk_item['chunk_id']
                unique_entity_path3_chunk_ids.add(chunk_id)

        total_expert_path3_chunk_count = len(unique_entity_path3_chunk_ids)
        logger.info(f" Phase 3 completed: {total_expert_entities_count} entities from {len(activated_experts)} experts in {phase_times['phase3_expert_retrieval']:.2f}s")
        logger.warning(f" Phase 3 completed: {total_expert_path3_chunk_count} chunks from {len(activated_experts)} experts in {phase_times['phase3_expert_retrieval']:.2f}s")


        logger.info("")
        logger.info("🔀 Phase 4: Results Fusion")
        logger.info("-" * 80)
        phase4_start = time.time()


        logger.warning("Using Default Fusion Strategy (hAallE)")
        final_chunk_ids, final_triples, final_triple_paths, final_entity_ids = self.fusion_engine.fuse_results(
            entity_results=all_subGraph_entity_results,
            chunk_results=all_subGraph_chunk_results,
            query_embedding=query_embedding,
            chunk_id2text=self.chunk_id2text,
            final_top_k=top_k
        )


        phase_times['phase4_fusion'] = time.time() - phase4_start

        logger.warning(f" Phase 4 completed: {len(final_chunk_ids)} chunks in {phase_times['phase4_fusion']:.2f}s")
        logger.info(f" Phase 4 completed: {len(final_triples)} triples in {phase_times['phase4_fusion']:.2f}s")
        logger.info(f" Phase 4 completed: {len(final_triple_paths)} triple_paths in {phase_times['phase4_fusion']:.2f}s")

        logger.warning("🔄 Starting esActEs Additional Expert Activation and Retrieval (esActEs)")
        esActEs_final_entity_ids = [entity_id for entity_id in final_entity_ids
                                    if entity_id not in topk_shared_entity_ids]

        logger.warning("Using Default Hub-based Expert Activation Strategy")
        logger.warning("Using Hub-based Expert Activation Strategy (hubAct)")
        esActEs_activated_experts, _ = self.activator.activate_experts_by_hubs(
            topk_shared_entity_ids=esActEs_final_entity_ids,
            retrieval_mode=self.retrieval_mode,
        )
        filtered_esActEs = [expert_id for expert_id in esActEs_activated_experts if expert_id not in activated_experts]
        if len(filtered_esActEs) == 0:
            logger.warning("⚠️ No additional experts activated in the esAcTEs mode")
        else:
            logger.warning(f"🔄 esActEs mode activated additional experts: {filtered_esActEs}")
            max_workers = min(
                len(filtered_esActEs),
                os.cpu_count() // 3,
                8
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_expert = {
                    executor.submit(self.retrieve_single_expert, expert_id, query, query_embedding,
                                    subGraph_config): expert_id
                    for expert_id in filtered_esActEs
                }
                for future in concurrent.futures.as_completed(future_to_expert):
                    expert_id, expert_entity_result, expert_chunk_result, error = future.result()
                    if expert_entity_result is not None:
                        all_subGraph_entity_results[expert_id] = expert_entity_result
                        logger.warning(f"🔄 esActEs added {len(expert_entity_result)} entities from the additional expert: {expert_id}")
                    if expert_chunk_result is not None:
                        all_subGraph_chunk_results[expert_id] = expert_chunk_result
                        logger.warning(f"🔄 esActEs added {len(expert_chunk_result)} chunks from the additional expert: {expert_id}")

            activated_experts.extend(filtered_esActEs)
            final_chunk_ids, final_triples, final_triple_paths, final_entity_ids = self.fusion_engine.fuse_results(
                entity_results=all_subGraph_entity_results,
                chunk_results=all_subGraph_chunk_results,
                query_embedding=query_embedding,
                chunk_id2text=self.chunk_id2text,
                final_top_k=top_k
            )
            logger.warning(f"The end of esActEs mode.")

        total_time = time.time() - start_time

        fusion_stats = self.fusion_engine.get_fusion_stats(
            entity_results=all_subGraph_entity_results,
            final_chunk_ids=final_chunk_ids,
            final_triples=final_triples,
            final_triple_paths=final_triple_paths
        )

        result = MoGRetrievalResult(
            final_chunk_ids=final_chunk_ids,
            final_triples=final_triples,
            final_triple_paths=final_triple_paths,
            activated_experts=activated_experts,
            total_time=total_time,
            phase_times=phase_times,
            fusion_stats=fusion_stats
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info(f" MoGRETRIEVAL COMPLETED")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Phase 1 (Shared): {phase_times['phase1_shared_retrieval']:.2f}s")
        logger.info(f"   Phase 2 (Activation): {phase_times['phase2_expert_activation']:.2f}s")
        logger.info(f"   Phase 3 (Expert): {phase_times['phase3_expert_retrieval']:.2f}s")
        logger.info(f"   Phase 4 (Fusion): {phase_times['phase4_fusion']:.2f}s")
        logger.info(f"   Activated Experts: {activated_experts}")
        logger.info(f"   Final Chunks: {len(final_chunk_ids)}")
        logger.info(f"   Final Triples: {len(final_triples)}")
        logger.info(f"   Final Triple Paths: {len(final_triple_paths)}")
        logger.info("=" * 80)

        return result