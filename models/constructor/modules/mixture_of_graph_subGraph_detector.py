
import time
import random
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm
import networkx as nx
import numpy as np
import os
from ...utils import logger
from sentence_transformers import SentenceTransformer

import pickle
import torch


class MixtureOfGraphHubExpertDetector:

    def __init__(self, config, dataset_name: str, chunk_id2text):
        self.config = config
        self.hub_percentile = config.mixture_of_graph.hub_percentile
        self.dataset_name = dataset_name

        dataset_cache_dir = os.path.join(config.output.cache_dir, dataset_name, "entity_embeddings")
        os.makedirs(dataset_cache_dir, exist_ok=True)
        self.embeddings_cache_file = os.path.join(dataset_cache_dir, "entity_embeddings.pkl")

        self.chunk_id2text = chunk_id2text
        model_name = config.embeddings.model_name if hasattr(config, 'embeddings') else 'all-MiniLM-L6-v2'
        self.qa_encoder = SentenceTransformer(model_name)
        logger.info(f"SubGraphDetector initialized with encoder: {model_name}")

        self.entity_embeddings = {}
        self.entity_contents = {}
        logger.info("SubGraphDetector initialized with LLM description generator enabled")


    def _run_mixture_of_graph_algorithm(self, graph: nx.MultiDiGraph, construction_mode="") -> None:

        logger.warning("Default experiment mode: using Level 2 nodes for subGraph detection")
        subgraph_nodes = [n for n, d in graph.nodes(data=True) if d['level'] == 2]

        logger.info(f"Starting subGraph detection experiment with Mixture-of-Graph")
        logger.warning(f"Processing {len(subgraph_nodes)} nodes")

        subgraph = graph.subgraph(subgraph_nodes)
        subgraph_num_nodes = len(subgraph_nodes)

        logger.warning(" Using Hubs + fuzzy Expert detection")
        detection_result = self._run_hubs_experts_detector(subgraph, subgraph_num_nodes, graph, subgraph_nodes, construction_mode)
        detection_time = 0


        subGraphs = detection_result['subGraphs']
        node_memberships = detection_result['node_memberships']
        shared_subGraphs = detection_result['shared_subGraphs']
        expert_subGraphs = detection_result['expert_subGraphs']
        debug_info = detection_result.get('debug_info', {})

        total_subGraphs = len(subGraphs)
        total_nodes_in_subGraphs = sum(len(members) for members in subGraphs.values())
        hub_stats = detection_result.get('hub_statistics', {})

        logger.warning(f"Hub-Spoke subGraph detection completed:")
        logger.warning(f"  - Initial experts (Tree-Comm): {debug_info.get('initial_experts_count', 'Unknown')}")
        logger.warning(f"  - Final shared subGraphs: {len(shared_subGraphs)}")
        logger.warning(f"  - Final expert subGraphs: {len(expert_subGraphs)}")
        logger.warning(f"  - Total subGraphs: {total_subGraphs}")
        logger.warning(f"  - Total nodes processed: {total_nodes_in_subGraphs}")
        logger.warning(f"  - Hub coverage: {hub_stats.get('coverage_percentage', 0):.1f}%")
        logger.warning(f"  - Hub count: {hub_stats.get('hub_count', 0)}")
        logger.warning(f"  - Algorithm: Hub-Spoke with Expert Fusion")

        if shared_subGraphs:
            shared_sizes = [len(subGraphs[comm_id]) for comm_id in shared_subGraphs if comm_id in subGraphs]
            avg_shared_size = sum(shared_sizes) / len(shared_sizes) if shared_sizes else 0
            logger.warning(f"  - Average shared subGraph size: {avg_shared_size:.1f}")

        if expert_subGraphs:
            expert_sizes = [len(subGraphs[comm_id]) for comm_id in expert_subGraphs if comm_id in subGraphs]
            avg_expert_size = sum(expert_sizes) / len(expert_sizes) if expert_sizes else 0
            logger.warning(f"  - Average expert subGraph size: {avg_expert_size:.1f}")

        logger.warning("Generating descriptions for subGraphs...")
        logger.warning(f"  - Shared subGraphs: {len(shared_subGraphs)} (skipping LLM, using fixed description)")

        subGraph_descriptions = {}
        for comm_id in shared_subGraphs:
            subGraph_descriptions[comm_id] = {
                'description': 'Shared knowledge base covering general and common topics across the knowledge graph',
                'keywords': ['general']
            }
        all_subGraphs_info = []
        for comm_id in expert_subGraphs:
            if comm_id in subGraphs:
                all_subGraphs_info.append({
                    'subGraph_id': comm_id,
                    'subGraph_nodes': subGraphs[comm_id],
                    'subGraph_type': 'expert'
                })


        self._create_shared_super_nodes(graph, shared_subGraphs, subGraphs, level=4,
                                       descriptions={})

        self._create_expert_super_nodes(graph, expert_subGraphs, subGraphs, level=4,
                                       descriptions={})

        graph.graph['shared_expert_result'] = {
            'shared_subGraphs': shared_subGraphs,
            'expert_subGraphs': expert_subGraphs,
            'node_memberships': node_memberships,
            'hub_spoke_features': {
                'hub_count': hub_stats.get('hub_count', 0),
                'coverage_percentage': hub_stats.get('coverage_percentage', 0),
                'initial_subGraphs': hub_stats.get('initial_subGraphs', 0),
                'merged_subGraphs': hub_stats.get('merged_subGraphs', 0),
                'final_subGraphs': hub_stats.get('final_subGraphs', 0),
                'detection_method': 'hub_spoke_with_expert_fusion',
                'algorithm': 'Hub_Spoke_Model',
                'total_subGraphs': len(subGraphs),
                'shared_count': len(shared_subGraphs),
                'expert_count': len(expert_subGraphs),
                'graph_size': subgraph_num_nodes,
                'initial_experts_count': debug_info.get('initial_experts_count', 0)
            },
            'detection_time': detection_time,
            'algorithm_version': 'hub_spoke_v1.0'
        }

        if 'shared_expert_result' in graph.graph:
            stored_result = graph.graph['shared_expert_result']
            logger.warning(f"Debug: shared_expert_result stored successfully with {len(stored_result)} keys")
            logger.warning(f"Debug: Stored shared subGraphs: {len(stored_result.get('shared_subGraphs', []))}")
            logger.warning(f"Debug: Stored expert subGraphs: {len(stored_result.get('expert_subGraphs', []))}")
        else:
            logger.error("Debug: Failed to store shared_expert_result in graph metadata!")

        logger.warning("Hub-Spoke subGraph detection completed successfully")

    def _get_mog_params(self) -> Dict:
        default_params = {
            'struct_weight': 0.3,
            'embedding_model': 'all-MiniLM-L6-v2',
            'min_subGraph_size': 5,
            'hub_percentile': self.hub_percentile,
            'min_hub_degree': 0,
            'min_expert_size': 20,
        }
        config_params = {}

        if hasattr(self.config, 'mixture_of_graph'):
            mog_config = self.config.mixture_of_graph
            config_params.update({
                'min_subGraph_size': getattr(mog_config, 'min_subGraph_size', default_params['min_subGraph_size']),
                'hub_percentile': getattr(mog_config, 'hub_percentile', default_params['hub_percentile']),
                'min_hub_degree': getattr(mog_config, 'min_hub_degree', default_params['min_hub_degree']),
                'min_expert_size': getattr(mog_config, 'min_expert_size', default_params['min_expert_size']),
            })

        final_params = {**default_params, **config_params}

        logger.warning(f"Hub-Spoke detection parameters: {final_params}")
        return final_params


    def _create_shared_super_nodes(self, graph, shared_subGraphs: List,
                                  all_subGraphs: Dict, level: int = 4,
                                  descriptions: Dict = None) -> None:
        if descriptions is None:
            descriptions = {}

        for comm_id in shared_subGraphs:
            if comm_id in all_subGraphs:
                subGraph_nodes = all_subGraphs[comm_id]

                super_node_id = comm_id
                llm_desc = descriptions.get(comm_id, {})
                keywords = llm_desc.get('keywords', [])
                description = llm_desc.get('description', f"Shared knowledge subGraph with {len(subGraph_nodes)} nodes")
                graph.add_node(
                    super_node_id,
                    label="shared_subGraph",
                    level=level,
                    properties={
                        "member_nodes": subGraph_nodes,
                        "keywords": keywords,
                        "description": description,
                        "node_id": super_node_id
                    }
                )
                for node in subGraph_nodes:
                    if node in graph:
                        graph.add_edge(super_node_id, node, relation="contains")

                logger.warning(f"Created shared super node {super_node_id} with LLM-enhanced description ({len(keywords)} keywords)")

    def _create_expert_super_nodes(self, graph: nx.MultiDiGraph, expert_subGraphs: List,
                                  all_subGraphs: Dict, level: int = 4,
                                  descriptions: Dict = None) -> None:

        for comm_id in expert_subGraphs:
            if comm_id in all_subGraphs:
                subGraph_nodes = all_subGraphs[comm_id]

                super_node_id = comm_id
                graph.add_node(
                    super_node_id,
                    label="expert_subGraph",
                    level=level,
                    properties={
                        "subGraph_id": comm_id,
                        "member_nodes": subGraph_nodes,
                        "keywords": "",
                        "auto_discovered": True,
                        "description": "",
                        "node_id": super_node_id
                    }
                )

                for node in subGraph_nodes:
                    if node in graph:
                        graph.add_edge(super_node_id, node, relation="contains")


    def connect_keywords_to_subGraphs(self, graph: nx.MultiDiGraph) -> None:
        comm_nodes = [n for n, d in graph.nodes(data=True) if d.get('level') == 4]
        kw_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'keyword']

        for comm in comm_nodes:
            if comm not in graph.nodes:
                continue
            comm_name = graph.nodes[comm].get('properties', {}).get('name', '').lower()
            for kw in kw_nodes:
                if kw not in graph.nodes:
                    continue
                kw_name = graph.nodes[kw].get('properties', {}).get('name', '').lower()
                if kw_name in comm_name or comm_name in kw_name:
                    graph.add_edge(kw, comm, relation="describes")


    def _run_hubs_experts_detector(self, subgraph, subgraph_num_nodes, graph, subgraph_nodes, construction_mode: str) -> Dict:
        start_time = time.time()

        logger.warning("=" * 80)
        logger.warning("STARTING hub detection FRAMEWORK")
        logger.warning("=" * 80)
        logger.warning(f"Processing {subgraph_num_nodes} nodes")
        
        final_params = self._get_detector_params(construction_mode)
        logger.warning(f"Parameters: {final_params}")

        logger.warning("\n[Step 1] Preparing node embeddings")


        if self._load_cached_embeddings():
            logger.warning(" Loaded cached entity embeddings")
        else:
            logger.warning("Building entity embeddings from scratch...")
            self._build_entity_embeddings(graph)
            self._save_embeddings_cache()
            logger.warning("Built entity embeddings successfully")


        logger.warning("\n[Step 2] Dual Hub Detection")

        from .dual_hub_detector import DualHubDetector

        hub_detector = DualHubDetector(self.config)
        hub_detector.set_parameters(
            semantic_percentile=final_params.get('hub_percentile', 90),
            structural_percentile=final_params.get('hub_percentile', 90)
        )
        dual_hubs = hub_detector.detect_dual_hubs(
            subgraph, self.entity_embeddings, construction_mode
        )
        logger.warning("\n[Step 3] Collecting unassigned nodes")

        all_hub_nodes = set()
        all_hub_nodes.update(dual_hubs['semantic']['nodes'])
        all_hub_nodes.update(dual_hubs['structural']['nodes'])

        no_hub_nodes = [n for n in subgraph_nodes if n not in all_hub_nodes]

        hub_overlap = len(set(dual_hubs['semantic']['nodes']) & set(dual_hubs['structural']['nodes']))
        logger.warning(f"No hub nodes: {len(no_hub_nodes)} ({len(no_hub_nodes)/subgraph_num_nodes*100:.1f}%)")
        logger.warning(f"Hub overlap: {hub_overlap} nodes")

        logger.warning("\n[Step 4] Fuzzy Expert Detection")

        from .fuzzy_expert_detector import FuzzyExpertDetector




        expert_detector = FuzzyExpertDetector(self.config)
        expert_detector.set_parameters(
            min_experts=final_params.get('min_experts', 10),
            max_experts=final_params.get('max_experts', 100),
            fuzziness_m=final_params.get('fuzziness_m', 1.5),
            membership_threshold=final_params.get('membership_threshold', 0.3)
        )

        expert_detector.use_sparse = final_params.get('use_sparse_matrix', True)
        expert_detector.batch_size = final_params.get('batch_size', 1000)
        expert_detector.max_fcm_iterations = final_params.get('max_fcm_iterations', 100)

        expert_processing_nodes = [n for n in subgraph_nodes]
        logger.warning(f" Experiment mode 'all level2Nodes for Experts' activated: using all {len(expert_processing_nodes)} ({len(expert_processing_nodes)/subgraph_num_nodes*100:.1f}%) nodes for expert detection")

        expert_result = expert_detector.detect_fuzzy_experts(
            expert_processing_nodes, self.entity_embeddings
        )

        expert_subGraphs = expert_result['expert_subGraphs']
        expert_overlap_stats = expert_result['expert_overlap_statistics']


        logger.warning("\n[Step 5] Initializing expert descriptions (will be generated in outer layer)")
        for expert_id, comm_info in expert_subGraphs.items():
            if not hasattr(comm_info, 'description') or not comm_info.description:
                comm_info.description = ''
            if not hasattr(comm_info, 'keywords') or not comm_info.keywords:
                comm_info.keywords = []

        logger.warning(f"  Initialized {len(expert_subGraphs)} expert subGraphs (descriptions pending)")

        logger.warning("\n[Step 6] Building detection_result")

        detection_time = time.time() - start_time

        detection_result = self._build_detection_result(
            dual_hubs, expert_subGraphs, subgraph_nodes, expert_overlap_stats,
            subgraph_num_nodes
        )

        logger.warning("\n" + "=" * 80)
        logger.warning("hubs_experts FRAMEWORK COMPLETED")
        logger.warning("=" * 80)
        logger.warning(f"Detection time: {detection_time:.2f}s")
        logger.warning(f"Shared subGraphs: {len(detection_result['shared_subGraphs'])}")
        logger.warning(f"Expert subGraphs: {len(detection_result['expert_subGraphs'])}")
        logger.warning(f"Total subGraphs: {len(detection_result['subGraphs'])}")
        logger.warning("=" * 80)

        return detection_result

    def _load_cached_embeddings(self) -> bool:
        try:
            if os.path.exists(self.embeddings_cache_file):
                with open(self.embeddings_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.entity_embeddings = cache_data['embeddings']
                    self.entity_contents = cache_data['contents']
                    logger.warning(f"Load cached embeddings from: {self.embeddings_cache_file}")
                    return True
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
        return False

    def _save_embeddings_cache(self):
        try:
            cache_data = {
                'embeddings': self.entity_embeddings,
                'contents': self.entity_contents
            }
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.warning(f" Saved entity embeddings to {self.embeddings_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")

    def _build_entity_embeddings(self, graph: nx.MultiDiGraph):
        logger.warning("Building entity embeddings...")

        contents = []
        node_ids = []

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            if node_data.get('label') != 'entity':
                continue

            content = self._extract_entity_full_content(graph, node_id, node_data)

            if content and len(content.strip()) > 5:
                contents.append(content)
                node_ids.append(node_id)

        batch_size = 512
        total_entities = len(contents)
        entity_count = 0

        for i in tqdm(range(0, total_entities, batch_size),
                       desc="Processing entities",
                       unit="batch"):
            batch_end = min(i + batch_size, total_entities)
            batch_contents = contents[i:batch_end]
            batch_node_ids = node_ids[i:batch_end]

            try:
                batch_embeddings = self.qa_encoder.encode(batch_contents)

                for j, node_id in enumerate(batch_node_ids):
                    embedding = batch_embeddings[j]
                    self.entity_embeddings[node_id] = torch.tensor(embedding, dtype=torch.float32)
                    self.entity_contents[node_id] = batch_contents[j]
                    entity_count += 1

                logger.warning(f"Processed {entity_count}/{total_entities} entities...")

            except Exception as e:
                logger.warning(f"Failed to embed batch {i // batch_size}: {e}")
                for j, node_id in enumerate(batch_node_ids):
                    try:
                        embedding = self.qa_encoder.encode([batch_contents[j]])
                        self.entity_embeddings[node_id] = torch.tensor(embedding[0], dtype=torch.float32)
                        self.entity_contents[node_id] = batch_contents[j]
                        entity_count += 1
                    except Exception as single_e:
                        logger.warning(f"Failed to embed single entity {node_id}: {single_e}")

        logger.warning(f" Built embeddings for {len(self.entity_embeddings)} entities")


    def _extract_node_triples(self, graph, node_id: str) -> List[str]:
        triples = []

        try:
            if hasattr(graph, 'successors'):
                for neighbor in list(graph.successors(node_id))[:5]:
                    edges = graph.get_edge_data(node_id, neighbor)
                    if edges:
                        if isinstance(edges, dict):
                            for edge_data in edges.values():
                                if isinstance(edge_data, dict) and 'relation' in edge_data:
                                    relation = edge_data['relation']
                                    triples.append(f"{node_id} {relation} {neighbor}")
                                    break
        except Exception as e:
            logger.debug(f"Error extracting triples for {node_id}: {e}")

        return triples

    def _extract_entity_full_content(self, graph, node_id: str, node_data: Dict[str, Any]) -> str:
        content_parts = []

        chunk_id = self._extract_chunk_id(node_data)
        if chunk_id and chunk_id in self.chunk_id2text:
            chunk_content = self.chunk_id2text[chunk_id]
            if len(chunk_content) > 50:
                content_parts.append(chunk_content)

        if 'properties' in node_data:
            props = node_data['properties']
            for field in ['name', 'description', 'title', 'content', 'summary']:
                if field in props and props[field]:
                    content_parts.append(str(props[field]))

        for field in ['name', 'description', 'title', 'content', 'summary']:
            if field in node_data and node_data[field]:
                content_parts.append(str(node_data[field]))

        triples = self._extract_node_triples(graph, node_id)
        if triples:
            triple_text = ' '.join(triples[:3])
            content_parts.append(triple_text)

        return ' '.join(content_parts)

    def _extract_chunk_id(self, node_data: Dict[str, Any]) -> Optional[str]:
        possible_fields = ['chunk_id', 'chunk id', 'chunkId', 'id', 'source_id']

        if 'properties' in node_data:
            for field in possible_fields:
                if field in node_data['properties']:
                    return str(node_data['properties'][field])

        for field in possible_fields:
            if field in node_data:
                return str(node_data[field])

        return None


    def _build_hubs_experts_metadata(self, dual_hubs: Dict, expert_subGraphs: Dict,
                            expert_overlap_stats: Dict, total_nodes: int) -> Dict:
        metadata = {
            'algorithm': 'hubs_experts',
            'version': '1.0',

            'dual_hubs': {
                'semantic': {
                    'members': dual_hubs['semantic']['nodes'],
                    'description': dual_hubs['semantic']['description'],
                    'size': dual_hubs['semantic']['size'],
                    'coverage': dual_hubs['semantic']['coverage']
                },
                'structural': {
                    'members': dual_hubs['structural']['nodes'],
                    'description': dual_hubs['structural']['description'],
                    'size': dual_hubs['structural']['size'],
                    'coverage': dual_hubs['structural']['coverage']
                }
            },

            'expert_subGraphs': {
                expert_id: {
                    'members': comm_info.members,
                    'keywords': comm_info.keywords,
                    'description': comm_info.description
                }
                for expert_id, comm_info in expert_subGraphs.items()
            },

            'statistics': {
                'total_nodes': total_nodes,
                'hub_nodes': {
                    'semantic': dual_hubs['semantic']['size'],
                    'structural': dual_hubs['structural']['size'],
                    'overlap': len(set(dual_hubs['semantic']['nodes']) &
                                 set(dual_hubs['structural']['nodes'])),
                    'total_unique': len(set(dual_hubs['semantic']['nodes']) |
                                      set(dual_hubs['structural']['nodes']))
                },
                'expert_count': len(expert_subGraphs),
                'avg_expert_size': np.mean([len(c.members) for c in expert_subGraphs.values()]) if expert_subGraphs else 0,
                'expert_overlap_statistics': expert_overlap_stats,
                'coverage': {
                    'hub_coverage': (dual_hubs['semantic']['coverage'] +
                                   dual_hubs['structural']['coverage']) / 2,
                    'expert_coverage': expert_overlap_stats['coverage'],
                    'total_coverage': 1.0
                }
            }
        }
        metadata['shared_subGraphs'] = {
            'Shared_Semantic': dual_hubs['semantic']['nodes'],
            'Shared_Structural': dual_hubs['structural']['nodes']
        }

        metadata['subGraphs'] = {
            **metadata['shared_subGraphs'],
            **{expert_id: comm_info.members
               for expert_id, comm_info in expert_subGraphs.items()}
        }

        return metadata

    def _get_detector_params(self, construction_mode= "") -> Dict:
        default_params = {
            'hub_percentile': self.hub_percentile,

            'min_experts': 5,
            'max_experts': 50,
            'fuzziness_m': 1.5,
            'membership_threshold': 0.3,

            'use_sparse_matrix': True,
            'membership_sparsity_threshold': 0.3,
            'max_fcm_iterations': 100,
            'batch_size': 1000,

            'embedding_model': 'all-MiniLM-L6-v2',
            'min_subGraph_size': 20
        }

        config_params = {}

        if hasattr(self.config, 'mixture_of_graph'):
            mog_config = self.config.mixture_of_graph
            config_params.update({
                'hub_percentile': getattr(mog_config, 'hub_percentile', default_params['hub_percentile'])
            })

        final_params = {**default_params, **config_params}
        return final_params

    def _build_detection_result(
        self,
        dual_hubs: Dict,
        expert_subGraphs: Dict,
        subgraph_nodes: List,
        expert_overlap_stats: Dict,
        subgraph_num_nodes: int
    ) -> Dict:
        logger.warning("  Converting results to standard detection_result format...")

        subGraphs = {}
        shared_subGraphs = {}
        expert_subGraphs_dict = {}

        semantic_comm_id = 'Shared_Semantic'
        subGraphs[semantic_comm_id] = dual_hubs['semantic']['nodes']
        shared_subGraphs[semantic_comm_id] = dual_hubs['semantic']['nodes']

        structural_comm_id = 'Shared_Structural'
        subGraphs[structural_comm_id] = dual_hubs['structural']['nodes']
        shared_subGraphs[structural_comm_id] = dual_hubs['structural']['nodes']

        for expert_id, comm_info in expert_subGraphs.items():
            subGraphs[expert_id] = comm_info.members
            expert_subGraphs_dict[expert_id] = comm_info.members

        node_memberships = {}

        semantic_hub_nodes = set()
        structural_hub_nodes = set()
        all_expert_nodes = set()

        for node in dual_hubs['semantic']['nodes']:
            if node not in node_memberships:
                node_memberships[node] = []
            node_memberships[node].append(semantic_comm_id)
            semantic_hub_nodes.add(node)

        for node in dual_hubs['structural']['nodes']:
            if node not in node_memberships:
                node_memberships[node] = []
            node_memberships[node].append(structural_comm_id)
            structural_hub_nodes.add(node)

        for expert_id, comm_info in expert_subGraphs.items():
            for node in comm_info.members:
                if node not in node_memberships:
                    node_memberships[node] = []
                node_memberships[node].append(expert_id)
                all_expert_nodes.add(node)

        all_assigned_nodes = set(node_memberships.keys())
        logger.info(f"Assigned nodes: {len(all_assigned_nodes)} "
                   f"({len(all_assigned_nodes)/subgraph_num_nodes*100:.1f}%) after hubs_experts.")

        semantic_structural_intersection = semantic_hub_nodes & structural_hub_nodes
        all_hub_nodes = semantic_hub_nodes | structural_hub_nodes

        semantic_expert_intersection = semantic_hub_nodes & all_expert_nodes
        structural_expert_intersection = structural_hub_nodes & all_expert_nodes
        hub_expert_intersection = all_hub_nodes & all_expert_nodes
        all_assigned_nodes = all_hub_nodes | all_expert_nodes
        unassigned_nodes_count = subgraph_num_nodes - len(all_assigned_nodes)


        structural_comm_id = 'Shared_Unassigned'
        shared_unassigned_nodes =  [n for n in subgraph_nodes if n not in all_assigned_nodes]
        subGraphs[structural_comm_id] = shared_unassigned_nodes
        shared_subGraphs[structural_comm_id] = shared_unassigned_nodes


        logger.warning("-" * 50)
        logger.warning(f"Semantic hub nodes: {len(semantic_hub_nodes)} "
                    f"({len(semantic_hub_nodes) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"Structural hub nodes: {len(structural_hub_nodes)} "
                    f"({len(structural_hub_nodes) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"Semantic ∩ Structural hubs: {len(semantic_structural_intersection)} "
                    f"({len(semantic_structural_intersection) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"All hub nodes (union): {len(all_hub_nodes)} "
                    f"({len(all_hub_nodes) / subgraph_num_nodes * 100:.1f}%)")

        logger.warning("-" * 50)
        logger.warning(f"All expert nodes: {len(all_expert_nodes)} "
                    f"({len(all_expert_nodes) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"Semantic hubs ∩ Experts: {len(semantic_expert_intersection)} "
                    f"({len(semantic_expert_intersection) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"Structural hubs ∩ Experts: {len(structural_expert_intersection)} "
                    f"({len(structural_expert_intersection) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"All hubs ∩ Experts: {len(hub_expert_intersection)} "
                    f"({len(hub_expert_intersection) / subgraph_num_nodes * 100:.1f}%)")

        logger.warning("-" * 50)
        logger.warning(f"Assigned nodes: {len(all_assigned_nodes)} "
                    f"({len(all_assigned_nodes) / subgraph_num_nodes * 100:.1f}%) after hubs_experts.")
        logger.warning(f"Unassigned nodes: {unassigned_nodes_count} "
                    f"({unassigned_nodes_count / subgraph_num_nodes * 100:.1f}%)")
        logger.warning(f"Shard_Unassigned nodes: {len(shared_unassigned_nodes)} "
                    f"({len(shared_unassigned_nodes) / subgraph_num_nodes * 100:.1f}%)")
        logger.warning("-" * 50)

        expert_descriptions = {}
        for expert_id, comm_info in expert_subGraphs.items():
            expert_descriptions[expert_id] = {
                'description': getattr(comm_info, 'description', ''),
                'keywords': getattr(comm_info, 'keywords', [])
            }

        hub_overlap = len(set(dual_hubs['semantic']['nodes']) &
                         set(dual_hubs['structural']['nodes']))

        hub_statistics = {
            'hub_count': 2,
            'coverage_percentage': (dual_hubs['semantic']['coverage'] +
                                   dual_hubs['structural']['coverage']) / 2 * 100,
            'semantic_hub_size': dual_hubs['semantic']['size'],
            'structural_hub_size': dual_hubs['structural']['size'],
            'hub_overlap': hub_overlap,
            'total_unique_hub_nodes': len(set(dual_hubs['semantic']['nodes']) |
                                         set(dual_hubs['structural']['nodes'])),
            'algorithm': 'hubs_experts'
        }
        debug_info = {
            'initial_experts_count': len(expert_subGraphs),
            'final_shared_count': 3,
            'final_expert_count': len(expert_subGraphs),
            'total_nodes': subgraph_num_nodes,
            'algorithm': 'hubs_experts',
            'hub_overlap': hub_overlap,
            'fuzzy_clustering': True,
            'avg_memberships_per_node': expert_overlap_stats['avg_memberships_per_node'],
            'coverage': expert_overlap_stats['coverage']
        }

        logger.warning(f"  ✅ Built detection_result: {len(subGraphs)} subGraphs, "
                   f"{len(node_memberships)} nodes with memberships")

        return {
            'subGraphs': subGraphs,
            'node_memberships': node_memberships,
            'shared_subGraphs': shared_subGraphs,
            'expert_subGraphs': expert_subGraphs_dict,
            'hub_statistics': hub_statistics,
            'expert_descriptions': expert_descriptions,
            'debug_info': debug_info
        }


    def _create_hub_centered_subGraphs(
        self,
        graph: nx.Graph,
        hub_nodes: List[str]
    ) -> Dict[int, Set[str]]:
        subGraphs = {}

        for idx, hub in enumerate(hub_nodes):
            if hub not in graph:
                continue

            # Hub + all neighbors
            subGraph = {hub}
            try:
                neighbors = set(graph.neighbors(hub))
                subGraph.update(neighbors)
            except Exception:
                pass

            subGraphs[idx] = subGraph

        logger.warning(f"   Created {len(subGraphs)} hub-centered subGraphs")

        if subGraphs:
            sizes = [len(c) for c in subGraphs.values()]
            logger.warning(f"   Size range: [{min(sizes)}, {max(sizes)}]")
            logger.warning(f"   Average size: {np.mean(sizes):.1f}")

        return subGraphs

    def _merge_subGraphs_by_connectivity(
        self,
        subGraphs: Dict[int, Set[str]],
        graph: nx.Graph,
        max_edge_samples: int = 50000
    ) -> Dict[int, Set[str]]:
        if len(subGraphs) <= 1:
            return subGraphs

        logger.warning(f"   Merging {len(subGraphs)} subGraphs...")

        parent = {cid: cid for cid in subGraphs.keys()}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        node_to_subGraphs = defaultdict(list)
        for comm_id, nodes in subGraphs.items():
            for node in nodes:
                node_to_subGraphs[node].append(comm_id)

        merge_count_overlap = 0
        for node, comm_list in node_to_subGraphs.items():
            if len(comm_list) > 1:
                for i in range(len(comm_list) - 1):
                    if union(comm_list[i], comm_list[i+1]):
                        merge_count_overlap += 1

        logger.warning(f"   Phase 1: Merged {merge_count_overlap} pairs by node overlap")

        merge_count_connectivity = 0

        edges = list(graph.edges())

        if len(edges) > max_edge_samples:
            edges = random.sample(edges, max_edge_samples)
            logger.warning(f"   Sampling {max_edge_samples} / {len(list(graph.edges()))} edges")

        for node1, node2 in edges:
            comms1 = node_to_subGraphs.get(node1, [])
            comms2 = node_to_subGraphs.get(node2, [])

            if comms1 and comms2:
                # Adjacent nodes in graph → merge their subGraphs
                for c1 in comms1:
                    for c2 in comms2:
                        if union(c1, c2):
                            merge_count_connectivity += 1

        logger.warning(f"   Phase 2: Merged {merge_count_connectivity} pairs by graph connectivity")

        merged_groups = defaultdict(set)
        for comm_id, nodes in subGraphs.items():
            root = find(comm_id)
            merged_groups[root].update(nodes)

        merged_subGraphs = {
            idx: nodes
            for idx, (root, nodes) in enumerate(merged_groups.items())
        }

        reduction_rate = (1 - len(merged_subGraphs) / len(subGraphs)) * 100
        logger.warning(f"   Result: {len(subGraphs)} → {len(merged_subGraphs)} subGraphs")
        logger.warning(f"   Reduction: {reduction_rate:.1f}%")

        sorted_comms = sorted(merged_subGraphs.items(),
                             key=lambda x: len(x[1]),
                             reverse=True)

        logger.warning(f"   Top 5 largest:")
        for idx, (cid, nodes) in enumerate(sorted_comms[:5]):
            logger.warning(f"      #{idx+1}: {len(nodes)} nodes")

        return merged_subGraphs

    def _extract_largest_subGraph(
        self,
        merged_subGraphs: Dict[int, Set[str]]
    ) -> Set[str]:
        if not merged_subGraphs:
            return set()

        largest_id = max(merged_subGraphs.items(),
                        key=lambda x: len(x[1]))[0]
        largest_subGraph = merged_subGraphs[largest_id]

        logger.info(f"   Largest subGraph: {len(largest_subGraph)} nodes")
        logger.info(f"   Discarded: {len(merged_subGraphs) - 1} smaller subGraphs")

        return largest_subGraph

    def _expand_hubs_with_hub_creator(
        self,
        graph: nx.Graph,
        subgraph_nodes: List[str],
        dual_hubs: Dict,
        final_params: Dict
    ) -> Dict:
        logger.info("=" * 80)
        logger.info("HUB EXPANSION WITH CONNECTIVITY PROPAGATION")
        logger.info("=" * 80)

        max_edge_samples = final_params.get('hub_merge_max_edges', 50000)

        logger.info("\n[Expanding Semantic Hub]")
        semantic_initial = dual_hubs['semantic']['nodes']
        logger.info(f"Initial size: {len(semantic_initial)} nodes")

        semantic_subGraphs = self._create_hub_centered_subGraphs(
            graph, semantic_initial
        )

        semantic_merged = self._merge_subGraphs_by_connectivity(
            semantic_subGraphs, graph, max_edge_samples
        )

        semantic_expanded = self._extract_largest_subGraph(semantic_merged)

        expansion_ratio_semantic = len(semantic_expanded) / len(semantic_initial) if len(semantic_initial) > 0 else 0
        logger.info(f"✓ Semantic Hub expanded: {len(semantic_initial)} → {len(semantic_expanded)} nodes")
        logger.info(f"  Expansion ratio: {expansion_ratio_semantic:.2f}x")

        logger.info("\n[Expanding Structural Hub]")
        structural_initial = dual_hubs['structural']['nodes']
        logger.info(f"Initial size: {len(structural_initial)} nodes")

        structural_subGraphs = self._create_hub_centered_subGraphs(
            graph, structural_initial
        )

        structural_merged = self._merge_subGraphs_by_connectivity(
            structural_subGraphs, graph, max_edge_samples
        )

        structural_expanded = self._extract_largest_subGraph(structural_merged)

        expansion_ratio_structural = len(structural_expanded) / len(structural_initial) if len(structural_initial) > 0 else 0
        logger.info(f"✓ Structural Hub expanded: {len(structural_initial)} → {len(structural_expanded)} nodes")
        logger.info(f"  Expansion ratio: {expansion_ratio_structural:.2f}x")

        logger.info("\n[Expansion Summary]")

        semantic_set = set(semantic_expanded)
        structural_set = set(structural_expanded)
        overlap = semantic_set & structural_set
        total_unique = semantic_set | structural_set

        total_nodes = len(subgraph_nodes)
        semantic_cov = len(semantic_set) / total_nodes * 100 if total_nodes > 0 else 0
        structural_cov = len(structural_set) / total_nodes * 100 if total_nodes > 0 else 0
        overlap_pct = len(overlap) / total_nodes * 100 if total_nodes > 0 else 0
        total_cov = len(total_unique) / total_nodes * 100 if total_nodes > 0 else 0

        logger.info(f"Semantic Hub: {len(semantic_set)} nodes ({semantic_cov:.1f}% coverage)")
        logger.info(f"Structural Hub: {len(structural_set)} nodes ({structural_cov:.1f}% coverage)")
        logger.info(f"Hub Overlap: {len(overlap)} nodes ({overlap_pct:.1f}% coverage)")
        logger.info(f"Total Unique: {len(total_unique)} nodes ({total_cov:.1f}% coverage)")

        dual_hubs['semantic']['nodes'] = list(semantic_expanded)
        dual_hubs['structural']['nodes'] = list(structural_expanded)
        dual_hubs['overlap'] = list(overlap)

        if 'profile' not in dual_hubs['semantic']:
            dual_hubs['semantic']['profile'] = {}
        dual_hubs['semantic']['profile']['node_count'] = len(semantic_expanded)
        dual_hubs['semantic']['profile']['coverage'] = semantic_cov / 100
        dual_hubs['semantic']['profile']['expansion_ratio'] = expansion_ratio_semantic

        if 'profile' not in dual_hubs['structural']:
            dual_hubs['structural']['profile'] = {}
        dual_hubs['structural']['profile']['node_count'] = len(structural_expanded)
        dual_hubs['structural']['profile']['coverage'] = structural_cov / 100
        dual_hubs['structural']['profile']['expansion_ratio'] = expansion_ratio_structural

        logger.info("=" * 80)

        return dual_hubs


