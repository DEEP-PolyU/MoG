
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from ...utils import logger


class DualHubDetector:

    def __init__(self, config, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.config = config
        self.embedding_model = embedding_model
        self.semantic_percentile = 90
        self.structural_percentile = 90

    def detect_dual_hubs(
        self,
        graph: nx.MultiDiGraph,
        node_embeddings: Dict[str, np.ndarray],
        construction_mode: str
    ) -> Dict[str, Dict]:
        logger.warning("=" * 80)
        logger.warning("DUAL HUB DETECTION")
        logger.warning("=" * 80)

        total_nodes = len(graph.nodes())
        logger.warning(f"Processing {total_nodes} nodes")

        logger.warning("\n[Step 1] Detecting Semantic Hub")
        semantic_hub_nodes = self._detect_semantic_hub(
            graph, node_embeddings, self.semantic_percentile
        )
        logger.warning(f"  Semantic Hub: {len(semantic_hub_nodes)} nodes "
                   f"({len(semantic_hub_nodes)/total_nodes*100:.2f}%)")

        logger.warning("\n[Step 2] Detecting Structural Hub")
        structural_hub_nodes = self._detect_structural_hub(
            graph, self.structural_percentile, construction_mode
        )
        logger.warning(f"  Structural Hub: {len(structural_hub_nodes)} nodes "
                   f"({len(structural_hub_nodes)/total_nodes*100:.2f}%)")

        overlap = semantic_hub_nodes & structural_hub_nodes
        logger.warning(f"  Hub overlap: {len(overlap)} nodes "
                   f"({len(overlap)/total_nodes*100:.2f}%)")

        logger.warning("\n[Step 3] Generating hub profiles")

        dual_hubs = {
            'semantic': self._generate_hub_profile(
                graph, semantic_hub_nodes, node_embeddings, 'semantic'
            ),
            'structural': self._generate_hub_profile(
                graph, structural_hub_nodes, node_embeddings, 'structural'
            )
        }

        logger.info("=" * 80)
        logger.info(" Dual Hub Detection Completed")
        logger.info("=" * 80)

        return dual_hubs

    def _detect_semantic_hub(
        self,
        graph: nx.MultiDiGraph,
        node_embeddings: Dict[str, np.ndarray],
        percentile: float
    ) -> Set[str]:
        logger.info("  Computing semantic centrality...")
        all_embeddings = list(node_embeddings.values())
        if not all_embeddings:
            logger.warning("  No node embeddings available")
            return set()

        mean_embedding = np.mean(all_embeddings, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-9)

        semantic_centrality = {}
        for node, emb in node_embeddings.items():
            emb_normalized = emb / (np.linalg.norm(emb) + 1e-9)
            similarity = np.dot(emb_normalized, mean_embedding)
            semantic_centrality[node] = max(0, similarity)

        for node in graph.nodes():
            if node not in semantic_centrality:
                semantic_centrality[node] = 0.0

        threshold = np.percentile(list(semantic_centrality.values()), percentile)
        semantic_hub_nodes = {
            node for node, score in semantic_centrality.items()
            if score >= threshold
        }

        logger.info(f"  Semantic centrality: threshold={threshold:.4f}, "
                   f"selected {len(semantic_hub_nodes)} nodes")

        return semantic_hub_nodes

    def _detect_structural_hub(
        self,
        graph: nx.MultiDiGraph,
        percentile: float,
            construction_mode: str
    ) -> Set[str]:
        logger.info("  Computing structural centrality...")

        degree_centrality = nx.degree_centrality(graph)
        try:
            sample_k = min(1000, len(graph.nodes()))
            betweenness_centrality = nx.betweenness_centrality(
                graph.to_undirected(),
                k=sample_k
            )
        except Exception:
            logger.warning("  Betweenness centrality failed, using zeros")
            betweenness_centrality = {node: 0.0 for node in graph.nodes()}

        structural_centrality = {}
        if "sHonlyDeg" in construction_mode:
            logger.warning("  Using only Degree Centrality for Structural Hub (sHonlyDeg)")
            for node in graph.nodes():
                deg_score = degree_centrality.get(node, 0)
                structural_centrality[node] = (deg_score)
        elif "sHavg" in construction_mode:
            logger.warning( "  Using Average of Degree and Betweenness Centrality for Structural Hub (sHavg)")
            for node in graph.nodes():
                deg_score = degree_centrality.get(node, 0)
                between_score = betweenness_centrality.get(node, 0)
                structural_centrality[node] = (
                        0.5 * deg_score +
                        0.5 * between_score
                )
        elif "sHonlyBet" in construction_mode:
            logger.warning( "  Using only Betweenness Centrality for Structural Hub (sHonlyBet)")
            for node in graph.nodes():
                between_score = betweenness_centrality.get(node, 0)
                structural_centrality[node] = (between_score)
        elif "sHD8B2" in construction_mode:
            logger.warning("  Using Weighted Combination of Degree and Betweenness Centrality for Structural Hub (sHD8B2)")
            for node in graph.nodes():
                deg_score = degree_centrality.get(node, 0)
                between_score = betweenness_centrality.get(node, 0)
                structural_centrality[node] = (
                    0.8 * deg_score +
                    0.2 * between_score
                )
        else:
            logger.warning("  Using Weighted Combination of Degree and Betweenness Centrality for Structural Hub (sHD8B2)")
            for node in graph.nodes():
                deg_score = degree_centrality.get(node, 0)
                between_score = betweenness_centrality.get(node, 0)
                structural_centrality[node] = (
                    0.8 * deg_score +
                    0.2 * between_score
                )
        threshold = np.percentile(list(structural_centrality.values()), percentile)
        structural_hub_nodes = {
            node for node, score in structural_centrality.items()
            if score >= threshold
        }

        logger.info(f"  Structural centrality: threshold={threshold:.4f}, "
                   f"selected {len(structural_hub_nodes)} nodes")

        return structural_hub_nodes

    def _generate_hub_profile(
        self,
        graph: nx.MultiDiGraph,
        hub_nodes: Set[str],
        node_embeddings: Dict[str, np.ndarray],
        hub_type: str
    ) -> Dict:
        if not hub_nodes:
            logger.warning(f"  No nodes in {hub_type} Hub, creating empty profile")
            return {
                'nodes': [],
                'description': f"Empty {hub_type} hub",
                'embedding': np.zeros(384),
                'size': 0,
                'coverage': 0.0
            }

        hub_nodes_list = list(hub_nodes)
        sample_size = min(50, len(hub_nodes_list))
        sampled_nodes = np.random.choice(
            hub_nodes_list,
            size=sample_size,
            replace=False
        ).tolist()
        node_names = []
        for node in sampled_nodes:
            if node in graph.nodes:
                node_data = graph.nodes[node]
                name = node_data.get('properties', {}).get('name', '')
                if name:
                    node_names.append(name)

        if node_names:
            if hub_type == 'semantic':
                description = (
                    f"Semantic Hub: Contains {len(hub_nodes)} semantically central concepts "
                    f"including {', '.join(node_names[:5])}, etc. "
                    f"These nodes are conceptually representative of the knowledge graph's main themes."
                )
            else:
                description = (
                    f"Structural Hub: Contains {len(hub_nodes)} structurally central nodes "
                    f"including {', '.join(node_names[:5])}, etc. "
                    f"These nodes serve as connection hubs with high degree, betweenness, or eigenvector centrality."
                )
        else:
            description = f"{hub_type.capitalize()} Hub: Contains {len(hub_nodes)} central nodes."

        hub_embeddings = [
            node_embeddings[n] for n in hub_nodes_list
            if n in node_embeddings
        ]

        if hub_embeddings:
            hub_embedding = np.mean(hub_embeddings, axis=0)
            hub_embedding = hub_embedding / (np.linalg.norm(hub_embedding) + 1e-9)
        else:
            hub_embedding = np.zeros(384)

        total_nodes = len(graph.nodes())
        coverage = len(hub_nodes) / total_nodes if total_nodes > 0 else 0.0

        profile = {
            'nodes': hub_nodes_list,
            'description': description,
            'embedding': hub_embedding,
            'size': len(hub_nodes),
            'coverage': coverage
        }

        logger.info(f"  Generated profile for {hub_type} Hub: "
                   f"{len(hub_nodes)} nodes, coverage={coverage*100:.2f}%")

        return profile

    def set_parameters(
        self,
        semantic_percentile: float = None,
        structural_percentile: float = None
    ):
        if semantic_percentile is not None:
            self.semantic_percentile = semantic_percentile
        if structural_percentile is not None:
            self.structural_percentile = structural_percentile

        logger.info("Hub detection parameters updated:")
        logger.info(f"  Semantic Hub: P{self.semantic_percentile}")
        logger.info(f"  Structural Hub: P{self.structural_percentile}")

