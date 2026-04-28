
import torch
import torch.nn.functional as F
import networkx as nx
from typing import List, Tuple, Dict, Set
from ...utils import logger
from .faiss_utils import FAISSUtils


class FAISSRetrievalEngine:

    def __init__(self, graph: nx.MultiDiGraph, model, device: torch.device,
                 index_manager, text_processor, cache_manager):
        self.graph = graph
        self.model = model
        self.device = device
        self.index_manager = index_manager
        self.text_processor = text_processor
        self.cache_manager = cache_manager

    def retrieve_via_triples(self, query_embed, top_k: int = 5) -> List[Tuple[str, str, str, float]]:
        if not self.index_manager.triple_index:
            raise ValueError("Triple index not built!")

        if isinstance(query_embed, torch.Tensor):
            query_embed = query_embed.to(self.device)
        else:
            query_embed = torch.FloatTensor(query_embed).to(self.device)

        query_embed = self.index_manager.transform_vector(query_embed)

        cache_key = f"triple_search_{hash(query_embed.cpu().numpy().tobytes())}_{top_k}"
        D, I = self._cached_faiss_search(self.index_manager.triple_index, query_embed, top_k, cache_key)

        all_triples = []
        for idx in I[0]:
            all_triples.extend(self._process_triple_index(idx))

        unique_triples = self._deduplicate_triples(all_triples)

        logger.info(f"Processing {len(unique_triples)} unique triples")

        scored_triples = self._calculate_triple_relevance_scores(
            query_embed, unique_triples, threshold=0.1, top_k=top_k
        )

        logger.info(f"Returned {len(scored_triples)} scored triples")
        return scored_triples

    def retrieve_via_subGraphs(self, query_embed, top_k: int = 5) -> List[str]:
        if not self.index_manager.comm_index:
            raise ValueError("SubGraph index not built!")

        if isinstance(query_embed, torch.Tensor):
            query_embed = query_embed.to(self.device)
        else:
            query_embed = torch.FloatTensor(query_embed).to(self.device)

        query_embed = self.index_manager.transform_vector(query_embed)

        cache_key = f"comm_search_{hash(query_embed.cpu().numpy().tobytes())}_{top_k}"

        D, I = self._cached_faiss_search(self.index_manager.comm_index, query_embed, top_k, cache_key)

        nodes = []
        for idx in I[0]:
            if idx >= 0:
                try:
                    subGraph = self.index_manager.comm_map[str(idx)]
                    subGraph_nodes = self.text_processor.get_subGraph_nodes(subGraph)
                    nodes.extend(subGraph_nodes)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing subGraph index {idx}: {e}")
                    continue

        unique_nodes = []
        seen = set()
        for node in nodes:
            if node not in seen and node in self.index_manager.node_id_to_embedding:
                unique_nodes.append(node)
                seen.add(node)

        return unique_nodes

    def dual_path_retrieval(self, query_emb, top_k: int = 5) -> Dict:
        import time

        start_time = time.time()
        scored_triples = self.retrieve_via_triples(query_emb, top_k)

        triple_nodes = set()
        for h, r, t, score in scored_triples:
            triple_nodes.add(h)
            triple_nodes.add(t)

        triple_nodes = [node for node in triple_nodes if node in self.graph.nodes]
        logger.info(f"Triple retrieval time: {time.time() - start_time:.2f}s")

        start_time = time.time()
        comm_nodes = self.retrieve_via_subGraphs(query_emb, top_k)
        comm_nodes = [node for node in comm_nodes if node in self.graph.nodes]
        logger.info(f"SubGraph retrieval time: {time.time() - start_time:.2f}s")

        merged_nodes = list(set(triple_nodes + comm_nodes))

        start_time = time.time()
        node_scores = self._calculate_node_scores(query_emb, merged_nodes)
        logger.info(f"Node scoring time: {time.time() - start_time:.2f}s")

        result = {
            "triple_nodes": triple_nodes,
            "comm_nodes": comm_nodes,
            "scores": node_scores,
            "scored_triples": scored_triples
        }

        return result

    def _cached_faiss_search(self, index, query_embed, top_k: int, cache_key: str):
        def compute():
            query_embed_np = query_embed.cpu().detach().numpy().reshape(1, -1)
            D, I = FAISSUtils.search(index, query_embed_np, top_k=top_k)
            return (D, I)

        return self.cache_manager.get_faiss_search_result(cache_key, compute)

    def _collect_neighbor_triples(self, node: str) -> List[Tuple[str, str, str]]:
        if node not in self.index_manager.node_id_to_embedding:
            return []

        neighbor_triples = []
        neighbors = self._get_3hop_neighbors(node)

        for neighbor in neighbors:
            for _, target, edge_data in self.graph.out_edges(neighbor, data=True):
                if 'relation' in edge_data and target in self.index_manager.node_id_to_embedding:
                    neighbor_triples.append((neighbor, target, edge_data['relation']))
            for source, _, edge_data in self.graph.in_edges(neighbor, data=True):
                if 'relation' in edge_data and source in self.index_manager.node_id_to_embedding:
                    neighbor_triples.append((source, neighbor, edge_data['relation']))

        return neighbor_triples

    def _process_triple_index(self, idx: int) -> List[Tuple[str, str, str]]:
        try:
            h, r, t = self.index_manager.triple_map[str(idx)]
            triples = [(h, r, t)]

            triples.extend(self._collect_neighbor_triples(h))
            triples.extend(self._collect_neighbor_triples(t))

            return triples

        except (KeyError, ValueError) as e:
            logger.error(f"Error processing triple index {idx}: {e}")
            return []

    def _deduplicate_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        unique_triples = []
        seen = set()

        for triple in triples:
            if triple not in seen:
                unique_triples.append(triple)
                seen.add(triple)

        return unique_triples

    def _get_3hop_neighbors(self, center: str) -> Set[str]:
        if center not in self.index_manager.node_id_to_embedding:
            return set()

        if center not in self.graph.nodes:
            return set()

        def compute():
            neighbors = {center}
            visited = {center}

            try:
                queue = [(center, 0)]

                while queue:
                    current_node, depth = queue.pop(0)

                    if depth >= 3:
                        continue

                    if current_node not in self.graph.nodes:
                        continue

                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor in self.index_manager.node_id_to_embedding and neighbor not in visited:
                            visited.add(neighbor)
                            neighbors.add(neighbor)
                            if depth < 2:
                                queue.append((neighbor, depth + 1))

            except Exception as e:
                logger.error(f"Error getting neighbors for {center}: {e}")

            return neighbors

        return self.cache_manager.get_neighbors(center, n_hops=3, compute_func=compute)

    def _calculate_triple_relevance_scores(self, query_embed: torch.Tensor,
                                          triples: List[Tuple[str, str, str]],
                                          threshold: float = 0.3,
                                          top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        if not triples:
            return []

        scored_triples = []

        for h, r, t in triples:
            try:
                h_embed = self.index_manager.node_id_to_embedding.get(h)
                r_embed = self.index_manager.relation_to_embedding.get(r)
                t_embed = self.index_manager.node_id_to_embedding.get(t)

                if h_embed is None or r_embed is None or t_embed is None:
                    continue

                h_embed = h_embed.to(self.device)
                r_embed = r_embed.to(self.device)
                t_embed = t_embed.to(self.device)

                triple_embed = (h_embed + r_embed + t_embed) / 3.0

                similarity = F.cosine_similarity(
                    query_embed.unsqueeze(0),
                    triple_embed.unsqueeze(0),
                    dim=1
                ).item()

                if similarity >= threshold:
                    scored_triples.append((h, r, t, similarity))

            except Exception as e:
                logger.debug(f"Error scoring triple {h, r, t}: {e}")
                continue

        scored_triples.sort(key=lambda x: x[3], reverse=True)
        return scored_triples[:top_k]

    def _calculate_node_scores(self, query_embed, nodes: List[str]) -> Dict[str, float]:
        if not nodes:
            return {}

        query_embed = query_embed.cpu().detach().numpy()
        query_tensor = torch.FloatTensor(query_embed).to(self.device)
        query_tensor = self.index_manager.transform_vector(query_tensor)

        scores = {}
        node_embeddings = []
        node_names = []

        for node in nodes:
            if node in self.index_manager.node_id_to_embedding:
                embed = self.index_manager.node_id_to_embedding[node].to(self.device)
                node_embeddings.append(embed)
                node_names.append(node)

        if node_embeddings:
            embeddings_tensor = torch.stack(node_embeddings)
            similarities = F.cosine_similarity(
                query_tensor.unsqueeze(0),
                embeddings_tensor,
                dim=1
            )

            for i, node in enumerate(node_names):
                scores[node] = similarities[i].item()

        return scores

