
import os
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from typing import Dict, List, Optional, Any, Tuple

from .embedding_utils import EmbeddingUtils
from .faiss_utils import FAISSUtils

import pickle


from ...utils import logger

class EntityEmbeddingManager:
    def __init__(
        self,
        graph,
        qa_encoder,
        chunk_id2text: Dict[str, str],
        dataset_cache_dir: str
    ):
        self.graph = graph
        self.model = qa_encoder
        self.chunk_id2text = chunk_id2text
        self.device = EmbeddingUtils.setup_device("cuda")
        self.dataset_cache_dir = dataset_cache_dir

        os.makedirs(dataset_cache_dir, exist_ok=True)

        self.entity_embeddings = {}
        self.entity_contents = {}
        self.entity_faiss_index = None
        self.entity_index_map = {}
        self.gpu_resources = None

        self.embeddings_cache_file = os.path.join(dataset_cache_dir, "entity_embeddings.pkl")
        self.index_cache_file = os.path.join(dataset_cache_dir, "entity_faiss_index.pkl")

        logger.info("Initializing EntityEmbeddingManager...")

        if self._load_cached_embeddings():
            logger.info(" Loaded cached entity embeddings")
        else:
            logger.info("Building entity embeddings from scratch...")
            self._build_entity_embeddings()
            self._save_embeddings_cache()

        self._build_entity_faiss_index()

        self.triple_embeddings = {}

        self.subGraph_embeddings = {}


        logger.info(f" EntityEmbeddingManager initialized with {len(self.entity_embeddings)} entities")

    def _load_cached_embeddings(self) -> bool:
        try:
            if os.path.exists(self.embeddings_cache_file):
                with open(self.embeddings_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.entity_embeddings = cache_data['embeddings']
                    self.entity_contents = cache_data['contents']
                    logger.warning(f" Loaded {len(self.entity_embeddings)} entity embeddings from cache.")
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
            logger.info(" Saved entity embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")

    def _build_entity_embeddings(self):
        logger.info("Building entity embeddings...")

        contents = []
        node_ids = []

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]

            if node_data.get('label') != 'entity':
                continue

            content = self._extract_entity_full_content(node_id, node_data)

            if content and len(content.strip()) > 5:
                contents.append(content)
                node_ids.append(node_id)

        batch_size = 512
        total_entities = len(contents)
        entity_count = 0

        for i in range(0, total_entities, batch_size):
            batch_end = min(i + batch_size, total_entities)
            batch_contents = contents[i:batch_end]
            batch_node_ids = node_ids[i:batch_end]

            try:
                batch_embeddings = self.model.encode(batch_contents)

                for j, node_id in enumerate(batch_node_ids):
                    embedding = batch_embeddings[j]
                    self.entity_embeddings[node_id] = torch.tensor(embedding, dtype=torch.float32)
                    self.entity_contents[node_id] = batch_contents[j]
                    entity_count += 1

                logger.info(f"Processed {entity_count}/{total_entities} entities...")

            except Exception as e:
                logger.warning(f"Failed to embed batch {i // batch_size}: {e}")
                for j, node_id in enumerate(batch_node_ids):
                    try:
                        embedding = self.model.encode([batch_contents[j]])
                        self.entity_embeddings[node_id] = torch.tensor(embedding[0], dtype=torch.float32)
                        self.entity_contents[node_id] = batch_contents[j]
                        entity_count += 1
                    except Exception as single_e:
                        logger.warning(f"Failed to embed single entity {node_id}: {single_e}")

        logger.info(f" Built embeddings for {len(self.entity_embeddings)} entities")


    def _extract_entity_full_content(self, node_id: str, node_data: Dict[str, Any]) -> str:
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

        triples = self._extract_node_triples(node_id)
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

    def _extract_node_triples(self, node_id: str) -> List[str]:
        triples = []

        try:
            if hasattr(self.graph, 'successors'):
                for neighbor in list(self.graph.successors(node_id))[:5]:
                    edges = self.graph.get_edge_data(node_id, neighbor)
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

    def _build_entity_faiss_index(self):
        if not self.entity_embeddings:
            logger.warning("No entity embeddings found, skipping FAISS index build")
            return

        logger.info("Building Entity FAISS index...")

        entity_ids = list(self.entity_embeddings.keys())
        embeddings = [self.entity_embeddings[eid].cpu().numpy() for eid in entity_ids]

        embeddings_matrix = np.array(embeddings).astype('float32')


        self.entity_faiss_index = FAISSUtils.build_faiss_index(
            embeddings=embeddings_matrix,
            index_type="flat_ip",
            normalize=True,
            use_gpu=True
        )


        for idx, entity_id in enumerate(entity_ids):
            self.entity_index_map[idx] = entity_id

        if torch.cuda.is_available():
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.entity_faiss_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, 0, self.entity_faiss_index
                )
                logger.info(" Entity FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move entity index to GPU: {e}")

        logger.info(f" Built Entity FAISS index with {len(entity_ids)} entities")

    def get_entity_embedding(self, entity_id: str) -> Optional[torch.Tensor]:
        return self.entity_embeddings.get(entity_id)

    def get_triple_embedding(self, u: str, v: str, relation: str) -> torch.Tensor:
        key = f"{u}_{relation}_{v}"
        if key in self.triple_embeddings:
            return self.triple_embeddings[key]

        triple_text = f"{u} {relation} {v}"
        embedding = self.model.encode(triple_text)
        self.triple_embeddings[key] = torch.tensor(embedding)
        return self.triple_embeddings[key]

    def batch_compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def batch_entity_similarity(
        self,
        query_embedding: torch.Tensor,
        entity_ids: List[str]
    ) -> Dict[str, float]:
        similarities = {}

        valid_entities = []
        valid_embeddings = []

        for entity_id in entity_ids:
            if entity_id in self.entity_embeddings:
                valid_entities.append(entity_id)
                valid_embeddings.append(self.entity_embeddings[entity_id])

        if not valid_embeddings:
            return similarities

        query_tensor = query_embedding.to(self.device).unsqueeze(0)
        entity_tensor = torch.stack(valid_embeddings).to(self.device)

        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        entity_tensor = F.normalize(entity_tensor, p=2, dim=1)

        similarity_scores = torch.mm(query_tensor, entity_tensor.t())[0]
        for i, entity_id in enumerate(valid_entities):
            similarities[entity_id] = similarity_scores[i].cpu().item()

        return similarities

    def faiss_search_entities(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if self.entity_faiss_index is None:
            logger.warning("Entity FAISS index not available")
            return []

        query_np = query_embedding.cpu().numpy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_np)

        scores, indices = FAISSUtils.search(self.entity_faiss_index, query_np, top_k=top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in self.entity_index_map:
                entity_id = self.entity_index_map[idx]
                results.append((entity_id, float(score)))

        return results

    def get_entity_content(self, entity_id: str) -> Optional[str]:
        return self.entity_contents.get(entity_id)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_entities': len(self.entity_embeddings),
            'embedding_dimension': list(self.entity_embeddings.values())[0].shape[0] if self.entity_embeddings else 0,
            'faiss_index_available': self.entity_faiss_index is not None,
            'gpu_accelerated': self.gpu_resources is not None,
            'cached_contents': len(self.entity_contents)
        }

    def batch_compute_triple_embeddings(
        self,
        triples: List[Tuple[str, str, str]],
        use_cache: bool = True
    ) -> torch.Tensor:
        cached_embeddings = []
        uncached_triples = []
        uncached_indices = []

        for idx, (u, v, relation) in enumerate(triples):
            cache_key = (u, v, relation)
            if use_cache and cache_key in self.triple_embeddings:
                cached_embeddings.append((idx, self.triple_embeddings[cache_key]))
            else:
                uncached_triples.append((u, v, relation))
                uncached_indices.append(idx)

        if uncached_triples:
            triple_texts = []
            for u, v, relation in uncached_triples:
                u_name = self._get_node_name(u)
                v_name = self._get_node_name(v)
                triple_texts.append(f"{u_name} {relation} {v_name}")

            new_embeddings = self.model.encode(
                triple_texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )

            if use_cache:
                for triple, embedding in zip(uncached_triples, new_embeddings):
                    self.triple_embeddings[triple] = embedding
        else:
            new_embeddings = []

        all_embeddings = [None] * len(triples)

        for idx, emb in cached_embeddings:
            all_embeddings[idx] = emb

        for idx, emb in zip(uncached_indices, new_embeddings):
            all_embeddings[idx] = emb

        return torch.stack(all_embeddings)

    def get_node_embedding(
        self,
        node_id: str,
        use_cache: bool = True
    ) -> Optional[torch.Tensor]:
        if node_id in self.entity_embeddings:
            return self.entity_embeddings[node_id]

        node_data = self.graph.nodes.get(node_id, {})
        if not node_data:
            return None

        content = self._extract_node_content(node_id, node_data)
        if not content or len(content.strip()) < 5:
            return None

        embedding = self.model.encode(
            content,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

        return embedding

    def batch_compute_node_embeddings(
        self,
        node_ids: List[str],
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        results = {}
        uncached_nodes = []
        uncached_texts = []

        for node_id in node_ids:
            if node_id in self.entity_embeddings:
                results[node_id] = self.entity_embeddings[node_id]
                continue

            node_data = self.graph.nodes.get(node_id, {})
            if node_data:
                content = self._extract_node_content(node_id, node_data)
                if content and len(content.strip()) >= 5:
                    uncached_nodes.append(node_id)
                    uncached_texts.append(content)

        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )

            for node_id, embedding in zip(uncached_nodes, embeddings):
                results[node_id] = embedding

        return results

    def encode_query(
        self,
        query: str,
        use_cache: bool = False
    ) -> torch.Tensor:
        embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

        return embedding

    def _get_node_name(self, node_id: str) -> str:
        node_data = self.graph.nodes.get(node_id, {})
        if 'properties' in node_data:
            name = node_data['properties'].get('name', node_id)
            return str(name)
        return str(node_data.get('name', node_id))

    def _extract_node_content(self, node_id: str, node_data: Dict) -> str:
        content_parts = []

        if 'properties' in node_data:
            props = node_data['properties']
            for field in ['name', 'description', 'title', 'content']:
                if field in props and props[field]:
                    content_parts.append(str(props[field]))

        for field in ['name', 'description', 'title', 'content']:
            if field in node_data and node_data[field]:
                content_parts.append(str(node_data[field]))

        chunk_id = self._extract_chunk_id(node_data)
        if chunk_id and chunk_id in self.chunk_id2text:
            chunk_text = self.chunk_id2text[chunk_id]
            if len(chunk_text) > 50:
                content_parts.append(chunk_text)

        return ' '.join(content_parts)
