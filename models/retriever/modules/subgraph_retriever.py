
import os
import gc
import numpy as np
import torch
import faiss
import spacy
from typing import List, Dict, Set, Any, Optional, Tuple
import concurrent.futures
from pympler import asizeof

from ...utils import logger
from ..faiss_embedding import FAISSUtils


class SubgraphRetriever:
    def __init__(
        self,
        subgraph_nodes: Set[str],
        full_graph,
        entity_embedding_manager,
        chunk_embedding_manager,
        qa_encoder,
        subgraph_name: str = "subgraph",
        cache_dir: Optional[str] = None,
        config=None
    ):
        self.qa_encoder = qa_encoder
        self.subgraph_nodes = set(subgraph_nodes)
        self.graph = full_graph
        self.entity_embedding_manager = entity_embedding_manager
        self.chunk_embedding_manager = chunk_embedding_manager
        self.subgraph_name = subgraph_name
        self.config = config

        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.entity_index = None
        self.entity_map = {}
        self.triple_index = None
        self.triple_map = {}
        self.triple_embeddings = None
        self.entity2idx_map = {}
        self.triple2idx_map = {}
        self.cache_metadata = {
            'subgraph_nodes_hash': self._compute_nodes_hash(),
            'node_count': len(self.subgraph_nodes)
        }
        logger.info(f"🔧 Initializing SubgraphRetriever for '{subgraph_name}'")
        logger.info(f"   - Subgraph nodes: {len(self.subgraph_nodes)}")
        if self.cache_dir:
            logger.info(f"   - Cache directory: {self.cache_dir}")

        if not self._load_faiss_indices_from_cache():
            self._build_entity_faiss_index()
            logger.info(f"✅ Entity faiss built successfully for '{self.subgraph_name}'")
            self._build_triple_faiss_index()
            logger.info(f"✅ Triple faiss built successfully for '{self.subgraph_name}'")
            if self.cache_dir:
                self._save_faiss_indices_to_cache()
                logger.info(f"✅ Cache saved successfully for '{self.subgraph_name}'")
        size_gb = asizeof.asizeof(self.entity_index) / (1024 ** 3)
        logger.info(f"✅ SubgraphRetriever '{subgraph_name}' initialized")

    def _build_entity_faiss_index(self):
        logger.info(f"🔨 Building entity FAISS index for '{self.subgraph_name}'...")

        subgraph_embeddings = []
        subgraph_entity_map = {}

        idx = 0
        for entity_id in self.subgraph_nodes:
            embedding = self.entity_embedding_manager.get_entity_embedding(entity_id)
            if embedding is not None:
                subgraph_embeddings.append(embedding.cpu().numpy())
                subgraph_entity_map[idx] = entity_id
                self.entity2idx_map[entity_id] = idx
                idx += 1

        if not subgraph_embeddings:
            raise ValueError(
                f"No entity embeddings found for subgraph '{self.subgraph_name}'. "
                f"Subgraph has {len(self.subgraph_nodes)} nodes but none have embeddings. "
                f"Please check entity_embedding_manager initialization."
            )

        embeddings_matrix = np.array(subgraph_embeddings).astype('float32')
        self.entity_index = FAISSUtils.build_faiss_index(
            embeddings=embeddings_matrix,
            index_type="flat_ip",
            normalize=True,
            use_gpu=True
        )


        self.entity_map = subgraph_entity_map

        logger.info(f"✅ Built entity FAISS index: {len(subgraph_entity_map)} entities")

    def stringify(self,item):
        if isinstance(item, dict):
            pairs = [f"{k}: {v}" for k, v in sorted(item.items())]
            return ', '.join(pairs)
        return str(item)


    def _build_triple_faiss_index(self):
        logger.info(f"🔨 Building triple FAISS index for '{self.subgraph_name}'...")

        subgraph_triples = []
        subgraph_triple_map = {}

        idx = 0
        for u, v, data in self.graph.edges(data=True):
            if u not in self.subgraph_nodes:
                continue

            relation = data.get('relation', '')
            u_name = self.stringify(self.graph.nodes[u]['properties']['name'])
            v_name = self.stringify(self.graph.nodes[v]['properties']['name'])
            triple_text = f"{u_name} {relation} {v_name}"

            subgraph_triples.append(triple_text)
            subgraph_triple_map[idx] = (u, v, relation)
            self.triple2idx_map[(u_name, relation, v_name)] = idx
            idx += 1

        if not subgraph_triples:
            logger.debug(f"No internal edges found in '{self.subgraph_name}' "
                        f"({len(self.subgraph_nodes)} nodes). Triple retrieval will be skipped.")
            self.triple_index = None
            self.triple_map = {}
            return

        logger.info(f"   Computing embeddings for {len(subgraph_triples)} triples...")
        self.triple_embeddings = self.qa_encoder.encode(
            subgraph_triples,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        self.triple_index = FAISSUtils.build_faiss_index(
            embeddings=self.triple_embeddings,
            index_type="flat_ip",
            normalize=True,
            use_gpu=True
        )

        self.triple_map = subgraph_triple_map

        logger.info(f"✅ Built triple FAISS index: {len(subgraph_triple_map)} triples")



    def compute_query_triple_similarity_by_tuple(self, query_embedding, triple_tuple):
        if triple_tuple not in self.triple2idx_map:
            return 0.0
        idx = self.triple2idx_map.get(triple_tuple)
        if idx is None:
            raise ValueError(f"Triple {triple_tuple} not found in triple2idx_map.")

        triple_emb = self.triple_embeddings[idx]

        normed_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        normed_triple_emb = triple_emb / (np.linalg.norm(triple_emb) + 1e-8)
        similarity = float(np.dot(normed_query, normed_triple_emb))
        return similarity


    def retrieve_entities_by_faiss(
            self,
            query_embedding: torch.Tensor,
            search_top_k: int = 50,
            retrieval_mode=""
    ) -> (List[Dict], List[Dict]):
        if self.entity_index is None:
            raise RuntimeError(
                f"Entity index is None for '{self.subgraph_name}'. "
                f"This should not happen after initialization. "
                f"Check _build_entity_faiss_index() execution."
            )

        query_vec = query_embedding.cpu().numpy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)
        search_k = min(search_top_k, len(self.entity_map))

        distances, indices = FAISSUtils.search(self.entity_index, query_vec, top_k=search_k)

        entity_results = []
        chunk_results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue

            entity_id = self.entity_map[int(idx)]

            node_data = self.graph.nodes.get(entity_id, {})
            chunk_id = node_data.get('properties', {}).get('chunk id', None)
            entity_name = self.stringify(node_data.get('properties', {}).get('name', ""))
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                neighbor_data = self.graph.nodes.get(neighbor, {})
                neighbor_chunk_id = neighbor_data.get('properties', {}).get('chunk id', None)
                neighbor_entity_name = self.stringify(neighbor_data.get('properties', {}).get('name', ""))
                relation = list(edge_data.values())[0]['relation']
                triple =  (entity_name, relation, neighbor_entity_name)
                triple_similarity = self.compute_query_triple_similarity_by_tuple(query_embedding, triple)
                if neighbor_chunk_id:
                    entity_results.append({
                        'entity_id': neighbor,
                        'entity_name': neighbor_entity_name,
                        'chunk_id': neighbor_chunk_id,
                        'retrieval_path': 'entity_one_hop_expand',
                        'triple':triple,
                        'triple_similarity': triple_similarity,
                        'source_phase': 'unknown'
                    })
                    chunk_results.append({
                        'chunk_id': neighbor_chunk_id,
                        'retrieval_path': 'entity_one_hop_expand',
                        'source_phase': 'unknown'
                    })
                else:
                    entity_results.append({
                        'entity_id': neighbor,
                        'entity_name': neighbor_entity_name,
                        'retrieval_path': 'entity_one_hop_expand',
                        'triple': triple,
                        'triple_similarity': triple_similarity,
                        'source_phase': 'unknown'
                    })

            if chunk_id:
                entity_results.append({
                    'entity_id': entity_id,
                    'entity_name': entity_name,
                    'entity_similarity': float(dist),
                    'chunk_id': chunk_id,
                    'retrieval_path': 'entity_faiss',
                    'source_phase': 'unknown',
                })
                chunk_results.append({
                    'chunk_id': chunk_id,
                    'retrieval_path': 'chunk_faiss',
                    'source_phase': 'unknown'
                })


        return entity_results, chunk_results


    def retrieve_entities_by_triple_faiss(
        self,
        query_embedding: torch.Tensor,
        search_top_k: int = 30
    ) -> (List[Dict], List[Dict]):
        if self.triple_index is None:
            logger.debug(f"Triple index not available for '{self.subgraph_name}', skipping triple retrieval")
            return []

        query_vec = query_embedding.cpu().numpy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)

        search_k = min(search_top_k, len(self.triple_map))
        distances, indices = FAISSUtils.search(self.triple_index, query_vec, top_k=search_k)


        entity_results = []
        chunk_results = []
        seen_entities = set()

        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue

            u, v, relation = self.triple_map[int(idx)]

            u_data = self.graph.nodes.get(u, {})
            v_data = self.graph.nodes.get(v, {})
            u_name = self.stringify(u_data.get('properties', {}).get('name', None))
            v_name = self.stringify(v_data.get('properties', {}).get('name', None))
            entities = {
                u: u_data,
                v: v_data
            }
            for entity_id, node_data in entities.items():
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)
                chunk_id = node_data.get('properties', {}).get('chunk id', None)
                entity_name = node_data.get('properties', {}).get('name', "")

                if chunk_id:
                    entity_results.append({
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'triple_similarity': float(dist),
                        'chunk_id': chunk_id,
                        'retrieval_path': 'triple_faiss',
                        'triple': (u_name, relation, v_name),
                        'source_phase': 'unknown'
                    })
                    chunk_results.append({
                        'chunk_id': chunk_id,
                        'retrieval_path': 'chunk_faiss',
                        'source_phase': 'unknown'
                    })
                else:
                    entity_results.append({
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'triple_similarity': float(dist),
                        'retrieval_path': 'triple_faiss',
                        'triple': (u_name, relation, v_name),
                        'source_phase': 'unknown'
                    })


        return entity_results, chunk_results


    def _get_node_props_string(self, node_id: str) -> str:
        node_data = self.graph.nodes.get(node_id, {})
        name = node_data.get('properties', {}).get('name', node_id)
        description = node_data.get('properties', {}).get('description', '')
        return f"{name} [{description}]" if description else str(name)


    def retrieve_full_pipeline(
        self,
        query_text,
        query_embedding: torch.Tensor,
        phase: str = 'unknown',
        config_overrides: Optional[Dict] = None,
        retrieval_mode = ""
    ) -> (List[Dict], List[Dict]):

        return self.retrieve_entityTriple_pipelines(
            query_embedding=query_embedding,
            phase=phase,
            config_overrides=config_overrides
        )

    def retrieve_entityTriple_pipelines(
        self,
        query_embedding: torch.Tensor,
        phase: str = 'unknown',
        config_overrides: Optional[Dict] = None
    ) -> (List[Dict], List[Dict]):
        logger.info(f"🚀 [{self.subgraph_name}] Starting full pipeline retrieval for phase='{phase}'")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_path1 = executor.submit(
                self.retrieve_entities_by_faiss,
                query_embedding,
                search_top_k=config_overrides['search_top_k']
            )
            future_path2 = executor.submit(
                self.retrieve_entities_by_triple_faiss,
                query_embedding,
                search_top_k=config_overrides['search_top_k']
            )
            path1_entity_results, path1_chunk_results = future_path1.result()
            path2_entity_results, path2_chunk_results  = future_path2.result()

        logger.info(f"  Path1 (Entity FAISS): {len(path1_entity_results)} entities, {len(path1_chunk_results)} chunks")
        logger.info(f"  Path2 (Triple FAISS): {len(path2_entity_results)} entities, {len(path2_chunk_results)} chunks")

        for result in path1_entity_results:
            result['source_phase'] = phase
        for result in path1_chunk_results:
            result['source_phase'] = phase
        for result in path2_entity_results:
            result['source_phase'] = phase
        for result in path2_chunk_results:
            result['source_phase'] = phase

        all_entity_results = path1_entity_results + path2_entity_results
        all_chunk_results = path1_chunk_results + path2_chunk_results
        entity_unique = {}
        for result in all_entity_results:
            entity_id = result['entity_id']
            if entity_id not in entity_unique:
                entity_unique[entity_id] = result
            else:
                old_path = entity_unique[entity_id]['retrieval_path']
                entity_unique[entity_id] = result
                entity_unique[entity_id]['retrieval_path'] = f"{old_path}+{result['retrieval_path']}"


        chunk_unique = {}
        for result in all_chunk_results:
            chunk_id = result['chunk_id']
            if chunk_id not in chunk_unique:
                chunk_unique[chunk_id] = result

        final_entity_results = entity_unique.values()
        final_chunk_results = chunk_unique.values()
        logger.info(f"✅ [{self.subgraph_name}] Full pipeline completed: {len(final_entity_results)} unique entities")
        logger.info(f"✅ [{self.subgraph_name}] pipeline-3 completed: {len(final_chunk_results)} unique chunks")

        return final_entity_results, final_chunk_results


    def _compute_nodes_hash(self) -> str:
        import hashlib
        nodes_str = '|'.join(sorted(self.subgraph_nodes))
        return hashlib.md5(nodes_str.encode()).hexdigest()[:16]

    def _get_cache_paths(self) -> Dict[str, str]:
        if not self.cache_dir:
            return {}

        safe_name = self.subgraph_name.replace('/', '_').replace('\\', '_')
        return {
            'entity_index': os.path.join(self.cache_dir, f"{safe_name}_entity.faiss"),
            'entity_map': os.path.join(self.cache_dir, f"{safe_name}_entity_map.pkl"),
            'triple_index': os.path.join(self.cache_dir, f"{safe_name}_triple.faiss"),
            'triple_map': os.path.join(self.cache_dir, f"{safe_name}_triple_map.pkl"),
            'metadata': os.path.join(self.cache_dir, f"{safe_name}_metadata.pkl"),
            'triple_embeddings': os.path.join(self.cache_dir, f"{safe_name}_triple_embeddings.npy"),
            'triple2idx_map': os.path.join(self.cache_dir, f"{safe_name}_triple2idx_map.pkl"),
            'entity2idx_map': os.path.join(self.cache_dir, f"{safe_name}_entity2idx_map.pkl"),
        }

    def _load_faiss_indices_from_cache(self) -> bool:
        if not self.cache_dir:
            return False

        cache_paths = self._get_cache_paths()

        if not os.path.exists(cache_paths['entity_index']) or \
           not os.path.exists(cache_paths['entity_map']) or \
           not os.path.exists(cache_paths['metadata']):
            logger.debug(f"Cache not found for '{self.subgraph_name}'")
            return False

        try:
            import pickle
            with open(cache_paths['metadata'], 'rb') as f:
                cached_metadata = pickle.load(f)

            if cached_metadata.get('subgraph_nodes_hash') != self.cache_metadata['subgraph_nodes_hash']:
                logger.warning(f"Cache invalid for '{self.subgraph_name}': nodes changed")
                return False

            logger.info(f"📦 Loading entity FAISS index from cache for '{self.subgraph_name}'...")
            self.entity_index = faiss.read_index(cache_paths['entity_index'])
            with open(cache_paths['entity_map'], 'rb') as f:
                self.entity_map = pickle.load(f)

            logger.info(f"✅ Loaded entity index: {len(self.entity_map)} entities")

            logger.info(f"📦 Loading triple FAISS index from cache for '{self.subgraph_name}'...")
            self.triple_index = faiss.read_index(cache_paths['triple_index'])

            with open(cache_paths['triple_map'], 'rb') as f:
                self.triple_map = pickle.load(f)

            logger.info(f"✅ Loaded triple index: {len(self.triple_map)} triples")

            triple_emb_path = cache_paths['triple_embeddings']
            if os.path.exists(triple_emb_path):
                self.triple_embeddings = np.load(triple_emb_path)
            else:
                logger.warning(f"Triple embeddings file not found: {triple_emb_path}")
                self.triple_embeddings = None

            triple2idx_map_path = cache_paths['triple2idx_map']
            if os.path.exists(triple2idx_map_path):
                with open(triple2idx_map_path, 'rb') as f:
                    self.triple2idx_map = pickle.load(f)
            else:
                logger.warning(f"triple2idx_map file not found: {triple2idx_map_path}")
                self.triple2idx_map = {}

            entity2idx_map_path = cache_paths['entity2idx_map']
            if os.path.exists(entity2idx_map_path):
                with open(entity2idx_map_path, 'rb') as f:
                    self.entity2idx_map = pickle.load(f)
            else:
                logger.warning(f"entity2idx_map file not found: {entity2idx_map_path}")
                self.entity2idx_map = {}

            logger.info(f"✅ Cache loaded successfully for '{self.subgraph_name}'")
            return True

        except Exception as e:
            logger.warning(f"Failed to load cache for '{self.subgraph_name}': {e}")
            return False

    def _save_faiss_indices_to_cache(self):
        if not self.cache_dir:
            return

        cache_paths = self._get_cache_paths()

        try:
            import pickle

            logger.debug(f"💾 Saving entity FAISS index to cache for '{self.subgraph_name}'...")
            faiss.write_index(self.entity_index, cache_paths['entity_index'])

            with open(cache_paths['entity_map'], 'wb') as f:
                pickle.dump(self.entity_map, f)

            if self.entity2idx_map is not None:
                with open(cache_paths['entity2idx_map'], 'wb') as f:
                    pickle.dump(self.entity2idx_map, f)
            if self.triple_index is not None:
                logger.debug(f"💾 Saving triple FAISS index to cache for '{self.subgraph_name}'...")
                faiss.write_index(self.triple_index, cache_paths['triple_index'])

                with open(cache_paths['triple_map'], 'wb') as f:
                    pickle.dump(self.triple_map, f)
            if self.triple_embeddings is not None:
                np.save(cache_paths['triple_embeddings'], self.triple_embeddings)

            if self.triple2idx_map is not None:
                with open(cache_paths['triple2idx_map'], 'wb') as f:
                    pickle.dump(self.triple2idx_map, f)
            if self.chunk_index is not None:
                logger.debug(f"💾 Saving chunk FAISS index to cache for '{self.subgraph_name}'...")
                faiss.write_index(self.chunk_index, cache_paths['chunk_index'])

                with open(cache_paths['chunk_map'], 'wb') as f:
                    pickle.dump(self.chunk_map, f)
            with open(cache_paths['metadata'], 'wb') as f:
                pickle.dump(self.cache_metadata, f)

            logger.debug(f"✅ Cache saved for '{self.subgraph_name}'")

        except Exception as e:
            logger.warning(f"Failed to save cache for '{self.subgraph_name}': {e}")

    def release_memory(self):
        logger.debug(f"🗑️  Releasing memory for '{self.subgraph_name}'")

        if self.entity_index is not None:
            del self.entity_index
            self.entity_index = None

        if self.triple_index is not None:
            del self.triple_index
            self.triple_index = None

        if self.chunk_index is not None:
            del self.chunk_index
            self.chunk_index = None

        self.entity_map.clear()
        self.triple_map.clear()
        if hasattr(self, 'triple_embeddings'):
            del self.triple_embeddings
            self.triple_embeddings = None
        if hasattr(self, 'entity2idx_map'):
            self.entity2idx_map.clear()
        if hasattr(self, 'triple2idx_map'):
            self.triple2idx_map.clear()
        if hasattr(self, 'subgraph_nodes'):
            self.subgraph_nodes.clear()

        import gc
        gc.collect()

        logger.debug(f"✅ Memory released for '{self.subgraph_name}'")

    def is_memory_loaded(self) -> bool:
        return self.entity_index is not None

    def reload_indices(self):
        if self.is_memory_loaded():
            logger.debug(f"Indices for '{self.subgraph_name}' already loaded, skipping reload")
            return

        logger.info(f"🔄 Reloading FAISS indices for '{self.subgraph_name}'...")

        if not self._load_faiss_indices_from_cache():
            logger.info(f"  Cache not available, rebuilding indices...")
            self._build_entity_faiss_index()
            self._build_triple_faiss_index()
            if self.cache_dir:
                self._save_faiss_indices_to_cache()

        logger.info(f"✅ Indices reloaded for '{self.subgraph_name}'")

