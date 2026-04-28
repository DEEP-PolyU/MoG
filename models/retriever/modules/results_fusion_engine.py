
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from ...utils import logger

class ResultsFusionEngine:
    def __init__(self, config, chunk_embedding_manager, entity_embedding_manager):
        self.config = config
        self.chunk_embedding_manager = chunk_embedding_manager
        self.entity_embedding_manager = entity_embedding_manager
        model_name = config.embeddings.model_name if config else 'all-MiniLM-L6-v2'
        self.qa_encoder = SentenceTransformer(model_name)
        logger.info("🔀 Initialized ResultsFusionEngine")

    def fuse_results(
            self,
            entity_results: Dict[str, List[Dict]],
            chunk_results: Dict[str, List[Dict]],
            query_embedding,
            chunk_id2text: Dict[str, str],
            final_top_k: int = 5
    ) -> (List[str], List[Any]):

        logger.info("🔀 Starting chunk-level results fusion with query-based re-ranking...")
        all_entities = set()
        chunk_dict = {}
        triple_dict = {}
        triple_path_list = []

        def add_chunk_retrieval_result_to_chunk_dict(result, source):
            chunk_id = result.get('chunk_id')
            if chunk_id not in chunk_id2text:
                raise KeyError(f"chunk_id {chunk_id} not found in chunk_id2text mapping!")

            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = {
                    'chunk_text': chunk_id2text[chunk_id],
                    'sources': source
                }

        def add_entity_retrieval_result_to_chunk_dict(result, source):
            entity_id = result.get('entity_id')
            if 'entity' in entity_id:
                all_entities.add(entity_id)
            chunk_id = result.get('chunk_id')
            if not chunk_id:
                raise ValueError(f"Result missing chunk_id: {result}")

            if chunk_id not in chunk_id2text:
                raise KeyError(f"chunk_id {chunk_id} not found in chunk_id2text mapping!")

            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = {
                    'chunk_text': chunk_id2text[chunk_id],
                    'sources': source
                }

            triple = result.get('triple', None)
            if triple:
                if triple not in triple_dict:

                    triple_similarity = result.get('triple_similarity', 0.0)
                    triple_dict[triple] = {
                        'target_entity': result.get('entity_id'),
                        'triple_similarity': triple_similarity,
                        'chunk_id': chunk_id,
                        'sources': source
                    }

            triple_path = result.get('triple_path', None)
            if triple_path:
                path_str = " -> ".join(['({}, {}, {})'.format(*triple) for triple in triple_path])
                if path_str not in triple_dict:
                    triple_path_list.append(path_str)

        for subGraph_id, results in entity_results.items():
            for result in results:
                add_entity_retrieval_result_to_chunk_dict(result, subGraph_id)

        for subGraph_id, results in chunk_results.items():
            for result in results:
                add_chunk_retrieval_result_to_chunk_dict(result, subGraph_id)

        logger.info(f"  Collected {len(chunk_dict)} unique chunks (strict deduplication)")

        chunk_ids = list(chunk_dict.keys())
        valid_chunk_ids = []
        for chunk_id in chunk_ids:
            chunk_embed = self.chunk_embedding_manager.get_chunk_embedding(chunk_id)
            if chunk_embed is None:
                logger.error(f"Embedding not found for chunk_id={chunk_id}!")
            else:
                valid_chunk_ids.append(chunk_id)

        logger.info(f"  {len(valid_chunk_ids)} chunks with available embedding.")

        chunk_scores = []
        if valid_chunk_ids:
            similarity_map = self.chunk_embedding_manager.batch_chunk_similarity(
                query_embedding=query_embedding,
                chunk_id_list=valid_chunk_ids
            )
            for chunk_id in valid_chunk_ids:
                if chunk_id in similarity_map:
                    score = similarity_map[chunk_id]
                    chunk_info = chunk_dict[chunk_id]
                    chunk_scores.append((chunk_id, score, chunk_info))
                else:
                    logger.warning(f"Missing similarity score for chunk_id={chunk_id}.")
        else:
            logger.warning("No chunk embeddings available, returning empty result.")

        sorted_chunks = sorted(
            chunk_scores,
            key=lambda x: x[1], reverse=True
        )[:final_top_k]

        final_chunk_ids = [chunk_id for chunk_id, _, _ in sorted_chunks]

        logger.info(f"✅ Fusion completed: {len(final_chunk_ids)} chunks (Top-{final_top_k})")
        final_entity_ids = list(all_entities)
        for i, (chunk_id, score, chunk_info) in enumerate(sorted_chunks):
            logger.info(
                f"  {i + 1}. chunk={chunk_id}, chunk similarity={score:.4f}, sources={chunk_info['sources']}."
            )

        sorted_triple_list = sorted(
            triple_dict.keys(),
            key=lambda triple: triple_dict[triple]['triple_similarity'],
            reverse=True
        )[:final_top_k]

        for i, triple in enumerate(sorted_triple_list):
            logger.info(
                f"  {i + 1}. triple={triple}, triple_similarity={triple_dict[triple]['triple_similarity']:.4f}, sources={triple_dict[triple]['sources']}, "
            )
        return final_chunk_ids, sorted_triple_list, triple_path_list, final_entity_ids

    def get_fusion_stats(
        self,
        entity_results: Dict[str, List[Dict]],
        final_chunk_ids: List[str],
        final_triples: List[Any],
        final_triple_paths:List[Any]
    ) -> Dict[str, Any]:
        stats = {
            'retrieved_entities': sum(len(r) for r in entity_results.values()),
            'final_chunk_count': len(final_chunk_ids),
            'final_triple_count': len(final_triples),
            'final_triple_path_count': len(final_triple_paths)
        }

        return stats