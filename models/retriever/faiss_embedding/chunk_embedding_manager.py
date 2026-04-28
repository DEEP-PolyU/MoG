import os
import pickle
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import faiss

from ...utils import logger
from .embedding_utils import EmbeddingUtils
from .faiss_utils import FAISSUtils

class ChunkEmbeddingManager:

    def __init__(
        self,
        chunk_id2text: Dict[str, str],
        qa_encoder,
        cache_dir: str
    ):
        self.chunk_id2text = chunk_id2text
        self.qa_encoder = qa_encoder
        self.cache_dir = cache_dir
        self.device = EmbeddingUtils.setup_device("cuda")

        os.makedirs(cache_dir, exist_ok=True)
        self.chunk_embeddings = {}
        self.chunk_texts = {}

        self.faiss_index = None
        self.faiss_index_map = {}
        self.gpu_resources = None

        self.chunks_cache_file = os.path.join(cache_dir, "chunk_embeddings.pkl")
        self.index_cache_file = os.path.join(cache_dir, "chunk_faiss_index.pkl")

        logger.warning("Initializing ChunkEmbeddingManager...")

        if self._load_cached_embeddings():
            logger.warning(" Loaded cached chunk embeddings.")
        else:
            logger.warning("Building chunk embeddings from scratch...")
            self._build_chunk_embeddings()
            self._save_embeddings_cache()

        self._build_faiss_index()
        logger.info(f" ChunkEmbeddingManager initialized with {len(self.chunk_embeddings)} chunks")

    def _load_cached_embeddings(self) -> bool:
        logger.warning(f" Try to load {len(self.chunk_embeddings)} chunk embeddings from cache.")
        try:
            if os.path.exists(self.chunks_cache_file):
                with open(self.chunks_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.chunk_embeddings = cache_data['embeddings']
                    self.chunk_texts = cache_data['texts']
                    logger.warning(f" Loaded {len(self.chunk_embeddings)} chunk embeddings from cache.")
                    return True
        except Exception as e:
            logger.warning(f"Failed to load cached chunk embeddings: {e}")
        return False

    def _save_embeddings_cache(self):
        try:
            cache_data = {
                'embeddings': self.chunk_embeddings,
                'texts': self.chunk_texts
            }
            with open(self.chunks_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.warning(f" Saved chunk embeddings to {self.chunks_cache_file}.")
        except Exception as e:
            logger.warning(f"Failed to save chunk embeddings cache: {e}")

    def _build_chunk_embeddings(self):
        logger.warning("Building chunk embeddings ...")
        chunk_ids = list(self.chunk_id2text.keys())
        texts = [self.chunk_id2text[cid] for cid in chunk_ids]

        batch_size = 256
        total = len(chunk_ids)
        done = 0
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_ids = chunk_ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            try:
                batch_emb = self.qa_encoder.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                batch_emb = F.normalize(batch_emb, p=2, dim=1)
                for j, chunk_id in enumerate(batch_ids):
                    self.chunk_embeddings[chunk_id] = batch_emb[j].detach().cpu()
                    self.chunk_texts[chunk_id] = batch_texts[j]
                done += len(batch_ids)
                logger.info(f"Processed {done}/{total} chunks...")
            except Exception as e:
                logger.warning(f"Failed embedding batch {i // batch_size}: {e}")
                for j, chunk_id in enumerate(batch_ids):
                    try:
                        emb = self.qa_encoder.encode(
                            [batch_texts[j]],
                            convert_to_tensor=True,
                            device=self.device,
                            show_progress_bar=False
                        )
                        emb = F.normalize(emb, p=2, dim=1)
                        self.chunk_embeddings[chunk_id] = emb[0].detach().cpu()
                        self.chunk_texts[chunk_id] = batch_texts[j]
                        done += 1
                    except Exception as single_e:
                        logger.warning(f"Failed to embed single chunk {chunk_id}: {single_e}")
        logger.warning(f" Built embeddings for {len(self.chunk_embeddings)} chunks.")

    def _build_faiss_index(self):
        if not self.chunk_embeddings:
            logger.warning("No chunk embeddings found, skipping FAISS index build")
            return

        logger.info("Building FAISS index for chunks...")
        chunk_ids = list(self.chunk_embeddings.keys())
        embeddings = [self.chunk_embeddings[cid].numpy() for cid in chunk_ids]
        emb_matrix = np.array(embeddings).astype('float32')

        self.faiss_index = FAISSUtils.build_faiss_index(
            embeddings=emb_matrix,
            index_type="flat_ip",
            normalize=True,
            use_gpu=True
        )

        self.faiss_index_map = {idx: chunk_id for idx, chunk_id in enumerate(chunk_ids)}
        if torch.cuda.is_available():
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info(" FAISS index is using GPU acceleration.")
            except Exception as e:
                logger.warning(f"Failed to move chunk FAISS index to GPU: {e}")

        logger.info(f" Built chunk FAISS index with {len(chunk_ids)} chunks.")

    def get_chunk_embedding(self, chunk_id: str) -> Optional[torch.Tensor]:
        return self.chunk_embeddings.get(chunk_id)

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        return self.chunk_texts.get(chunk_id)

    def batch_compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.qa_encoder.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def batch_chunk_similarity(
        self,
        query_embedding: torch.Tensor,
        chunk_id_list: List[str]
    ) -> Dict[str, float]:
        similarities = {}
        valid_ids = []
        valid_embeddings = []
        for cid in chunk_id_list:
            emb = self.chunk_embeddings.get(cid)
            if emb is not None:
                valid_ids.append(cid)
                valid_embeddings.append(emb)
        if not valid_embeddings:
            return similarities
        query_tensor = query_embedding.to(self.device).unsqueeze(0)
        chunk_tensor = torch.stack(valid_embeddings).to(self.device)

        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        chunk_tensor = F.normalize(chunk_tensor, p=2, dim=1)

        similarity_scores = torch.mm(query_tensor, chunk_tensor.t())[0]
        for i, cid in enumerate(valid_ids):
            similarities[cid] = similarity_scores[i].cpu().item()
        return similarities

    def faiss_search_chunks(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if self.faiss_index is None:
            logger.warning("FAISS index unavailable.")
            return []

        query_np = query_embedding.detach().cpu().numpy().reshape(1,-1).astype('float32')
        faiss.normalize_L2(query_np)
        scores, indices = FAISSUtils.search(self.faiss_index, query_np, top_k=top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in self.faiss_index_map:
                chunk_id = self.faiss_index_map[idx]
                results.append((chunk_id, float(score)))
        return results

    def encode_query(self, text: str) -> torch.Tensor:
        emb = self.qa_encoder.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        emb = F.normalize(emb, p=2, dim=0 if emb.dim()==1 else 1)
        return emb

    def get_statistics(self) -> Dict[str, Any]:
        total_chunks = len(self.chunk_embeddings)
        emb_dim = 0
        if total_chunks > 0:
            any_emb = next(iter(self.chunk_embeddings.values()))
            emb_dim = any_emb.shape[0]
        return {
            "total_chunks": total_chunks,
            "embedding_dimension": emb_dim,
            "faiss_index_available": self.faiss_index is not None,
            "gpu_accelerated": self.gpu_resources is not None,
            "cached_texts": len(self.chunk_texts)
        }

    def batch_get_chunk_embeddings(self, chunk_id_list: List[str]) -> Dict[str, torch.Tensor]:
        results = {}
        for cid in chunk_id_list:
            emb = self.chunk_embeddings.get(cid)
            if emb is not None:
                results[cid] = emb
        return results
