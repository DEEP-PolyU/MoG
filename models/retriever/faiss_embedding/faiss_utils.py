
import json
import os
import time
from typing import Dict, List, Set, Tuple

import faiss
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from ...utils import logger



class FAISSUtils:

    @staticmethod
    def build_faiss_index(
            embeddings: np.ndarray,
            index_type: str = "flat_ip",
            normalize: bool = True,
            use_gpu: bool = True,
            gpu_device: int = 0
    ) -> faiss.Index:
        embeddings = embeddings.astype('float32')

        if normalize:
            faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        if index_type == "flat_ip":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "flat_l2":
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        index.add(embeddings)

        if use_gpu and torch.cuda.is_available():
            index = FAISSUtils.move_index_to_gpu(index, gpu_device)

        logger.info(f" Built FAISS index: {embeddings.shape[0]} vectors, dim={dimension}")
        return index

    @staticmethod
    def move_index_to_gpu(
            index: faiss.Index,
            device_id: int = 0
    ) -> faiss.Index:
        try:
            gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, device_id, index)
            logger.info(f" Index moved to GPU:{device_id}")
            return gpu_index
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")
            return index

    @staticmethod
    def search(
            index: faiss.Index,
            query_embedding: np.ndarray,
            top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)

        distances, indices = index.search(query_vec, top_k)
        return distances[0], indices[0]