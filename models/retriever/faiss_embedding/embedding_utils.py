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




class EmbeddingUtils:
    @staticmethod
    def setup_device(device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            device_obj = torch.device("cuda")
            logger.info(f" Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device_obj = torch.device("cpu")
            if device == "cuda":
                logger.info("⚠️ CUDA requested but not available, using CPU")
            else:
                logger.info(" Using CPU")
        return device_obj

