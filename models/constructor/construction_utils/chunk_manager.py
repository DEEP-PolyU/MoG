
import json
import os
import time
from typing import Tuple
from typing import Dict, List, Tuple
import networkx as nx
import threading
import nanoid
import tiktoken

from ...utils import logger

class ChunkManager:

    def __init__(self, config, dataset_name: str, datasets_no_chunk: List[str], use_chunks_cache: bool = True):
        self.chunks_dir = config.output.chunks_dir
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.dataset_name = dataset_name
        self.chunks_file = f"{config.output.chunks_dir}/{dataset_name}.txt"
        self.chunks_cached = False
        self.use_chunks_cache = use_chunks_cache
        self.all_chunks = {}

        self.datasets_no_chunk = datasets_no_chunk

    def chunk_text(self, text, all_chunks: Dict[str, str], lock) -> Tuple[List[str], Dict[str, str]]:
        if self.dataset_name in self.datasets_no_chunk:
            chunks = [f"{text.get('title', '')} {text.get('text', '')}".strip()
                      if isinstance(text, dict) else str(text)]
        else:
            chunks = [str(text)]

        chunk_id2chunks = {}
        for chunk in chunks:
            try:
                chunk_id = nanoid.generate(size=8)
                chunk_id2chunks[chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Failed to generate chunk id with nanoid: {type(e).__name__}: {e}")

        with lock:
            all_chunks.update(chunk_id2chunks)

        return chunks, chunk_id2chunks

    def clean_text(self, text: str) -> str:
        if not text:
            return "[EMPTY_TEXT]"

        safe_chars = {
            *" .:,!?()-+="
        }
        cleaned = "".join(
            char for char in text
            if char.isalnum() or char.isspace() or char in safe_chars
        ).strip()

        return cleaned if cleaned else "[EMPTY_AFTER_CLEANING]"

    def calculate_tokens(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def check_chunks_cache(self, corpus_path: str) -> bool:
        try:
            if not os.path.exists(self.chunks_file):
                logger.info(f"Chunks cache file not found: {self.chunks_file}")
                return False

            if os.path.getsize(self.chunks_file) == 0:
                logger.info(f"Chunks cache file is empty: {self.chunks_file}")
                return False

            if os.path.exists(corpus_path):
                corpus_mtime = os.path.getmtime(corpus_path)
                chunks_mtime = os.path.getmtime(self.chunks_file)

                if corpus_mtime > chunks_mtime:
                    logger.info(f"Corpus file is newer than chunks cache, need to re-chunk")
                    return False

            logger.info(f"Valid chunks cache found: {self.chunks_file}")
            return True

        except Exception as e:
            logger.warning(f"Error checking chunks cache: {e}")
            return False

    def load_chunks_from_cache(self) -> bool:
        try:
            self.all_chunks = {}

            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "\t" in line:
                        parts = line.split("\t", 1)
                        if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                            chunk_id = parts[0][4:]
                            chunk_text = parts[1][7:]
                            self.all_chunks[chunk_id] = chunk_text

            self.chunks_cached = True
            logger.info(f"Successfully loaded {len(self.all_chunks)} chunks from cache")
            return True

        except Exception as e:
            logger.error(f"Failed to load chunks from cache: {e}")
            self.all_chunks = {}
            self.chunks_cached = False
            return False

    def save_chunks_to_file(self):
        existing_data = {}
        if os.path.exists(self.chunks_file):
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "\t" in line:
                            parts = line.split("\t", 1)
                            if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                                chunk_id = parts[0][4:]
                                chunk_text = parts[1][7:]
                                existing_data[chunk_id] = chunk_text
            except Exception as e:
                logger.warning(f"Failed to parse existing chunks from {self.chunks_file}: {type(e).__name__}: {e}")

        all_data = {**existing_data, **self.all_chunks}

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in all_data.items():
                f.write(f"id: {chunk_id}\tChunk: {chunk_text}\n")

        logger.info(f"Chunk data saved to {self.chunks_file} ({len(all_data)} chunks)")
