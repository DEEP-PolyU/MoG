import json
import os
import time
from typing import Dict, List, Tuple, Optional
import networkx as nx
from ...utils import logger


class CheckpointManager:

    def __init__(self, config, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.checkpoints_dir = os.path.join(config.output.checkpoints_dir, dataset_name)
        self.progress_file = os.path.join(self.checkpoints_dir, "progress.json")

        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def get_batch_dir(self, batch_id: int) -> str:
        return os.path.join(self.checkpoints_dir, f"batch_{batch_id}")

    def save_checkpoint(
            self,
            batch_id: int,
            graph: nx.MultiDiGraph,
            chunks: Dict[str, str],
            batch_info: Dict,
            graph_io_utils
    ):
        batch_dir = self.get_batch_dir(batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        graph_path = os.path.join(batch_dir, "graph_checkpoint.json")
        logger.info(f"Saving graph checkpoint for batch {batch_id}...")
        graph_io_utils.save_graph_to_json(graph, graph_path)

        chunks_path = os.path.join(batch_dir, "chunks_checkpoint.txt")
        logger.info(f"Saving chunks checkpoint for batch {batch_id}...")
        self._save_chunks(chunks, chunks_path)

        metadata_path = os.path.join(batch_dir, "metadata.json")
        batch_metadata = {
            "batch_id": batch_id,
            "batch_start_idx": batch_info["start_idx"],
            "batch_end_idx": batch_info["end_idx"],
            "documents_count": batch_info["documents_count"],
            "chunks_count": len(chunks),
            "nodes_count": graph.number_of_nodes(),
            "edges_count": graph.number_of_edges(),
            "processing_time": batch_info["processing_time"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "completed"
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f" Checkpoint saved for batch {batch_id}: "
                    f"{batch_metadata['nodes_count']} nodes, "
                    f"{batch_metadata['edges_count']} edges, "
                    f"{batch_metadata['chunks_count']} chunks")

    def load_checkpoint(
            self,
            batch_id: int,
            graph_io_utils
    ) -> Tuple[nx.MultiDiGraph, Dict[str, str], Dict]:
        batch_dir = self.get_batch_dir(batch_id)

        if not os.path.exists(batch_dir):
            raise FileNotFoundError(f"Checkpoint not found for batch {batch_id}")

        graph_path = os.path.join(batch_dir, "graph_checkpoint.json")
        logger.info(f"Loading graph from checkpoint batch {batch_id}...")
        graph = graph_io_utils.load_graph_from_json(graph_path)

        chunks_path = os.path.join(batch_dir, "chunks_checkpoint.txt")
        logger.info(f"Loading chunks from checkpoint batch {batch_id}...")
        chunks = self._load_chunks(chunks_path)

        metadata_path = os.path.join(batch_dir, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.info(f" Loaded checkpoint for batch {batch_id}: "
                    f"{metadata['nodes_count']} nodes, "
                    f"{metadata['edges_count']} edges, "
                    f"{metadata['chunks_count']} chunks")

        return graph, chunks, metadata

    def save_progress(self, progress_data: Dict):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Progress saved: batch {progress_data['current_batch']}/{progress_data['total_batches']}")

    def load_progress(self) -> Optional[Dict]:
        if not os.path.exists(self.progress_file):
            return None

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logger.info(f"📂 Found existing progress: "
                        f"batch {progress['current_batch']}/{progress['total_batches']}")
            return progress
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None

    def check_resume_capability(self) -> Tuple[bool, Optional[Dict]]:
        progress = self.load_progress()
        logger.warning(progress)
        if not progress:
            return False, None

        if progress["status"] == "completed":
            logger.info(" All batches already completed!")
            return False, progress

        completed_batches = progress.get("completed_batches", [])[-5:]
        for batch_id in completed_batches:
            batch_dir = self.get_batch_dir(batch_id)
            if not os.path.exists(batch_dir):
                logger.warning(f"⚠️ Checkpoint missing for completed batch {batch_id}")
                return False, progress

        logger.info(f" Can resume from batch {progress['current_batch']}")
        return True, progress

    def _save_chunks(self, chunks: Dict[str, str], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for chunk_id, chunk_text in chunks.items():
                f.write(f"id: {chunk_id}\tChunk: {chunk_text}\n")

    def _load_chunks(self, file_path: str) -> Dict[str, str]:
        chunks = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and "\t" in line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2 and parts[0].startswith("id: ") and parts[1].startswith("Chunk: "):
                        chunk_id = parts[0][4:]
                        chunk_text = parts[1][7:]
                        chunks[chunk_id] = chunk_text
        return chunks

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        import re
        import shutil

        try:
            all_entries = os.listdir(self.checkpoints_dir)
        except FileNotFoundError:
            logger.debug(f"Checkpoints directory not found: {self.checkpoints_dir}")
            return

        batch_dirs = []
        for entry in all_entries:
            match = re.match(r'batch_(\d+)', entry)
            if match:
                batch_id = int(match.group(1))
                full_path = os.path.join(self.checkpoints_dir, entry)
                if os.path.isdir(full_path):
                    batch_dirs.append((batch_id, full_path))

        batch_dirs.sort(key=lambda x: x[0])

        total_batches = len(batch_dirs)

        if total_batches <= keep_last_n:
            logger.debug(f"📦 Total {total_batches} checkpoints, no cleanup needed (keep_last_n={keep_last_n})")
            return

        to_delete_count = total_batches - keep_last_n
        batches_to_delete = batch_dirs[:to_delete_count]
        batches_to_keep = batch_dirs[to_delete_count:]

        deleted_count = 0
        for batch_id, batch_path in batches_to_delete:
            try:
                shutil.rmtree(batch_path)
                deleted_count += 1
                logger.debug(f"Deleted checkpoint: batch_{batch_id}")
            except Exception as e:
                logger.warning(f"Failed to delete batch_{batch_id}: {e}")

        if deleted_count > 0:
            logger.info(f"🗑️ Cleaned up {deleted_count} old checkpoint(s)")
            logger.info(f"📦 Kept latest {len(batches_to_keep)} checkpoints: "
                       f"batch_{batches_to_keep[0][0]} to batch_{batches_to_keep[-1][0]}")

    def cleanup_checkpoints(self):
        import shutil
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)
            logger.info(f"🗑️ Cleaned up checkpoints directory: {self.checkpoints_dir}")