
import json
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any

from ...utils import logger
from ...constructor.construction_utils import GraphIOUtils

class GraphDataLoader:
    def __init__(self, dataset_name: str, json_path: str, config):
        self.json_path = json_path
        self.raw_data = None
        self.graph = None

        logger.info(f"GraphDataLoader initialized for: {json_path}")
        self.graph_IO = GraphIOUtils(dataset_name, config)

    def load(self) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
        logger.info(f"Loading graph data from: {self.json_path}")

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {self.json_path}: {e}")
            raise

        self.graph = self.graph_IO.load_graph_from_json(self.json_path)
        logger.info(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")

        subGraph_nodes = self._extract_subGraph_nodes_from_metadata()

        metadata = {
            'metadata': self.raw_data.get('metadata', {}),
            'mixture_of_graph': self.raw_data.get('mixture_of_graph'),
            'subGraph_nodes': subGraph_nodes,
            'has_mixture_format': 'mixture_of_graph' in self.raw_data,
            'json_path': self.json_path
        }


        if metadata['has_mixture_format']:
            mog = metadata['mixture_of_graph']
            logger.info(f"✅ Mixture of graph data found: "
                       f"{len(mog.get('subGraphs', []))} subGraphs")
        else:
            logger.warning("⚠️ No mixture_of_graph data found in JSON")

        return self.graph, metadata

    def get_mixture_of_graph_data(self) -> Optional[Dict]:
        if self.raw_data is None:
            logger.warning("Data not loaded yet, call load() first")
            return None
        return self.raw_data.get('mixture_of_graph')

    def get_subGraph_nodes(self) -> List[Dict]:
        if self.raw_data is None:
            logger.warning("Data not loaded yet, call load() first")
            return []
        return self.raw_data.get('subGraph_nodes', [])

    def get_metadata(self) -> Dict:
        if self.raw_data is None:
            logger.warning("Data not loaded yet, call load() first")
            return {}
        return self.raw_data.get('metadata', {})

    def _extract_subGraph_nodes_from_metadata(self) -> List[Dict]:
        subGraph_nodes = []

        mog = self.raw_data['mixture_of_graph']

        all_comms = mog.get('subGraphs', {})

        for key, value in all_comms.items():
            if key.startswith('shared_'):
                label_comm = 'shared_subGraph'
            else:
                label_comm = 'expert_subGraph'
            subGraph_nodes.append({
                'label': label_comm,
                'properties': {
                    'subGraph_id': value['node_id'],
                    'node_id': value['node_id'],
                    'member_nodes': value['member_nodes'] ,
                    'keywords': value['keywords'],
                    'description': value['description']
                }
            })

        logger.info(f"✅ Extracted {len(subGraph_nodes)} subGraph nodes from metadata (no graph traversal needed)")
        return subGraph_nodes


    def get_statistics(self) -> Dict[str, Any]:
        if self.raw_data is None or self.graph is None:
            return {}

        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'has_mixture_format': 'mixture_of_graph' in self.raw_data
        }

        if stats['has_mixture_format']:
            mog = self.raw_data['mixture_of_graph']
            stats['subGraphs'] = len(mog.get('subGraphs', []))
            stats['nodes_with_memberships'] = len(mog.get('node_memberships', {}))

            if 'subGraph_statistics' in mog:
                stats['subGraph_statistics'] = mog['subGraph_statistics']

        return stats

