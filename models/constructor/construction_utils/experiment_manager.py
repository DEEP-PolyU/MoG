import json
import os
import time
from typing import Tuple
import networkx as nx
from ...utils import logger

from .serializer import GraphSerializer
from .graph_IO_utils import GraphIOUtils

class ExperimentManager:

    def __init__(self, dataset_name: str, config):
        self.dataset_name = dataset_name
        self.graph_IO = GraphIOUtils(self.dataset_name, config)
        self.graphs_meta_data_dir = config.output.graphs_meta_data_dir

    def save_mog_metadata(self, graph: nx.MultiDiGraph, execution_time: float):
        os.makedirs(self.graphs_meta_data_dir, exist_ok=True)

        traditional_subGraphs = [n for n, d in graph.nodes(data=True)
                                   if d.get('label') == 'subGraph']
        shared_subGraphs = [n for n, d in graph.nodes(data=True)
                              if d.get('label') == 'shared_subGraph']
        expert_subGraphs = [n for n, d in graph.nodes(data=True)
                              if d.get('label') == 'expert_subGraph']

        all_subGraphs = traditional_subGraphs + shared_subGraphs + expert_subGraphs

        subGraph_stats = {
            "total_subGraphs": len(all_subGraphs),
            "traditional_subGraphs": len(traditional_subGraphs),
            "shared_subGraphs": len(shared_subGraphs),
            "expert_subGraphs": len(expert_subGraphs),
            "subGraph_sizes": [],
            "shared_subGraph_sizes": [],
            "expert_subGraph_sizes": [],
            "avg_subGraph_size": 0,
            "max_subGraph_size": 0,
            "min_subGraph_size": float('inf')
        }

        for node in traditional_subGraphs:
            members = graph.nodes[node]["properties"].get("members", [])
            size = len(members)
            subGraph_stats["subGraph_sizes"].append(size)
            subGraph_stats["max_subGraph_size"] = max(subGraph_stats["max_subGraph_size"], size)
            subGraph_stats["min_subGraph_size"] = min(subGraph_stats["min_subGraph_size"], size)

        for node in shared_subGraphs:
            properties = graph.nodes[node].get("properties", {})
            members = properties.get("member_nodes", [])
            size = len(members)
            subGraph_stats["subGraph_sizes"].append(size)
            subGraph_stats["shared_subGraph_sizes"].append(size)
            subGraph_stats["max_subGraph_size"] = max(subGraph_stats["max_subGraph_size"], size)
            subGraph_stats["min_subGraph_size"] = min(subGraph_stats["min_subGraph_size"], size)

        for node in expert_subGraphs:
            properties = graph.nodes[node].get("properties", {})
            members = properties.get("member_nodes", [])
            size = len(members)
            subGraph_stats["subGraph_sizes"].append(size)
            subGraph_stats["expert_subGraph_sizes"].append(size)
            subGraph_stats["max_subGraph_size"] = max(subGraph_stats["max_subGraph_size"], size)
            subGraph_stats["min_subGraph_size"] = min(subGraph_stats["min_subGraph_size"], size)

        if subGraph_stats["subGraph_sizes"]:
            subGraph_stats["avg_subGraph_size"] = sum(subGraph_stats["subGraph_sizes"]) / len(
                subGraph_stats["subGraph_sizes"])
            subGraph_stats["min_subGraph_size"] = min(subGraph_stats["subGraph_sizes"])
        else:
            subGraph_stats["min_subGraph_size"] = 0

        experiment_report = {
            "dataset_name": self.dataset_name,
            "algorithm_name": "Mixture-of-Graph",
            "execution_time": GraphSerializer.make_json_serializable(execution_time),
            "timestamp": GraphSerializer.make_json_serializable(time.time()),
            "graph_stats": {
                "total_nodes": GraphSerializer.make_json_serializable(graph.number_of_nodes()),
                "total_edges": GraphSerializer.make_json_serializable(graph.number_of_edges()),
                "level2_nodes": GraphSerializer.make_json_serializable(
                    len([n for n, d in graph.nodes(data=True) if d.get('level') == 2]))
            },
            "subGraph_stats": GraphSerializer.make_json_serializable(subGraph_stats)
        }

        report_path = f"{self.graphs_meta_data_dir}/{self.dataset_name}_Mixture-of-Graph_{int(time.time())}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_report, f, ensure_ascii=False, indent=2)

        logger.info(f"Experiment report saved: {report_path}")
        logger.info(f"  - SubGraphs: {subGraph_stats['total_subGraphs']}")
        logger.info(f"  - Avg size: {subGraph_stats['avg_subGraph_size']:.2f}")
        logger.info(f"  - Graph will be saved separately by GraphIOUtils")
