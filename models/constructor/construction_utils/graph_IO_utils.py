import networkx as nx
import json
import os
from typing import Tuple, Any, Dict

import time
from ...utils import logger
from .serializer import GraphSerializer

class GraphIOUtils:
    def __init__(self, dataset_name: str, config):
        self.dataset_name = dataset_name
        self.config = config
        self.graphs_dir = config.output.graphs_dir
        self.graphs_processing_dir = config.output.graphs_processing_dir

    def save_graph_to_json(self, graph: nx.MultiDiGraph, json_output_path: str = "") -> None:
        output = self.format_output(graph, self.dataset_name)

        output_dir = os.path.dirname(json_output_path)
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")
        logger.info(f"Saving graph to: {json_output_path}")

        is_demo = (self.dataset_name.lower() == 'demo')

        if is_demo:
            logger.info("Saving demo dataset with pretty-print format (indent=2)...")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                logger.info(f" Demo graph saved with pretty format")
            except TypeError as e:
                logger.warning(f"Standard JSON serialization failed: {e}")
                logger.info("Attempting fallback serialization...")
                output = GraphSerializer.deep_serialize_for_json(output)
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                logger.info("Fallback serialization successful")
        else:
            logger.info("Saving large dataset with compact format (no indent)...")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False)
                logger.info(f" Graph saved with compact format (fast & memory-efficient)")
            except TypeError as e:
                logger.warning(f"Standard JSON serialization failed: {e}")
                logger.info("Attempting fallback serialization...")
                output = GraphSerializer.deep_serialize_for_json(output)
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False)
                logger.info("Fallback serialization successful")

        if "mixture_of_graph" in output:
            mog_info = output["mixture_of_graph"]
            logger.info(f"Graph saved to {json_output_path}")
            logger.info(f"Mixture of Graph info: {mog_info['subGraph_statistics']['shared_count']} shared, "
                       f"{mog_info['subGraph_statistics']['expert_count']} expert subGraphs")
        else:
            logger.info(f"Graph saved to {json_output_path}")


    def format_output(self, graph: nx.MultiDiGraph, dataset_name: str) -> Dict[str, Any]:

        def process_subGraph_data(properties):
            filtered_dict = properties.copy()
            fields_to_remove = ['member_nodes', 'keywords', 'domain', 'description']
            inner_props = filtered_dict['properties'].copy()
            for field in fields_to_remove:
                inner_props.pop(field, None)
            full_properties = properties.get('properties', {})

            return inner_props, full_properties


        def _get_node_data_subGraph(node_data):

            properties = {}

            exclude_keys = {"type"}
            for key, value in node_data.items():
                if key not in exclude_keys:
                    properties[key] = GraphSerializer.make_json_serializable(value)

            filtered_subGraph_properties, full_subGraph_properties = process_subGraph_data(properties)
            filtered_subGraph_node_content = {
                "label": node_data["label"],
                "properties": filtered_subGraph_properties
            }
            return filtered_subGraph_node_content, full_subGraph_properties



        def _get_node_data_entity(node_id, node_data):
            if "label" in node_data and "properties" in node_data:
                properties = node_data["properties"].copy() if isinstance(node_data["properties"], dict) else {}
                if "node_id" not in properties:
                    properties["node_id"] = node_id
                return {
                    "label": node_data["label"],
                    "properties": GraphSerializer.make_json_serializable(properties)
                }

            label = node_data.get("type", node_data.get("label", "entity"))

            properties = {}
            exclude_keys = {"label", "properties", "type"}

            for key, value in node_data.items():
                if key not in exclude_keys:
                    properties[key] = GraphSerializer.make_json_serializable(value)

            if "node_id" not in properties:
                properties["node_id"] = node_id

            return {
                "label": label,
                "properties": properties
            }

        relationships = []
        all_subGraphs = {}
        for u, v, data in graph.edges(data=True):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            if u_data["label"] in ["shared_subGraph", "expert_subGraph"]:
                filtered_subGraph_node_content, full_subGraph_properties = _get_node_data_subGraph(u_data)
                u_node_data = filtered_subGraph_node_content
                all_subGraphs[u] = full_subGraph_properties
            else:
                u_node_data = _get_node_data_entity(u, u_data)

            if v_data["label"] in ["shared_subGraph", "expert_subGraph"]:
                filtered_subGraph_node_content, full_subGraph_properties = _get_node_data_subGraph(v_data)
                v_node_data = filtered_subGraph_node_content
                all_subGraphs[v] = full_subGraph_properties
            else:
                v_node_data = _get_node_data_entity(v, v_data)

            relationship = {
                "start_node": u_node_data,
                "relation": GraphSerializer.make_json_serializable(data.get("relation", "related_to")),
                "end_node": v_node_data,
            }
            relationships.append(relationship)

        output = {
            "relationships": relationships,
            "metadata": {
                "total_nodes": graph.number_of_nodes(),
                "total_edges": graph.number_of_edges(),
                "dataset_name": dataset_name
            }
        }
        if hasattr(graph, 'graph') and 'shared_expert_result' in graph.graph:
            shared_expert_result = graph.graph['shared_expert_result']

            output["mixture_of_graph"] = {
                "algorithm_type": "mixture_of_graph",
                "subGraphs": all_subGraphs,
                "node_memberships": GraphSerializer.make_json_serializable(shared_expert_result.get('node_memberships', {})),
                "detection_params": GraphSerializer.make_json_serializable(shared_expert_result.get('detection_params', {})),
                "multi_dimensional_features": GraphSerializer.make_json_serializable(shared_expert_result.get('multi_dimensional_features', {})),
                "detection_time": GraphSerializer.make_json_serializable(shared_expert_result.get('detection_time')),
                "algorithm_version": shared_expert_result.get('algorithm_version', 'unknown'),
                "subGraph_statistics": {
                    "total_subGraphs": len(shared_expert_result.get('shared_subGraphs', [])) + len(shared_expert_result.get('expert_subGraphs', [])),
                    "shared_count": len(shared_expert_result.get('shared_subGraphs', [])),
                    "expert_count": len(shared_expert_result.get('expert_subGraphs', []))
                }
            }

        node_type_stats = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', node_data.get('label', 'unknown'))
            node_type_stats[node_type] = node_type_stats.get(node_type, 0) + 1

        output["metadata"]["node_type_statistics"] = node_type_stats

        subGraph_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['shared_subGraph', 'expert_subGraph']:
                subGraph_info = {
                    "node_id": node_id,
                    "type": node_data.get('type'),
                    "subGraph_id": node_data.get('subGraph_id'),
                    "member_count": len(node_data.get('member_nodes', [])),
                    "keywords": node_data.get('keywords', []),
                    "description": node_data.get('description', '')
                }

                if node_data.get('type') == 'expert_subGraph':
                    subGraph_info["domain"] = node_data.get('domain')
                    subGraph_info["auto_discovered"] = node_data.get('auto_discovered', False)

                subGraph_nodes.append(subGraph_info)

        if subGraph_nodes:
            output["subGraph_nodes"] = GraphSerializer.make_json_serializable(subGraph_nodes)

        return output

    def stringify(self, item):
        if isinstance(item, dict):
            return ', '.join(f"{k}: {v}" for k, v in sorted(item.items()))
        return str(item)
    def load_graph_from_json(self, input_path: str) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            relationships = data
            graph_attributes = {}
        elif isinstance(data, dict):
            relationships = data.get("relationships", [data])
            graph_attributes = {k: v for k, v in data.items() if k != "relationships"}
            for key, value in graph_attributes.items():
                graph.graph[key] = value
        else:
            raise ValueError(f"❌ Unsupported JSON structure in {input_path}: expected list or dict")

        added_nodes = set()

        for idx, rel in enumerate(relationships):
            start_node_data = rel["start_node"]
            end_node_data = rel["end_node"]
            relation = rel["relation"]

            start_node_id = start_node_data["properties"]["node_id"]

            if start_node_id not in added_nodes:
                self.add_node_to_graph_strict(graph, start_node_id, start_node_data)
                added_nodes.add(start_node_id)

            end_node_id = end_node_data["properties"]["node_id"]

            if end_node_id not in added_nodes:
                self.add_node_to_graph_strict(graph, end_node_id, end_node_data)
                added_nodes.add(end_node_id)

            graph.add_edge(start_node_id, end_node_id, relation=relation)

        logger.info(f" Successfully loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def assign_unique_node_ids(self, graph: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, Dict[str, str]]:
        PRESERVE_LEVELS = {4}
        counters = {}
        id_mapping = {}
        name_to_newid = {}
        new_graph = nx.MultiDiGraph()

        for old_node_id, node_data in graph.nodes(data=True):
            name = node_data["properties"]["name"]
            if isinstance(name, dict):
                name = json.dumps(name, sort_keys=True)
            else:
                name = str(name)
            level = node_data["level"]
            label = node_data["label"]

            if level in PRESERVE_LEVELS:
                new_node_id = old_node_id
                id_mapping[old_node_id] = new_node_id
                new_graph.add_node(new_node_id, **node_data)
                continue

            if name in name_to_newid:
                new_node_id = name_to_newid[name]
                id_mapping[old_node_id] = new_node_id
            else:
                if label not in counters:
                    counters[label] = 0
                new_node_id = f"{label}_{counters[label]}"
                counters[label] += 1

                name_to_newid[name] = new_node_id
                id_mapping[old_node_id] = new_node_id
                new_graph.add_node(new_node_id, **node_data)

        for u, v, k, edge_data in graph.edges(keys=True, data=True):
            new_u = id_mapping[u]
            new_v = id_mapping[v]
            new_graph.add_edge(new_u, new_v, key=k, **edge_data)

        return new_graph, id_mapping

    def save_preprocessing_graph(self, graph: nx.MultiDiGraph, time_start):
        os.makedirs(f"{self.graphs_processing_dir}", exist_ok=True)

        logger.warning(f"[DIAG-5] ===== FINAL GRAPH STATE =====")
        logger.warning(f"[DIAG-5] Before saving - Graph stats:")
        logger.warning(f"[DIAG-5]   - Nodes: {graph.number_of_nodes()}")
        logger.warning(f"[DIAG-5]   - Edges: {graph.number_of_edges()}")

        node_types = {}
        for node_id, node_data in graph.nodes(data=True):
            label = node_data.get('label', 'unknown')
            node_types[label] = node_types.get(label, 0) + 1
        logger.warning(f"[DIAG-5]   - Node types: {node_types}")

        if graph.number_of_edges() == 0 and graph.number_of_nodes() > 0:
            logger.error(f"[DIAG-5] ❌ CRITICAL: Graph has nodes but NO edges!")

            sample_nodes = list(graph.nodes(data=True))[:5]
            for node_id, node_data in sample_nodes:
                properties = node_data.get('properties', {})
                if isinstance(properties, dict):
                    prop_str = f"name={properties.get('name')}, chunk_id={properties.get('chunk id')}"
                else:
                    prop_str = str(properties)
                logger.error(
                    f"[DIAG-5]   Sample node: {node_id}, label={node_data.get('label')}, properties={prop_str}")

            entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'entity']
            attr_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'attribute']
            logger.error(f"[DIAG-5]   - Entity nodes: {len(entity_nodes)}")
            logger.error(f"[DIAG-5]   - Attribute nodes: {len(attr_nodes)}")
            logger.error(f"[DIAG-5]   ⚠️ All nodes are isolated (no connections)!")
        else:
            sample_edges = list(graph.edges(data=True))[:3]
            for u, v, edge_data in sample_edges:
                logger.warning(f"[DIAG-5]   Sample edge: {u} → {v}, relation={edge_data.get('relation')}")

        logger.warning("Reassigning IDs for Level 1/2/3 nodes (preserving Level 4 subGraph nodes)...")
        graph_with_ids, id_mapping = self.assign_unique_node_ids(graph)
        logger.warning(
            f"ID assignment completed. Preserved {sum(1 for old, new in id_mapping.items() if old == new)} subGraph nodes.")

        graph_path = f"{self.graphs_processing_dir}/{self.dataset_name}_preprocessing.json"
        self.save_graph_to_json(graph_with_ids, graph_path)

        metadata = {
            "dataset_name": self.dataset_name,
            "construction time": time.time() - time_start,
            "nodes_count": graph_with_ids.number_of_nodes(),
            "edges_count": graph_with_ids.number_of_edges(),
            "levels_completed": [1, 2]
        }

        metadata_path = f"{self.graphs_processing_dir}/processing_metadata/{self.dataset_name}_preprocessing_metadata.json"
        os.makedirs(f"{self.graphs_processing_dir}/processing_metadata", exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.warning(f"Preprocessing graph saved:")
        logger.warning(f"  - Graph: {graph_path}")
        logger.warning(f"  - Metadata: {metadata_path}")
        logger.warning(f"  - Nodes: {metadata['nodes_count']}, Edges: {metadata['edges_count']}")


    def load_preprocessed_graph(self) -> Tuple[nx.MultiDiGraph, int]:
        graph_path = f"{self.graphs_processing_dir}/{self.dataset_name}_preprocessing.json"

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Preprocessing graph not found: {graph_path}")

        graph = self.load_graph_from_json(graph_path)

        logger.warning(f"Preprocessing graph loaded:")
        logger.warning(f"  - Nodes: {graph.number_of_nodes()}")
        logger.warning(f"  - Edges: {graph.number_of_edges()}")
        return graph

    def add_node_to_graph_strict(self, graph: nx.MultiDiGraph, node_id: str, node_data: dict):
        if "label" not in node_data:
            raise ValueError(
                f"❌ ERROR for node '{node_id}': Missing 'label' field. "
                f"Node data: {node_data}"
            )

        label = node_data["label"]

        node_attrs = {"label": label}

        if "properties" in node_data:
            properties = node_data["properties"].copy()
            if "node_id" not in properties:
                if label in ["shared_subGraph", "expert_subGraph", "subGraph"]:
                    if "subGraph_id" in properties:
                        properties["node_id"] = properties["subGraph_id"]
                    else:
                        raise ValueError(
                            f"❌ ERROR for subGraph node '{node_id}': "
                            f"Missing both 'node_id' and 'subGraph_id' in properties."
                        )
                else:
                    raise ValueError(
                        f"❌ ERROR for node '{node_id}': Missing 'node_id' in properties. "
                        f"Properties: {properties}"
                    )

            node_attrs["properties"] = properties
        else:
            raise ValueError(
                f"❌ ERROR for node '{node_id}': Missing 'properties' field. "
                f"All nodes must have properties."
            )

        level_mapping = {
            "attribute": 1,
            "entity": 2,
            "keyword": 3,
            "subGraph": 4,
            "shared_subGraph": 4,
            "expert_subGraph": 4,
        }

        node_attrs["level"] = level_mapping.get(label, 2)

        graph.add_node(node_id, **node_attrs)


