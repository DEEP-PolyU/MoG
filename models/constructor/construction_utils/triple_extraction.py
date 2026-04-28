
from typing import Dict, List, Tuple, TYPE_CHECKING

import networkx as nx
from models.utils import logger

if TYPE_CHECKING:
    from models.constructor.mixture_of_graph_construction import MixtureOfGraphConstructor


class TripleExtractionUtils:

    def __init__(self, MoG_builder: 'MixtureOfGraphConstructor' = None):
        self.MoG_builder = MoG_builder

    def validate_triple_format(self, triple: List) -> Tuple:
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return ()

            return tuple(triple)
        except Exception as e:
            return ()

    def find_or_create_entity(self, entity_name: str, chunk_id: str, nodes_to_add: List, entity_type: str = None) -> str:

        with self.MoG_builder.lock:

            key_entity = f"{entity_name}-{str(chunk_id)}"
            if key_entity in self.MoG_builder.entity_to_id:
                entity_node_id = self.MoG_builder.entity_to_id[key_entity]
            else:
                entity_node_id = f"entity_{self.MoG_builder.node_counter}"
                self.MoG_builder.entity_to_id[key_entity] = entity_node_id
                self.MoG_builder.node_counter += 1

                properties = {"name": entity_name, "chunk id": chunk_id}
                if entity_type:
                    properties["schema_type"] = entity_type

                nodes_to_add.append((
                    entity_node_id,
                    {
                        "label": "entity",
                        "properties": properties,
                        "level": 2
                    }
                ))

        return entity_node_id

    def process_attributes(self, extracted_attr: Dict, chunk_id: str, graph: nx.MultiDiGraph,
                          entity_types: Dict = None) -> Tuple[List, List]:
        nodes_to_add = []
        edges_to_add = []

        logger.debug(f"[DIAG-ATTR] Processing {len(extracted_attr)} entities with attributes")

        for entity, attributes in extracted_attr.items():
            logger.debug(f"[DIAG-ATTR] Entity '{entity}' has {len(attributes)} attributes")
            for attr in attributes:
                key_attribute = f"{attr}-{str(chunk_id)}"
                with self.MoG_builder.lock:
                    if key_attribute in self.MoG_builder.attribute_to_id:
                        attr_node_id = self.MoG_builder.attribute_to_id[key_attribute]
                    else:
                        attr_node_id = f"attr_{self.MoG_builder.node_counter}"
                        self.MoG_builder.attribute_to_id[key_attribute] = attr_node_id
                        self.MoG_builder.node_counter += 1
                        nodes_to_add.append((
                            attr_node_id,
                            {
                                "label": "attribute",
                                "properties": {"name": attr, "chunk id": chunk_id},
                                "level": 1,
                            }
                        ))

                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self.find_or_create_entity(
                    entity, chunk_id, nodes_to_add, entity_type
                )

                edges_to_add.append((entity_node_id, attr_node_id, "has_attribute"))
                logger.debug(f"[DIAG-ATTR] Created edge: {entity_node_id} -> {attr_node_id}")

        logger.debug(f"[DIAG-ATTR] Total: {len(nodes_to_add)} nodes, {len(edges_to_add)} edges")

        return nodes_to_add, edges_to_add

    def process_triples(self, extracted_triples: List, chunk_id: str, graph: nx.MultiDiGraph,
                       entity_types: Dict = None) -> Tuple[List, List]:
        nodes_to_add = []
        edges_to_add = []

        logger.debug(f"[DIAG-TRIPLE] Processing {len(extracted_triples)} triples")

        for i, triple in enumerate(extracted_triples):
            validated_triple = self.validate_triple_format(triple)
            if not validated_triple or len(validated_triple) != 3:
                logger.debug(f"[DIAG-TRIPLE] Triple {i} invalid: {triple}")
                continue

            subj, pred, obj = validated_triple

            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None

            subj_node_id = self.find_or_create_entity(subj, chunk_id, nodes_to_add, subj_type)
            obj_node_id = self.find_or_create_entity(obj, chunk_id, nodes_to_add, obj_type)

            edges_to_add.append((subj_node_id, obj_node_id, pred))
            logger.debug(f"[DIAG-TRIPLE] Triple {i}: {subj_node_id} -> {obj_node_id} [{pred}]")

        logger.debug(f"[DIAG-TRIPLE] Total: {len(nodes_to_add)} nodes, {len(edges_to_add)} edges")

        return nodes_to_add, edges_to_add


    def deduplicate_triples(self, graph: nx.MultiDiGraph) -> None:
        logger.info(f"[DEDUP] ===== BEFORE DEDUPLICATION =====")
        logger.info(f"[DEDUP] Graph state: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

        nodes_data = list(graph.nodes(data=True))
        edges_data = list(graph.edges(keys=True, data=True))

        logger.info(f"[DEDUP] Saved {len(nodes_data)} nodes and {len(edges_data)} edges")

        graph.clear()

        for node, node_data in nodes_data:
            graph.add_node(node, **node_data)
        seen_triples = set()
        added_edges = 0
        duplicate_edges = 0

        for u, v, key, data in edges_data:
            relation = data.get('relation')
            if (u, v, relation) not in seen_triples:
                seen_triples.add((u, v, relation))
                graph.add_edge(u, v, **data)
                added_edges += 1
            else:
                duplicate_edges += 1

        logger.info(f"[DEDUP] ===== AFTER DEDUPLICATION =====")
        logger.info(f"[DEDUP] Graph state: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
        logger.info(f"[DEDUP] Added {added_edges} unique edges, removed {duplicate_edges} duplicates")
