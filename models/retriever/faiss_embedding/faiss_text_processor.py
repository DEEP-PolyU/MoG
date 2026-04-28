
import networkx as nx
from typing import List, Tuple


class FAISSTextProcessor:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

        self.name_to_id = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            name = self._extract_node_name_from_data(node_data)
            if name:
                self.name_to_id[name] = node_id

    def get_node_text(self, node_id: str) -> str:
        if node_id not in self.graph.nodes:
            return ""

        data = self.graph.nodes[node_id]
        name, description = self._extract_node_info(data)
        return self._format_node_text(name, description)

    def _extract_node_info(self, node_data: dict) -> Tuple[str, str]:
        def normalize_field(field) -> str:
            if not field:
                return ''
            if isinstance(field, list):
                return ", ".join(str(item) for item in field)
            return str(field).strip()

        if 'properties' in node_data and isinstance(node_data['properties'], dict):
            name = normalize_field(node_data['properties'].get('name'))
            description = normalize_field(node_data['properties'].get('description'))
        else:
            name = normalize_field(node_data.get('name'))
            description = normalize_field(node_data.get('description'))

        name = name if name else 'none'
        description = description if description else 'none'

        return name, description

    def _format_node_text(self, name: str, description: str) -> str:
        return f"{name},{description}".strip()

    def _extract_node_name(self, node_id: str) -> str:
        if node_id not in self.graph.nodes:
            return node_id

        node_data = self.graph.nodes[node_id]
        name, _ = self._extract_node_info(node_data)
        return name

    def _extract_node_name_from_data(self, node_data: dict) -> str:
        if 'properties' in node_data and isinstance(node_data['properties'], dict):
            name = node_data['properties'].get('name', '')
        else:
            name = node_data.get('name', '')

        if name:
            if isinstance(name, list):
                name = ", ".join(str(item) for item in name)
            elif not isinstance(name, str):
                name = str(name)

        return name

    def get_subGraph_nodes(self, subGraph: str) -> List[str]:
        if subGraph not in self.graph.nodes:
            return []

        node_data = self.graph.nodes[subGraph]

        if node_data.get('label') != 'subGraph':
            return []

        if 'properties' in node_data:
            member_names = node_data['properties'].get('members', [])

            member_ids = []
            for name in member_names:
                if isinstance(name, list):
                    name = ", ".join(str(item) for item in name)
                elif not isinstance(name, str):
                    name = str(name)
                if name in self.name_to_id:
                    member_ids.append(self.name_to_id[name])

            return member_ids

        return []

    def get_subGraph_members(self, subGraph_node: str) -> Tuple[List[str], List[str]]:
        if subGraph_node not in self.graph.nodes:
            return [], []

        node_data = self.graph.nodes[subGraph_node]

        if 'properties' not in node_data:
            return [], []

        properties = node_data['properties']

        members = properties.get('members', [])
        entities = [str(m) if not isinstance(m, list) else ", ".join(str(x) for x in m) for m in members]
        keywords = properties.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [str(k) for k in keywords]

        return entities, keywords

    def format_subGraph_content(self, base_text: str, entities: List[str],
                                 keywords: List[str]) -> str:
        parts = [base_text]

        if entities:
            parts.append(f"Entities: {', '.join(entities[:10])}")

        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:10])}")

        return " | ".join(parts)

    def nodes_to_text(self, nodes: List[str]) -> str:
        texts = []

        for node in nodes:
            if node in self.graph.nodes:
                node_text = self.get_node_text(node)
                if node_text:
                    texts.append(node_text)

        return " ".join(texts)

    def subgraph_to_text(self, subgraph: nx.MultiDiGraph) -> str:
        text_parts = []

        for node, data in subgraph.nodes(data=True):
            node_text = f"Node: {data.get('name', node)}\n"
            if 'description' in data:
                node_text += f"Description: {data['description']}\n"
            if 'properties' in data:
                node_text += f"Properties: {data['properties']}\n"
            text_parts.append(node_text)

        for u, v, data in subgraph.edges(data=True):
            u_name = subgraph.nodes[u].get('name', u)
            v_name = subgraph.nodes[v].get('name', v)
            relation = data.get('relation', '')
            edge_text = f"Relation: {relation} between {u_name} and {v_name}\n"
            text_parts.append(edge_text)

        return "\n".join(text_parts)

    def is_valid_node_text(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < 3:
            return False

        if text.lower() in ['none', 'none,none', ',none']:
            return False

        return True

