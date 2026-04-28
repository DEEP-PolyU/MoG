
import json
from typing import Dict, List, Any

from ...utils import logger
import os

class SchemaAgent:
    def __init__(self, config, dataset_name: str):
        self.schema_path_base = config.get_dataset_config(dataset_name).schema_path_base
        self.schema_path = config.get_dataset_config(dataset_name).schema_path
        self.dataset_name = dataset_name
        self.schema = self.load_schema(self.schema_path_base)
        logger.info(f"Load schema from {self.schema_path_base} for dataset:{dataset_name}:\n {self.schema}")

    def load_schema(self, schema_path) -> Dict[str, Any]:
        try:
            with open(schema_path) as f:
                schema = json.load(f)
                return schema
        except FileNotFoundError:
            return dict()

    def save_schema(self):
        os.makedirs(os.path.dirname(self.schema_path), exist_ok=True)
        with open(self.schema_path, 'w', encoding='utf-8') as f:
            json.dump(self.schema, f, ensure_ascii=False, indent=2)

    def update_schema_with_new_types(self, new_schema_types: Dict[str, List[str]]) -> Dict:
        try:
            current_schema = self.schema

            updated = False

            if "nodes" in new_schema_types:
                for new_node in new_schema_types["nodes"]:
                    if new_node not in current_schema.get("Nodes", []):
                        current_schema.setdefault("Nodes", []).append(new_node)
                        updated = True

            if "relations" in new_schema_types:
                for new_relation in new_schema_types["relations"]:
                    if new_relation not in current_schema.get("Relations", []):
                        current_schema.setdefault("Relations", []).append(new_relation)
                        updated = True

            if "attributes" in new_schema_types:
                for new_attribute in new_schema_types["attributes"]:
                    if new_attribute not in current_schema.get("Attributes", []):
                        current_schema.setdefault("Attributes", []).append(new_attribute)
                        updated = True

            if updated:
                os.makedirs(os.path.dirname(self.schema_path), exist_ok=True)
                with open(self.schema_path, 'w', encoding='utf-8') as f:
                    json.dump(current_schema, f, ensure_ascii=False, indent=2)
                self.schema = current_schema
        except Exception as e:
            logger.error(f"Failed to update schema for dataset '{self.dataset_name}': {type(e).__name__}: {e}")
