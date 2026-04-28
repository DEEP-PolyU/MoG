
import json
from typing import Dict

import json_repair
from ...utils import LLMCompletionCall



class LLMProcessorConstruction:
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMCompletionCall(config.output.results_dir)

    def extract_with_llm(self, prompt: str) -> str:
        response = self.llm_client.call_api(prompt)
        parsed_dict = json_repair.loads(response)
        parsed_json = json.dumps(parsed_dict, ensure_ascii=False)
        return parsed_json

    def get_construction_prompt(self, chunk: str, dataset_name: str, mode: str, schema) -> str:
        recommend_schema = json.dumps(schema, ensure_ascii=False)

        prompt_type_map = {
            "novel": "novel",
            "novel_eng": "novel_eng"
        }

        base_prompt_type = prompt_type_map.get(dataset_name, "general")

        if mode == "agent":
            prompt_type = f"{base_prompt_type}_agent"
        else:
            prompt_type = base_prompt_type

        return self.config.get_prompt_formatted("construction", prompt_type, schema=recommend_schema, chunk=chunk)

    def validate_and_parse_llm_response(self, prompt: str, llm_response: str, token_calculator) -> Dict:
        if llm_response is None:
            return {}

        try:
            if token_calculator:
                token_calculator(prompt + llm_response)

            result = json_repair.loads(llm_response)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            llm_response_str = str(llm_response) if llm_response is not None else "None"
            return {}
