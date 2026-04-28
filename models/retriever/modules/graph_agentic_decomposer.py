import json_repair
from  ...utils import call_llm_api, logger
from config import get_config

class GraphAgenticDecomposer:
    def __init__(self, dataset_name, config):
        self.config = config
        self.llm_client = call_llm_api.LLMCompletionCall(config.output.results_dir)
        self.dataset_name = dataset_name
            
    def read_schema(self, schema_path: str) -> str:
        with open(schema_path, "r") as f:
            schema = f.read()
        return schema
    
    def prompt_format(self, schema: str, question: str) -> str:
        return f"""
        You are a professional question decomposition expert specializing in multi-hop reasoning.
        Given the following schema and the question, decompose the complex question into the necessary number of focused sub-questions.
        
        CRITICAL REQUIREMENTS:
        1. Decompose into the MINIMUM number of sub-questions necessary to answer the original question.
        2. Each sub-question must be:
           - As SIMPLE and DIRECT as possible
           - Specific and focused on a single fact or relationship by identifying all entities, relationships, and reasoning steps needed
           - Answerable independently with the given schema
           - Explicitly reference entities and relations from the original question
           - Designed to retrieve relevant knowledge for the final answer
           - Non-redundant and non-overlapping with other sub-questions
        3. Output sub-questions in a logical sequential order that builds toward the final answer.
        4. For simple questions, return the original question as a single sub-question in a JSON array.
        5. Return a JSON array of strings, each string being a sub-question.

        Graph Schema:
        {schema}

        Question: {question}

        Example for complex question:
        Original: "Which film has the director died earlier, Ethnic Notions or Gordon Of Ghost City?"
        Sub-questions:
        [
            {{"sub-question": "Who is the director of Ethnic Notions?"}},
            {{"sub-question": "Who is the director of Gordon Of Ghost City?"}},
            {{"sub-question": "When did the director of Ethnic Notions die?"}},
            {{"sub-question": "When did the director of Gordon Of Ghost City die?"}}
        ]

        Example for simple question:
        Original: "What is the capital of France?"
        Sub-questions:
        [
            {{"sub-question": "What is the capital of France?"}}
        ]
        """
    
    def decompose(self, question: str, schema_path: str) -> dict:
        schema = self.read_schema(schema_path)
        prompt = self.prompt_format(schema, question)
        response = self.llm_client.call_api(prompt)
        content = json_repair.loads(response)

        if isinstance(content, list):
            content = {
                "sub_questions": content
            }
        
        return content  

    def question_decomposition(self, question, schema_path):
        try:
            decomposition_result = self.decompose(question, schema_path)
            sub_questions = decomposition_result.get("sub_questions", [])
            logger.info(f"Original question: {question}")
            logger.info(f"Decomposed into {len(sub_questions)} sub-questions:")
            for i, sub_question in enumerate(sub_questions):
                logger.info(f"  Sub-question {i + 1}: {sub_question}")
        except Exception as e:
            logger.error(f"Error decomposing question: {str(e)}")
            sub_questions = [{"sub-question": question}]
        return sub_questions