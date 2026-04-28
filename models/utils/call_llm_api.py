import os
import time
import json
import requests
import re

from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

from .logger import logger

load_dotenv()


class LLMCompletionCall:
    def __init__(self, results_dir):
        load_dotenv('llm.env', override=True)

        self.llm_model = os.getenv("LLM_MODEL")
        self.llm_base_url = os.getenv("LLM_BASE_URL")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        logger.warning(f"Call {self.llm_model} from {self.llm_base_url}")
        if not self.llm_api_key:
            raise ValueError("LLM API key not provided")
        self.openai_provider = os.getenv("OPENAI_PROVIDER", "openai").lower()

        self.max_retries = 20
        self.retry_delay = 2
        self.timeout = 30

        self.client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

    def call_api(self, content: str) -> str:
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.3,
                    timeout=self.timeout
                )
                raw = completion.choices[0].message.content or ""
                clean_completion = self.clean_llm_content(raw)
                return clean_completion

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM api calling failed (attempt {attempt + 1}/{self.max_retries}). Error: {error_msg}")

                if self.is_rate_limit_error(error_msg):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * attempt
                        logger.info(f"Rate limit detected, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"All retry attempts failed due to rate limiting. Last error: {error_msg}")
                        raise e
                else:
                    if attempt == self.max_retries - 1:
                        logger.error(f"All retry attempts failed. Last error: {error_msg}")
                        raise e
                    else:
                        wait_time = self.retry_delay
                        logger.info(f"Non-rate-limit error, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue

        raise Exception(f"API call failed after {self.max_retries} attempts")

    def is_rate_limit_error(self, error_msg: str) -> bool:
        rate_limit_indicators = [
            'rate limit',
            '429',
            'tpm',
            'too many requests',
            'quota exceeded',
            'limit exceeded'
        ]

        error_msg_lower = error_msg.lower()
        return any(indicator in error_msg_lower for indicator in rate_limit_indicators)

    def clean_llm_content(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        m = fence_re.match(t)
        if m:
            t = m.group("body").strip()
        else:
            if t.startswith("```") and t.endswith("```") and len(t) >= 6:
                t = t[3:-3].strip()

        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()

        return t

    def eval(self, question, gold_answer, answer):
        prompt = f"""
        You are an expert evaluator. Your task is to determine if the predicted answer is correct based on the question and gold answer.
        The criteria should be reasonable, not too strict or too lenient.

        Question: {question}
        Gold Answer: {gold_answer}
        Predicted Answer: {answer}

        Return only "1" (correct) or "0" (incorrect):
        """
        return self.call_api(prompt)