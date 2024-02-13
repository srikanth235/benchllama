
import logging
import pandas as pd

from typing import Dict, Any
from ollama import Client
from functools import partial
from .prompt_formatter import PromptFormatter
from .constants import PROMPT_EVAL_DURATION, PROMPT_EVAL_RATE, EVAL_DURATION, EVAL_RATE


log = logging.getLogger(__name__)

class ModelProvider(object):
    def __init__(self, host="http://localhost:11434"):
        self.client = Client(host)
        self.prompt_formatter = PromptFormatter()

    def execute_prompts(self, data: pd.DataFrame) -> pd.DataFrame:
        def infer(row):
            prompt, stop = self.prompt_formatter.format(row).values()
            result = self.client.generate(
                model=row['model'],
                prompt=prompt,
                options={
                    "stop": stop
                }
            )
            return result.get("prompt_eval_duration"), \
                result.get("prompt_eval_count", 0) / result.get("prompt_eval_duration"), \
                result.get("eval_duration"), \
                result.get("eval_count", 0) / result.get("eval_duration")


        data[[PROMPT_EVAL_DURATION, PROMPT_EVAL_RATE, EVAL_DURATION, EVAL_RATE]] =  \
            data.apply(infer, axis=1, result_type='expand')
        return data
