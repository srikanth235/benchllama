import pandas as pd
from typing import Dict, Any


class PromptFormatter:
    def __init__(self):
        pass

    def format(self, row: pd.core.series.Series):
        data = row.to_dict()
        additional_context = data.get("additional_context", "")
        prefix = data.get("prompt", "")
        suffix = data.get("suffix", "")
        model = data.get("model")

        if "deepseek" in model.lower():
            return {
                "prompt": f"<｜fim▁begin｜>{additional_context}\n{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
                "stop": ["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>", "<END>"],
            }
        elif "stable" in model.lower() or "starcoder" in model.lower():
            return {
                "prompt": f"<fim_prefix>{additional_context}\n{prefix}<fim_suffix>{suffix}<fim_middle>",
                "stop": ["<|endoftext|>"],
            }
        return {
            "prompt": f"<PRE>{additional_context}\n{prefix} <SUF>{suffix} <MID>",
            "stop": ["<PRE>", "<SUF>", "<MID>", "<END>", "EOT"],
        }
