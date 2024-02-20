import pandas as pd

from ollama import Client
from .prompt_formatter import PromptFormatter
from ..constants import (
    PROMPT_EVAL_DURATION,
    PROMPT_EVAL_COUNT,
    EVAL_COUNT,
    EVAL_DURATION,
    COMPLETION,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from rich import print

class ModelProvider(object):
    def __init__(self, provider_url="http://localhost:11434"):
        self.client = Client(provider_url)
        self.prompt_formatter = PromptFormatter()

    def run_inference(self, data: pd.DataFrame, num_completions: int) -> pd.DataFrame:
        def infer(row):
            prompt, stop = self.prompt_formatter.format(row).values()
            result = self.client.generate(
                model=row["model"], prompt=prompt, options={"stop": stop}
            )
            return (
                result.get("prompt_eval_duration"),
                result.get("prompt_eval_count", 0),
                result.get("eval_duration"),
                result.get("eval_count", 0),
                result.get("response"),
            )

        processed_rows = []
        with Progress(
            TextColumn(f"• [progress.percentage]" + "{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            data = pd.concat(
                [data.copy() for _ in range(num_completions)], ignore_index=True
            )
            # Iterate over the DataFrame rows and process each row
            for index, row in progress.track(
                data.iterrows(), description="Executing prompts...", total=len(data)
            ):
                result = infer(row)
                processed_row = row.copy()
                processed_row[PROMPT_EVAL_DURATION] = result[0]
                processed_row[PROMPT_EVAL_COUNT] = result[1]
                processed_row[EVAL_DURATION] = result[2]
                processed_row[EVAL_COUNT] = result[3]
                processed_row[COMPLETION] = result[4]
                processed_rows.append(processed_row)

        data = pd.DataFrame(processed_rows)
        return data
