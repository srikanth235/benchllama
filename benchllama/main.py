import typer
import pandas as pd
import time

from typing import List, Optional
from typing_extensions import Annotated, Optional, List

from rich import print
from pathlib import Path
from enum import Enum
from .data_io import Loader
from .execution import ModelProvider
from .utils import pretty_print
from .evaluation import Evaluator


class Language(str, Enum):
    python = "python",
    js = "js",
    java = "java",
    go = "go",
    cpp = "cpp",
    rust = "rust"

app = typer.Typer()

@app.command()
def evaluate(
        dataset: Annotated[Optional[Path], typer.Argument(help="Name of the dataset")] = "bigcode/humanevalpack",
        models: Annotated[Optional[List[str]], typer.Option(help="Names of models")] = list(["model_1", "model_2"]),
        languages: Annotated[Optional[List[Language]], typer.Option(help="Names of models", case_sensitive=False)] = list([Language.python]),
        num_completions: Annotated[Optional[int], typer.Option(help="Number of completions to be generated for each sample")] = 3,
        k: Annotated[Optional[List[int]], typer.Option(help="The k for calculating pass@k")] = list([1, 5]),
        samples: Annotated[Optional[int], typer.Option(help="Number of dataset samples to evaluate")] = 1,
        output: Annotated[Optional[Path], typer.Option(help="Output directory")] = "./tmp/outputs/",
    ):
    start_time = time.time()
    input_df = Loader(languages=languages).get_data(models, samples)
    print(f"Dataset loaded :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    result_df = ModelProvider().execute_prompts(input_df, num_completions)
    print(f"Prompts inferred :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    executed_df = Evaluator().execute_code(result_df)
    print(f"Code execution completed :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    result_df = Evaluator().estimate_score(executed_df, k)
    print(f"Evaluation completed :boom: in { time.time() - start_time :.4f} seconds.")

    pretty_print(result_df)
