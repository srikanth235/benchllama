import typer
import pandas as pd

from typing import List, Optional
from typing_extensions import Annotated, Optional, List

from rich import print
from pathlib import Path
from enum import Enum
from .data_io import Loader
from .execution import ModelProvider
from .utils import pretty_print

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
        model: Annotated[Optional[List[str]], typer.Option(help="Names of models")] = list(["model_1", "model_2"]),
        language: Annotated[Optional[List[Language]], typer.Option(help="Names of models", case_sensitive=False)] = list([Language.python]),
        dataset: Annotated[Optional[Path], typer.Argument(help="Name of the dataset")] = "/home/dataset_1",
        output: Annotated[Optional[Path], typer.Option(help="Output directory")] = "/tmp/outputs/",
    ):
    """
    Evaluate the performance of the models
    """
    input_df = Loader().get_data(model)
    result_df = ModelProvider().execute_prompts(input_df)
    evaluated_df = None
    pretty_print(result_df)
