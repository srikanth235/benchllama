import typer
import pandas as pd
import time
import shutil

from typing import List, Optional
from typing_extensions import Annotated, Optional, List

from rich import print
from pathlib import Path
from enum import Enum
from .data_io import Loader
from .inference import ModelProvider
from .utils import pretty_print
from .evaluation import Evaluator
from .constants import Language

app = typer.Typer()


@app.command()
def clean(
    run_id: Annotated[Optional[str], typer.Option(help="Run id")] = None,
    output: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=True, writable=True, resolve_path=True, help="Output directory")] = "/tmp",
):
    if run_id:
        directory_path = output / "benchllama" / str(run_id)
    else:
        directory_path = output / "benchllama"

    delete = typer.confirm(f"Are you sure you want to delete {str(directory_path)}?")

    if not delete:
        print("Not deleting")
        raise typer.Abort()

    try:
        shutil.rmtree(directory_path)
        directory_path.rmdir()
        print("Directory removed successfully.")
    except FileNotFoundError:
        print("Directory does not exist!")
    except OSError as e:
        print(f"Error: {e}")

@app.command()
def evaluate(
        dataset: Annotated[Optional[Path], typer.Option(help="Use this if you want to bring your own dataset", exists=True, dir_okay=False, readable=True, resolve_path=True)] = None,
        models: Annotated[Optional[List[str]], typer.Option(help="Names of models")] = list(["model_1", "model_2"]),
        languages: Annotated[Optional[List[Language]], typer.Option(help="List of languages to evaluate from bigcode/humanevalpack", case_sensitive=False)] = list([Language.python]),
        num_completions: Annotated[Optional[int], typer.Option(help="Number of completions to be generated for each sample")] = 3,
        k: Annotated[Optional[List[int]], typer.Option(help="The k for calculating pass@k")] = list([1, 2]),
        samples: Annotated[Optional[int], typer.Option(help="Number of dataset samples to evaluate")] = 2,
        output: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=True, writable=True, resolve_path=True, help="Output directory")] = "/tmp",
    ):
    start_time = time.time()
    input_df = Loader(dataset, languages=languages).get_data(models, samples)
    print(f"Dataset loaded :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    result_df = ModelProvider().run_inference(input_df, num_completions)
    print(f"Prompts inferred :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    evaluator =  Evaluator(output)
    executed_df = evaluator.execute_code(result_df)
    print(f"Code execution completed :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    result_df = evaluator.estimate_score(executed_df, k)
    print(f"Evaluation completed :boom: in { time.time() - start_time :.4f} seconds.")

    pretty_print(result_df)
