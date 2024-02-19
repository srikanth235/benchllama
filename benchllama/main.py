import shutil
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from typing_extensions import Annotated

from .constants import Language
from .data_io import Loader
from .evaluation import Evaluator
from .inference import ModelProvider
from .utils import pretty_print

app = typer.Typer()


@app.command()
def clean(
    run_id: Annotated[Optional[str], typer.Option(help="Run id")] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            dir_okay=True,
            writable=True,
            resolve_path=True,
            help="Output directory",
        ),
    ] = "/tmp",
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
    models: Annotated[
        List[str], typer.Option(help="Names of models that need to be evaluated.")
    ],
    provider_url: Annotated[
        Optional[str], typer.Option(help="The endpoint of the model provider.")
    ] = "http://localhost:11434",
    dataset: Annotated[
        Optional[Path],
        typer.Option(
            help="By default, bigcode/humanevalpack from Hugging Face will be used.  If you want to use your own dataset, specify the path here.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    languages: Annotated[
        Optional[List[Language]],
        typer.Option(
            help="List of languages to evaluate from bigcode/humanevalpack. Ignore this if you are brining your own data",
            case_sensitive=False,
        ),
    ] = list([Language.python]),
    num_completions: Annotated[
        Optional[int],
        typer.Option(help="Number of completions to be generated for each task."),
    ] = 3,
    k: Annotated[
        Optional[List[int]],
        typer.Option(
            help="The k for calculating pass@k. The values shouldn't exceed num_completions"
        ),
    ] = list([1, 2]),
    samples: Annotated[
        Optional[int],
        typer.Option(
            help="Number of dataset samples to evaluate. By default, all the samples get processed."
        ),
    ] = -1,
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Output directory",
            exists=True,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = "/tmp",
):
    start_time = time.time()
    input_df = Loader(dataset, languages=languages).get_data(models, samples)
    print(f"Dataset loaded :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    result_df = ModelProvider(provider_url).run_inference(input_df, num_completions)
    print(f"Prompts inferred :boom: in { time.time() - start_time :.4f} seconds.")

    start_time = time.time()
    evaluator = Evaluator(output)
    executed_df = evaluator.execute_code(result_df)
    print(
        f"Code execution completed :boom: in { time.time() - start_time :.4f} seconds."
    )

    start_time = time.time()
    result_df = evaluator.estimate_score(executed_df, k)
    print(f"Evaluation completed :boom: in { time.time() - start_time :.4f} seconds.")

    pretty_print(result_df)
    print(f"\nYou can access the run data at: ", evaluator.get_execution_directory())
