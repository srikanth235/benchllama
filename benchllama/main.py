import typer
import pandas as pd

from typing import List, Optional
from typing_extensions import Annotated, Optional, List
from rich.console import Console
from rich.table import Table, Column
from pathlib import Path
from .data_io import Loader

app = typer.Typer()
console = Console()

@app.command()
def evaluate(
        model: Annotated[Optional[List[str]], typer.Option(help="Names of models")] = list(["model_1", "model_2"]),
        dataset_dir: Annotated[Optional[Path], typer.Argument(help="Name of the dataset")] = "/home/dataset_1",
        output_dir: Annotated[Optional[Path], typer.Option(help="Output directory")] = "/tmp/outputs/",
        language: Annotated[Optional[List[str]], typer.Option(help="Names of models")] = list(["javascript", "python"]),
    ):
    """
    Evaluate the performance of the portal gun
    """
    loader = Loader()
    loader.get_data().select(range(5)).map(console.print)

    df = pd.DataFrame({
        "Model": ["DeepSeek", "DeepSeek", "DeepSeek"],
        "Benchmark": ["HumanEval", "HumanEvl", "HumanEval"],
        "Prompt Eval Duration": [6, 7, 8],
        "Prompt Eval Rate": [0, 1, 2],
        "Eval Duration": [9, 10, 11],
        "Eval Rate": [3, 4, 5],
        "pass@1": [1, 2, 3]
    })

    table = Table(
        Column(header="Model", style="cyan"),
        Column(header="Benchmark", style="cyan"),
        Column(header="Prompt Eval Duration", style="cyan"),
        Column(header="Prompt Eval Rate", style="cyan"),
        Column(header="Eval Duration", style="cyan"),
        Column(header="Eval Rate", style="cyan"),
        Column(header="pass@1", style="cyan"),
        title="[bold cyan] Language [bold magenta] Benchmarks[green]:heavy_check_mark:"
    )
    for _, row in df.iterrows():
        table.add_row(
            str(row["Model"]),
            str(row["Benchmark"]),
            str(row["Prompt Eval Duration"]),
            str(row["Prompt Eval Rate"]),
            str(row["Eval Duration"]),
            str(row["Eval Rate"]),
            str(row["pass@1"])
        )
    console.print(table)
