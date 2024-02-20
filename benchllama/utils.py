import pandas as pd
from rich.table import Table
from rich.console import Console
from .constants import PROMPT_EVAL_DURATION, EVAL_DURATION, PROMPT_EVAL_RATE, EVAL_RATE

console = Console()


def pretty_print(df: pd.DataFrame):
    table = Table(title=":fire: Benchmark Results")
    hidden_columns = []
    for column in df.columns:
        if column == "model":
            table.add_column("Model", justify="right", style="yellow", overflow="fold")
        elif column == "language":
            table.add_column("Language", style="purple")
        elif column == PROMPT_EVAL_DURATION:
            table.add_column(
                "Prompt Eval Duration (in secs)", justify="right", style="green"
            )
        elif column == PROMPT_EVAL_RATE:
            table.add_column(
                "Prompt Eval Rate (in tokens/sec)", justify="right", style="green"
            )
        elif column == EVAL_DURATION:
            table.add_column("Eval Duration (in secs)", justify="right", style="green")
        elif column == EVAL_RATE:
            table.add_column(
                "Eval Rate (in tokens/sec)", justify="right", style="green"
            )
        elif "pass@" in column:
            table.add_column(column, justify="right", style="cyan")
        else:
            hidden_columns.append(column)

    # # Add rows from DataFrame
    for _, row in df.iterrows():
        table.add_row(
            *[
                row[column] if isinstance(row[column], str) else f"{row[column]:.3f}"
                for column in df.columns if column not in hidden_columns
            ]
        )
    console.print(table)
