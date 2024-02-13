import pandas as pd
from rich.table import Table, Column
from rich.console import Console
from .execution.constants import PROMPT_EVAL_DURATION, PROMPT_EVAL_RATE, EVAL_DURATION, EVAL_RATE

console = Console()

def pretty_print(df: pd.DataFrame):
    table = Table()
    current_columns = ['task_id', 'model', PROMPT_EVAL_DURATION, PROMPT_EVAL_RATE, EVAL_DURATION, EVAL_RATE]

    # Dynamically add columns from DataFrame
    for column in current_columns:
        table.add_column(column, style="cyan")

    # Add rows from DataFrame
    for _, row in df[current_columns].iterrows():
        table.add_row(*[str(row[column]) for column in df[current_columns]])
    console.print(table)