import pandas as pd
from typing import List
from joblib import Parallel, delayed
from .code_executor import execute
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from .score_estimator import ScoreEstimator
class Evaluator:
    def __init__(self):
        self.score_estimator = ScoreEstimator()

    def execute_code(self, input_df: pd.DataFrame):
        outputs = []
        with Progress(
             TextColumn(
                f"• [progress.percentage]" + "{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Verifying code...", total=len(input_df))
            for result in Parallel(n_jobs=-1)(delayed(execute)(row) for index, row in input_df.iterrows()):
                outputs.append(result.value)
                progress.update(task, advance=1)

        input_df["result"] = outputs
        return input_df

    def estimate_score(self, input_df: pd.DataFrame, k: List[int]):
        return self.score_estimator.estimate_score(input_df, k)

