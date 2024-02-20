import uuid
import pandas as pd
from typing import List, Optional
from joblib import Parallel, delayed
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from pathlib import Path
from .score_estimator import ScoreEstimator
from .code_runner import CodeRunner



class Evaluator:
    def __init__(self, input_df: pd.DataFrame, output: Path):
        self.score_estimator = ScoreEstimator()
        self.execution_dir = output / "benchllama" / str(uuid.uuid4())
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        self.code_runner = CodeRunner(self.execution_dir)
        self.input_df = input_df

    def store_raw_data(self):
        with open(self.execution_dir / "input_data.jsonl", "w") as f:
            f.write(self.input_df.to_json(orient="records", lines=True))

    def execute_code(self):
        outputs = []
        errors = []
        with Progress(
            TextColumn(f"• [progress.percentage]" + "{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Verifying code...", total=len(self.input_df))
            for result, error in Parallel(n_jobs=-1)(
                delayed(self.code_runner.run)(row) for index, row in self.input_df.iterrows()
            ):
                outputs.append(result.value)
                errors.append(error)
                progress.update(task, advance=1)

        self.input_df["result"] = outputs
        self.input_df["errors"] = errors
        return self.input_df

    def estimate_score(self, no_eval: bool, k: Optional[List[int]] = None):
        return self.score_estimator.estimate_score(self.input_df, no_eval, k)

    def get_execution_directory(self):
        return self.execution_dir