import pandas as pd
import time

from rich import print
from typing import List
from datasets import load_dataset, concatenate_datasets
from rich.progress import Progress, SpinnerColumn, TextColumn


class Loader():
    dataset = None
    languages = None

    def __init__(self, name="bigcode/humanevalpack", languages=["python"]):
        self.languages = languages
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Loading dataset...", total=None)
            self.dataset = concatenate_datasets([
                load_dataset(name, language, trust_remote_code=True)["test"] for language in languages
            ])



    def get_data(self, models, samples=3):
        df = self.dataset.to_pandas()
        if samples is not None:
            df['language'] = df["task_id"].apply(lambda x: x.split("/")[0])
            df = df.groupby('language',  group_keys=False).apply(lambda group: group.sample(replace=False, n=min(samples, len(group))))
        dfs = []
        for model in models:
            curr_df = df.copy()
            curr_df['model'] = model
            dfs.append(curr_df)
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df

