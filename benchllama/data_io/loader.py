import pandas as pd

from datasets import load_dataset, concatenate_datasets
from rich.progress import Progress, SpinnerColumn, TextColumn


class Loader:
    dataset = None
    languages = None

    def __init__(self, name, languages=["python"]):
        self.languages = languages
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Loading dataset...", total=None)
            if name is None:
                self.dataset = concatenate_datasets(
                    [
                        load_dataset(
                            "bigcode/humanevalpack", language, trust_remote_code=True
                        )["test"]
                        for language in languages
                    ]
                )
            else:
                self.dataset = load_dataset("json", data_files={"test": str(name)})["test"]

    def get_data(self, models, samples=-1):
        df = self.dataset.to_pandas()
        df["language"] = df["task_id"].apply(lambda x: x.split("/")[0].lower())
        if samples != -1:
            df = df.groupby("language", group_keys=False).apply(
                lambda group: group.sample(replace=False, n=min(samples, len(group)))
            )
        dfs = []
        for model in models:
            curr_df = df.copy()
            curr_df["model"] = model
            dfs.append(curr_df)
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
