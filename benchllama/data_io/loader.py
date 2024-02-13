import logging
import pandas as pd

from rich import print
from typing import List
from datasets import load_dataset, concatenate_datasets

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

class Loader():
    dataset = None

    def __init__(self, name="bigcode/humanevalpack", languages=["python"]):
        self.dataset = concatenate_datasets([
            load_dataset(name, language, trust_remote_code=True)["test"] for language in languages
        ])
        print(f"Info: Dataset loaded :boom: for languages: {languages}")


    def get_data(self, models, num_samples=1):
        df = self.dataset.to_pandas()
        dfs = []
        for model in models:
            curr_df = df.copy()
            curr_df['model'] = model
            dfs.append(curr_df)
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df.sample(n=num_samples, replace=False, random_state=42)

