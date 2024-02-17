import pandas as pd
import subprocess

from benchllama.constants import Result
from pathlib import Path


class JavascriptRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir

    def run(self, problem: pd.Series):
        result = Result.FAILURE
        error = ""
        code = problem["prompt"] + problem["completion"] + "\n" + problem["test"] + "\n"

        cur_file = self.execution_dir / f"execution_{problem.name}.js"
        # Write the code to a file
        with open(cur_file, "w") as file:
            file.write(code)

        try:
            response = subprocess.run(
                ["node", str(cur_file)],
                timeout=5,
                check=True,
                capture_output=True,
                shell=True,
            )
            if response.stderr.decode():
                error = response.stderr.decode()
            elif response.stdout.decode():
                error = response.stdout.decode()
            else:
                result = Result.SUCCESS
        except Exception as e:
            error = str(e.stderr)
        return result, error
