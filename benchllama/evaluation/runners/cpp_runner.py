import pandas as pd
import subprocess

from benchllama.constants import Result
from pathlib import Path


class CppRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir

    def run(self, problem: pd.Series):
        result = Result.FAILURE
        error = ""
        code = problem["prompt"] + problem["canonical_solution"] + "\n" + problem["test"] + "\n"

        dir_path = (
            self.execution_dir
            / f"task_{problem.task_id.split('/')[-1]}"
            / f"execution_{problem.name}"
        )
        dir_path.mkdir(parents=True, exist_ok=True)

        cur_file = dir_path / "test.cpp"

        with open(cur_file, "w") as file:
            file.write(code)

        try:
            compilation_response = subprocess.run(
                ["g++  -std=c++17 test.cpp"],
                timeout=5,
                cwd=dir_path,
                check=True,
                capture_output=True,
                shell=True,
            )
            if compilation_response.returncode != 0:
                raise Exception("Compilation failed")

            response = subprocess.run(
                ["./a.out"],
                cwd=dir_path,
                timeout=5,
                check=True,
                capture_output=True,
                shell=True,
            )
            if response.returncode == 0:
                result = Result.SUCCESS
            elif response.stderr:
                error = response.stderr.decode()
            else:
                error = response.stdout.decode()
        except Exception as e:
            error = str(e)
        return result, error
