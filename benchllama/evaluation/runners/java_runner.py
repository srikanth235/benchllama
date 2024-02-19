import pandas as pd
import subprocess

from benchllama.constants import Result
from pathlib import Path


class JavaRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir

    def run(self, problem: pd.Series):
        result = Result.FAILURE
        error = ""
        code = (
            problem["prompt"]
            + problem["completion"]
            + "\n"
            + problem["test"]
            + "\n"
        )

        dir_path = self.execution_dir / f"execution_{problem.name}"
        dir_path.mkdir(parents=True, exist_ok=True)

        cur_file = self.execution_dir / f"execution_{problem.name}" / "Main.java"
        # Write the code to a file
        with open(cur_file, "w") as file:
            file.write(code)

        try:
            compilation_response = subprocess.run(
                ["javac Main.java"],
                timeout=5,
                cwd=dir_path,
                check=True,
                shell=True,
                capture_output=True,
            )
            if compilation_response.returncode != 0:
                raise Exception("Compilation failed")

            response = subprocess.run(
                ["java Main"],  # removing .java extension.
                cwd=dir_path,
                timeout=5,
                check=True,
                capture_output=True,
                shell=True
            )
            if response.returncode == 0:
                result = Result.SUCCESS
            if response.stderr:
                error = response.stderr.decode()
            elif response.stdout:
                error = response.stdout.decode()
        except Exception as e:
            error = str(e)
        return result, error
