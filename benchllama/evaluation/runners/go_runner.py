import pandas as pd
import subprocess
import shutil

from benchllama.constants import Result
from pathlib import Path


class GoRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir
        self.go_template_dir = None

    def run(self, problem: pd.Series):
        if self.go_template_dir is None:
            self.go_template_dir = self.execution_dir / "go_module"
            self.go_template_dir.mkdir(parents=True, exist_ok=True)
            response = subprocess.run(
                [
                    "go mod init main; go get github.com/stretchr/testify/assert",
                ],
                cwd=self.go_template_dir,
                check=True,
                shell=True,
                capture_output=True,
            )

        result = Result.FAILURE
        error = ""
        code = "package main" + "\n" + problem["prompt"] + problem["completion"]
        test_code = problem["test_setup"] + problem["test"]

        dir_path = (
            self.execution_dir
            / f"task_{problem.task_id.split('/')[-1]}"
            / f"execution_{problem.name}"
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.go_template_dir, dir_path, dirs_exist_ok=True)

        cur_file = dir_path / "main.go"
        cur_test_file = dir_path / "main_test.go"

        # Write the code to a file
        with open(cur_file, "w") as file:
            file.write(code)

        with open(cur_test_file, "w") as file:
            file.write(test_code)

        try:
            response = subprocess.run(
                ["go test *.go"],
                timeout=5,
                cwd=dir_path,
                check=True,
                shell=True,
                capture_output=True,
            )
            if response.returncode == 0:
                result = Result.SUCCESS
            elif response.stderr:
                error = response.stderr.decode()
            elif response.stdout:
                error = response.stdout.decode()
        except Exception as e:
            error = str(e)
        return result, error
