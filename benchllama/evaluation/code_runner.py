import pandas as pd
from pathlib import Path

from ..constants import Result

from .runners.cpp_runner import CppRunner
from .runners.go_runner import GoRunner
from .runners.java_runner import JavaRunner
from .runners.javascript_runner import JavascriptRunner
from .runners.rust_runner import RustRunner
from .runners.python_runner import PythonRunner


def debug(*text):
    with open("/tmp/debug", "a") as f:
        f.write(str(' '.join(text)))

class CodeRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir

        self.runners = {
            "cpp": CppRunner(self.execution_dir),
            "go": GoRunner(self.execution_dir),
            "javascript": JavascriptRunner(self.execution_dir),
            "java": JavaRunner(self.execution_dir),
            "python": PythonRunner(self.execution_dir),
            "rust": RustRunner(self.execution_dir),
        }

    def run(self, problem: pd.Series):
        language = problem["language"].lower()

        if language not in self.runners:
            return Result.FAILURE, "Language not supported!"

        runner = self.runners[language]

        return runner.run(problem)
