import pandas as pd
from pathlib import Path


from ..constants import Result
from .runners.python_runner import PythonRunner

from .runners.cpp_runner import CppRunner
# from .runners.rust_runner import RustRunner
from .runners.java_runner import JavaRunner
from .runners.javascript_runner import JavascriptRunner
from .runners.go_runner import GoRunner


class CodeRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir
        self.python_runner = PythonRunner(execution_dir)
        self.javascript_runner = JavascriptRunner(execution_dir)
        self.java_runner = JavaRunner(execution_dir)
        self.cpp_runner = CppRunner(execution_dir)
        self.go_runner = GoRunner(execution_dir)

    def run(self, problem: pd.Series):
        if problem["language"].lower() == "python":
            return self.python_runner.run(problem)
        elif problem["language"].lower() == "cpp":
            return self.cpp_runner.run(problem)
        elif problem["language"].lower() == "rust":
            return self.rust_runner.run(problem)
        elif problem["language"].lower() == "java":
            return self.java_runner.run(problem)
        elif problem["language"].lower() == "javascript":
            return self.javascript_runner.run(problem)
        elif problem["language"].lower() == "go":
            return self.go_runner.run(problem)
        else:
            return Result.FAILURE, "Language not supported!"
