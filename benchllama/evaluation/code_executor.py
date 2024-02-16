from typing import Optional, Callable, Dict
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
import subprocess
from enum import Enum
import pandas as pd
from rich import print


class Result(Enum):
    SUCCESS = 1
    FAILURE = 0

def execute(problem: pd.Series):
    result = Result.FAILURE
    python_code = problem["import"] + "\n" + problem["prompt"]  + \
        problem["completion"] + "\n" + \
        problem["test"] + "\n"

    # Write the Python code to a file
    with open(f'execution_{problem.name}.py', 'w') as file:
        file.write(python_code)

    # Execute the Python script
    try:
        subprocess.run(['python3', f'execution_{problem.name}.py'],
            timeout = 5,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        result = Result.SUCCESS
    except Exception as e:
        pass
    return result



@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)