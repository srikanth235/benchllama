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
from pathlib import Path

from ..constants import Result

def execute(problem: pd.Series, execution_dir: Path):
    result = Result.FAILURE
    code = problem["import"] + "\n" + problem["prompt"]  + \
        problem["completion"] + "\n" + \
        problem["test"] + "\n"

    cur_file = execution_dir / f"execution_{problem.name}.py"

    # Write the code to a file
    with open(cur_file, 'w') as file:
        file.write(code)

    # Execute the Python script
    try:
        subprocess.run(['python3', str(cur_file)],
            timeout = 5,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        result = Result.SUCCESS
    except Exception as e:
        pass
    return result
