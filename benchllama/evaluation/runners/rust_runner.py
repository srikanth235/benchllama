from pathlib import Path

from benchllama.constants import Result
import pandas as pd
import subprocess
import shutil
import fcntl
from .utils import get_prompt_and_completion


def initialize_cargo_once(rust_template_dir: Path):
    lockfile = rust_template_dir / ".init.lock"
    with open(lockfile, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        cargo_toml = rust_template_dir / "Cargo.toml"
        if not cargo_toml.exists():
            subprocess.run(
                ["cargo", "init", "--lib", "--name", "main"],
                cwd=rust_template_dir,
                check=True,
                capture_output=True,
            )
        fcntl.flock(f, fcntl.LOCK_UN)


class RustRunner:
    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir
        self.rust_template_dir = None

    def run(self, problem: pd.Series):
        if self.rust_template_dir is None:
            self.rust_template_dir = self.execution_dir / "rust_module"
            self.rust_template_dir.mkdir(parents=True, exist_ok=True)
            # Initialize a reusable Cargo library template once, using lockfile to prevent parallel deadlock:
            initialize_cargo_once(self.rust_template_dir)

        result = Result.FAILURE
        error = ""
        # Build code from prompt and completion
        prompt, completion = get_prompt_and_completion(problem)
        code = prompt + completion
        test_code = problem["test_setup"] + problem["test"]

        # Prepare an execution directory for the current run
        dir_path = (
            self.execution_dir
            / f"task_{problem.task_id.split('/')[-1]}"
            / f"execution_{problem.name}"
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.rust_template_dir, dir_path, dirs_exist_ok=True)

        # Write source and tests
        src_dir = dir_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        cur_file = src_dir / "lib.rs"

        # Write the code to a file
        with open(cur_file, "w", encoding="utf-8") as file:
            file.write(code)
            file.write(test_code)

        try:
            response = subprocess.run(
                ["cargo test --quiet"],
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
            # Prefer stderr/stdout from cargo if available
            if isinstance(e, subprocess.CalledProcessError):
                stderr = e.stderr.decode() if e.stderr else ""
                stdout = e.stdout.decode() if e.stdout else ""
                error = stderr or stdout or str(e)
            else:
                error = str(e)

        return result, error
