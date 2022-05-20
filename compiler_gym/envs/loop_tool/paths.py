import logging
import os
import pdb
import subprocess
from pathlib import Path
from typing import Iterable

"""
Purpose:
    In this file we define all paths important for the project.
    To run the project user needs to set COMPILER_GYM_ROOT to the path of loop_tool_env directory.
    BENCHMARKS_PATH - path to user-defined datasets source-code
    LOOP_TOOL_SERVICE_PY - path to loop_tool_env backend service
"""

COMPILER_GYM_ROOT = Path(os.environ.get("COMPILER_GYM_ROOT"))
print(f"\nCOMPILER_GYM_ROOT = {COMPILER_GYM_ROOT}\n")

assert (
    COMPILER_GYM_ROOT
), "\n\nInitialize envvar COMPILER_GYM_ROOT to path of the loop_tool_env folder \n"


BENCHMARKS_PATH: Path = Path(COMPILER_GYM_ROOT / "compiler_gym/envs/loop_tool/datasets")


LOOP_TOOL_SERVICE_PY: Path = Path(
    COMPILER_GYM_ROOT
    / "compiler_gym/envs/loop_tool/service/loop_tool_compilation_session.py"
)

logging.info(f"What is the path {LOOP_TOOL_SERVICE_PY}")
logging.info(f"Is that file: {LOOP_TOOL_SERVICE_PY.is_file()}")
assert LOOP_TOOL_SERVICE_PY.is_file(), "Service script not found"
