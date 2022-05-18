from pathlib import Path
import os
import subprocess
from typing import Iterable
import logging

import subprocess
import pdb

'''
Purpose:
    In this file we define all paths important for the project.
    To run the project user needs to set LOOP_TOOL_ROOT to the path of loop_tool directory.
    BENCHMARKS_PATH - path to user-defined benchmarks source-code
    LOOP_TOOL_SERVICE_PY - path to loop_tool backend service 
'''

LOOP_TOOL_ROOT = Path(os.environ.get("LOOP_TOOL_ROOT"))
assert LOOP_TOOL_ROOT, "\n\nInitialize envvar LOOP_TOOL_ROOT to path of the CompilerGym/envs/loop_tool folder \n"


BENCHMARKS_PATH: Path = Path(
    LOOP_TOOL_ROOT / "loop_tool_service/benchmarks"
)


LOOP_TOOL_SERVICE_PY: Path = Path(
    LOOP_TOOL_ROOT / "loop_tool_service/service/loop_tool_compilation_session.py"
)

logging.info(f"What is the path {LOOP_TOOL_SERVICE_PY}")
logging.info(f"Is that file: {LOOP_TOOL_SERVICE_PY.is_file()}")
assert LOOP_TOOL_SERVICE_PY.is_file(), "Service script not found"
