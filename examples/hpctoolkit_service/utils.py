from pathlib import Path

import subprocess
from typing import Iterable


hpctoolkit_service_path = Path(__file__).parent.absolute()
compiler_gym_path = hpctoolkit_service_path.parent.parent


HPCTOOLKIT_HEADER: Path = Path(
    compiler_gym_path / "compiler_gym/third_party/hpctoolkit/header.h"
)


HPCTOOLKIT_PY_SERVICE_BINARY: Path = Path(
    hpctoolkit_service_path / "service_py/example_service.py"
)
assert HPCTOOLKIT_PY_SERVICE_BINARY.is_file(), "Service script not found"


