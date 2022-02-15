import pandas as pd
import programl as pg
import pickle
import pdb
from typing import Dict, List, Optional, Tuple
from compiler_gym.util.commands import run_command

from compiler_gym.service.proto import (
    Observation,
)


class Profiler:
    def __init__(self, run_cmd, timeout_sec, src_path=None):
        self.run_cmd = run_cmd
        self.timeout_sec = timeout_sec
        self.llvm_path = src_path


    def get_observation(self) -> Observation:
        g_programl = self.programl_get_graph(self.llvm_path)
        pickled = pickle.dumps(g_programl)
        return Observation(binary_value=pickled)


    def programl_get_graph(self, ll_path: str) -> pg.ProgramGraph:

        with open(ll_path, "r") as f:
            code_str = f.read().rstrip()
            g_programl = pg.from_llvm_ir(code_str)
            g_programl = pg.to_networkx(g_programl)

        return g_programl