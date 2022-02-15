import pdb
import pickle
from typing import Dict, List, Optional, Tuple

import hatchet as ht
import pandas as pd
import utils

from compiler_gym.service.proto import Observation
from compiler_gym.util.commands import run_command


class Profiler:
    def __init__(self, run_cmd, timeout_sec, src_path=None):
        self.run_cmd = run_cmd
        self.timeout_sec = timeout_sec
        self.exe_path = run_cmd[0]
        self.llvm_path = src_path
        self.exe_struct_path = self.exe_path + ".hpcstruct"

        self.metrics_list = [
            "REALTIME@100",
        ]

    def get_observation(self) -> Observation:
        g_hatchet = self.hatchet_get_graph()
        pickled = pickle.dumps(g_hatchet)
        return Observation(binary_value=pickled)

    def hatchet_get_graph(self) -> ht.GraphFrame:
        events_list = []

        for m in self.metrics_list:
            events_list.extend(["-e", m])

        hpctoolkit_cmd = [
            [
                "rm",
                "-rf",
                self.exe_struct_path,
                "m",
                "db",
            ],
            [
                "hpcrun",
                "-o",
                "m",
                "-t",
            ]
            + events_list
            + self.run_cmd,
            ["hpcstruct", "-o", self.exe_struct_path, self.exe_path],
            [
                "hpcprof-mpi",
                "-o",
                "db",
                "--metric-db",
                "yes",
                "-S",
                self.exe_struct_path,
                "m",
            ],
        ]
        print("HPCToolkit get observation:")
        utils.print_list(hpctoolkit_cmd)
        for cmd in hpctoolkit_cmd:
            run_command(
                cmd,
                timeout=self.timeout_sec,
            )

        g_hatchet = ht.GraphFrame.from_hpctoolkit("db")

        if self.llvm_path:
            self.addInstStrToDataframe(g_hatchet, self.llvm_path)

        return g_hatchet

    def addInstStrToDataframe(self, g_hatchet: ht.GraphFrame, ll_path: str) -> None:

        inst_list = self.extractInstStr(ll_path)

        g_hatchet.dataframe["llvm_ins"] = ""

        for i, inst_idx in enumerate(g_hatchet.dataframe["line"]):
            if inst_idx < len(inst_list):
                g_hatchet.dataframe["llvm_ins"][i] = inst_list[inst_idx]

    def extractInstStr(self, ll_path: str) -> list:
        inst_list = []
        inst_list.append("dummy")

        with open(ll_path) as f:
            for line in f.readlines():
                if line[0:2] == "  " and line[2] != " ":
                    # print(len(inst_list), str(line)) # Important for Debug
                    inst_list.append(str(line[2:-1]))  # strip '  ' and '\n' at the end
        return inst_list
