#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import os
import pdb
import pickle
import shutil
import subprocess
from bdb import set_trace
from pathlib import Path
from typing import List, Optional, Tuple

import hatchet as ht
import numpy as np
import programl as pg
import utils

import compiler_gym.third_party.llvm as llvm
from compiler_gym import site_data_path
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    Action,
    ActionSpace,
    Benchmark,
    ChoiceSpace,
    NamedDiscreteSpace,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from compiler_gym.util.commands import run_command


class HPCToolkitCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    action_spaces = [
        ActionSpace(
            name="llvm",
            choice=[
                ChoiceSpace(
                    name="optimization_choice",
                    named_discrete_space=NamedDiscreteSpace(
                        value=[
                            "-O0",
                            "-O1",
                            "-O2",
                            "-O3",
                        ],
                    ),
                )
            ],
        ),
    ]

    # A list of observation spaces supported by this service. Each of these
    # ObservationSpace protos describes an observation space.
    observation_spaces = [
        ObservationSpace(
            name="runtime",
            scalar_double_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=False,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
        ObservationSpace(
            name="hpctoolkit",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
            ),
        ),
        ObservationSpace(
            name="programl",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
            ),
        ),
        ObservationSpace(
            name="programl_hpctoolkit",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
            ),
        ),
    ]

    def __init__(
        self,
        working_directory: Path,
        action_space: ActionSpace,
        benchmark: Benchmark,  # TODO: Dejan use Benchmark rather than hardcoding benchmark path here!
        # use_custom_opt: bool = True,
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)
        self._benchmark = benchmark
        self._action_space = action_space

        # Resolve the paths to LLVM binaries once now.
        self._clang = str(llvm.clang_path())
        self._llvm_dis = str(llvm.llvm_dis_path())
        self._llc = str(llvm.llc_path())
        self._llvm_diff = str(llvm.llvm_diff_path())
        self._opt = str(llvm.opt_path())

        self._llvm_path = str(self.working_dir / "benchmark.ll")
        self._llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self._bc_path = str(self.working_dir / "benchmark.bc")
        self._bc_download_path = str(self.working_dir / "benchmark-downloaded.bc")
        self._exe_path = str(self.working_dir / "benchmark.exe")
        self._exe_struct_path = self._exe_path + ".hpcstruct"

        # Set run commands
        self._run_exe = [self._exe_path]

        # List of metrics collected from observation space
        self._features_hpcrun = [
            "-e",
            "REALTIME@100",
        ]  # TODO: Use this in hpcrun command
        self._features_hatchet = [
            "REALTIME (sec) (I)",
            "REALTIME (sec) (E)",
        ]  # TODO: Use this from hatchat dataframe

        # Dump the benchmark source to disk.
        self._src_path = str(self.working_dir / "benchmark.c")
        with open(self._src_path, "wb") as f:
            f.write(benchmark.program.contents)



        # Set compile commands
        if action_space.name == "llvm":
            compile_to_ll = [
                self._clang,
                "-o",
                self._llvm_path,
                "-S",
                "-emit-llvm",
                self._src_path,
            ]

            # Optimization must go on position: self.optimize_and_save_ll[0][1]
            self.optimize_and_save_ll = [
                [self._opt, "-O0", "--debugify", "-o", self._bc_path, self._llvm_path],
                [self._clang, self._bc_path, "-o", self._exe_path],
                [self._llvm_dis, "-o", self._llvm_path, self._bc_path]
            ]

        else:
            print("Action space is doesn't exits: ", action_space)
            exit


        # Transform input code representation to ll format
        run_command(
            compile_to_ll,
            timeout=30,
        )

        # Compile baseline at the beginning
        for cmd in self.optimize_and_save_ll:
            run_command(
                cmd,
                timeout=30,
            )


        print("\n", self.working_dir, "\n")
        pdb.set_trace()

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:

        num_choices = len(self.action_spaces[0].choice[0].named_discrete_space.value)

        if len(action.choice) != 1:
            raise ValueError("Invalid choice count")

        choice_index = action.choice[0].named_discrete_value_index
        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")

        # Compile benchmark with given optimization
        opt = self._action_space.choice[0].named_discrete_space.value[choice_index]
        logging.info(
            "Applying action %d, equivalent command-line arguments: '%s'",
            choice_index,
            opt,
        )

        self.optimize_and_save_ll[0][1] = opt
        for cmd in self.optimize_and_save_ll:
            run_command(
                cmd,
                timeout=30,
            )

        # TODO: Dejan properly implement these
        action_had_no_effect = False
        end_of_session = False
        new_action_space = None
        return (end_of_session, new_action_space, action_had_no_effect)

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space.name)

        if observation_space.name == "runtime":
            print("get_observation: runtime")
            avg_exec_time = self.runtime_get_average()
            return Observation(scalar_double=avg_exec_time)

        elif observation_space.name == "hpctoolkit":
            print("get_observation: hpctoolkit")
            g_hatchet = self.hatchet_get_graph()
            pickled = pickle.dumps(g_hatchet)
            return Observation(binary_value=pickled)

        elif observation_space.name == "programl":
            print("get_observation: programl")
            g_programl = self.programl_get_graph(self._llvm_path)
            pickled = pickle.dumps(g_programl)
            return Observation(binary_value=pickled)

        elif observation_space.name == "programl_hpctoolkit":
            print("get_observation: programl_hpctoolkit")
            g_hatchet = self.hatchet_get_graph()
            g_programl = self.programl_get_graph(self._llvm_path)
            g_programl = self.programl_add_features(
                g_programl, g_hatchet, self._features_hatchet
            )
            pickled = pickle.dumps(g_programl)
            return Observation(binary_value=pickled)

        else:
            raise KeyError(observation_space.name)

    ##########################################################################
    # Observation functions
    ##########################################################################

    def runtime_get_average(self) -> float:

        # TODO: add documentation that benchmarks need print out execution time
        # Running 5 times and taking the average of middle 3
        exec_times = []
        for _ in range(5):
            stdout = run_command(
                self._run_exe,
                timeout=30,
            )
            print(stdout)
            try:
                exec_times.append(int(stdout))
            except ValueError:
                raise ValueError(
                    f"Error in parsing execution time from output of command\n"
                    f"Please ensure that the source code of the benchmark measures execution time and prints to stdout\n"
                    f"Stdout of the program: {stdout}"
                )

        exec_times = np.sort(exec_times)
        avg_exec_time = np.mean(exec_times[1:4])
        return avg_exec_time

    def hatchet_get_graph(self) -> ht.GraphFrame:
        hpctoolkit_cmd = [
            [
                "rm",
                "-rf",
                self._exe_struct_path,
                self.working_dir / "m",
                self.working_dir / "db",
            ],
            [
                "hpcrun",
                "-e",
                "REALTIME@100",
                "-t",
                "-o",
                self.working_dir / "m",
                self._exe_path,
            ],
            ["hpcstruct", "-o", self._exe_struct_path, self._exe_path],
            [
                "hpcprof-mpi",
                "-o",
                self.working_dir / "db",
                "--metric-db",
                "yes",
                "-S",
                self._exe_struct_path,
                self.working_dir / "m",
            ],
        ]

        for cmd in hpctoolkit_cmd:
            print(cmd)

            run_command(
                cmd,
                timeout=30,
            )

        g_hatchet = ht.GraphFrame.from_hpctoolkit(str(self.working_dir / "db"))
        self.addInstStrToDataframe(g_hatchet, self._llvm_path)

        return g_hatchet

    def programl_get_graph(self, ll_path: str) -> pg.ProgramGraph:

        with open(ll_path, "r") as f:
            code_str = f.read().rstrip()
            g_programl = pg.from_llvm_ir(code_str)
            g_programl = pg.to_networkx(g_programl)

        return g_programl

    ##########################################################################
    # Auxilary functions
    ##########################################################################

    def extractInstStr(self, ll_path: str) -> list:
        inst_list = []
        inst_list.append("dummy")

        with open(ll_path) as f:
            for line in f.readlines():
                if line[0:2] == "  " and line[2] != " ":
                    # print(len(inst_list), str(line))
                    inst_list.append(str(line[2:-1]))  # strip '  ' and '\n' at the end
        return inst_list

    def addInstStrToDataframe(self, g_hatchet: ht.GraphFrame, ll_path: str) -> None:

        inst_list = self.extractInstStr(ll_path)

        g_hatchet.dataframe["llvm_ins"] = ""

        for i, inst_idx in enumerate(g_hatchet.dataframe["line"]):
            if inst_idx < len(inst_list):
                g_hatchet.dataframe["llvm_ins"][i] = inst_list[inst_idx]

    def programl_add_features(
        self, g_programl: pg.ProgramGraph, g_hatchet: ht.GraphFrame, feature_names: list
    ) -> pg.ProgramGraph:

        df = g_hatchet.dataframe.sort_values(by=["line"])

        # The node 0 carries the information about <program root>
        hatchet_root = df[df["name"] == "<program root>"]
        g_programl.nodes[0]["features"] = {
            "dynamic": [sum(hatchet_root[fn]) for fn in feature_names]
        }

        ins_line = 1
        for n_id in list(g_programl.nodes())[1:]:
            node = g_programl.nodes[n_id]

            if node["type"] == 0 and "features" in node:  # instruction
                hatchet_row = df[
                    df["line"] == ins_line
                ]  # if there is multiple sum them
                ins_line += 1

                if hatchet_row.empty == True:
                    node["features"]["dynamic"] = [0] * len(feature_names)
                else:
                    node["features"]["dynamic"] = [
                        sum(hatchet_row[fn]) for fn in feature_names
                    ]

                    # Assert with pdb.set_trace()
                    if hatchet_row["llvm_ins"][0] != node["features"]["full_text"][0]:
                        print("hatchat llvm_ins different from programl llvm_ins")
                        pdb.set_trace()
                        print(hatchet_row["llvm_ins"], node["features"]["full_text"][0])

                # print(node['features']['dynamic'])

        return g_programl


if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
