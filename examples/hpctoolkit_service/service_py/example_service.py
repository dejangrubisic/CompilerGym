#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
from cmath import nan
import logging
import os
import pdb
import pickle
import shutil
import subprocess

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hatchet as ht
import numpy as np
import programl as pg
import pandas as pd
import os
import benchmark_builder

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


from profilers import (
    runtime,
    perf,
    hpctoolkit,
    programl,
    programl_hpctoolkit
)

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
        ObservationSpace(
            name="perf",
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
        self._action_space = action_space
        
        print("\n", str(working_directory), "\n")
        os.chdir(str(working_directory))

        pdb.set_trace()

        self.benchmark  = benchmark_builder.BenchmarkBuilder(working_directory, benchmark)
        self.runtime    = runtime.Profiler(self.benchmark.run_cmd)
        self.perf       = perf.Profiler(self.benchmark.run_cmd)
        self.hpctoolkit = hpctoolkit.Profiler(self.benchmark.run_cmd, self.benchmark.llvm_path)
        self.programl   = programl.Profiler(self.benchmark.run_cmd, self.benchmark.llvm_path)
        self.programl_hpctoolkit = programl_hpctoolkit.Profiler(self.benchmark.run_cmd, self.benchmark.llvm_path)


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

        self.benchmark.apply_action(opt)

        # TODO: Dejan properly implement these
        action_had_no_effect = False
        end_of_session = False
        new_action_space = None
        return (end_of_session, new_action_space, action_had_no_effect)

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space.name)

        if observation_space.name == "runtime":
            print("get_observation: runtime")
            return self.runtime.get_observation()

        if observation_space.name == "perf":
            print("get_observation: perf")            
            return self.perf.get_observation()

        elif observation_space.name == "hpctoolkit":
            print("get_observation: hpctoolkit")            
            return self.hpctoolkit.get_observation()

        elif observation_space.name == "programl":
            print("get_observation: programl")            
            return self.programl.get_observation()

        elif observation_space.name == "programl_hpctoolkit":
            print("get_observation: programl_hpctoolkit")
            return self.programl_hpctoolkit.get_observation()
        else:
            raise KeyError(observation_space.name)



if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
