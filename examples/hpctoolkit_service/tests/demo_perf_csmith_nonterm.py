# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python hpctoolkit_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
import os
import pdb
import pickle
import re
import subprocess
from math import isnan
from pathlib import Path
from typing import Iterable
from compiler_gym.service.connection import ServiceError

import gym

import compiler_gym

from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.envs.llvm.datasets import (
    AnghaBenchDataset,
    BlasDataset,
    CBenchDataset,
    CBenchLegacyDataset,
    CBenchLegacyDataset2,
    CHStoneDataset,
    CsmithDataset,
    NPBDataset,
)
from compiler_gym.envs.llvm.llvm_benchmark import get_system_includes
from compiler_gym.spaces import Reward
from compiler_gym.third_party import llvm
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

reward_metric = "REALTIME (sec) (I)"  # "time (inc)"


HPCTOOLKIT_PY_SERVICE_BINARY: Path = Path(
    "hpctoolkit_service/service_py/example_service.py"
)
assert HPCTOOLKIT_PY_SERVICE_BINARY.is_file(), "Service script not found"

# BENCHMARKS_PATH: Path = runfiles_path("examples/hpctoolkit_service/benchmarks")
BENCHMARKS_PATH: Path = (
    "/home/vi3/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks"
)

HPCTOOLKIT_HEADER: Path = Path(
    "/home/vi3/CompilerGym/compiler_gym/third_party/hpctoolkit/header.h"
)


class PerfReward(Reward):
    def __init__(self):
        super().__init__(
            id="perf",
            observation_spaces=["perf"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_cycles = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        perf_dict = pickle.loads(observation_view["perf"])
        self.baseline_cycles = int(perf_dict["cycles"])
        print("Reward Perf: reset reward = ", self.baseline_cycles)

    def update(self, action, observations, observation_view):
        perf_dict = pickle.loads(observations[0])
        new_cycles = int(perf_dict["cycles"])

        print("Reward Perf: update reward = ", new_cycles)

        return float(self.baseline_cycles - new_cycles) / self.baseline_cycles


class HPCToolkitDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://hpctoolkit-cpu-v0",
            license="MIT",
            description="HPCToolkit cpu dataset",
            site_data_base=site_data_path("example_dataset"),
        )

        self._benchmarks = {
            "benchmark://hpctoolkit-cpu-v0/conv2d": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/conv2d",
                self.preprocess(BENCHMARKS_PATH + "/conv2d.c"),
            ),
            "benchmark://hpctoolkit-cpu-v0/offsets1": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/offsets1",
                self.preprocess(BENCHMARKS_PATH + "/offsets1.c"),
            ),
            "benchmark://hpctoolkit-cpu-v0/nanosleep": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/nanosleep",
                self.preprocess(BENCHMARKS_PATH + "/nanosleep.c"),
            ),
        }

    @staticmethod
    def preprocess(src: Path) -> bytes:
        """Front a C source through the compiler frontend."""
        # TODO(github.com/facebookresearch/CompilerGym/issues/325): We can skip
        # this pre-processing, or do it on the service side, once support for
        # multi-file benchmarks lands.
        cmd = [
            str(llvm.clang_path()),
            "-E",
            "-o",
            "-",
            "-I",
            str(HPCTOOLKIT_HEADER.parent),
            src,
        ]
        for directory in get_system_includes():
            cmd += ["-isystem", str(directory)]
        return subprocess.check_output(
            cmd,
            timeout=300,
        )

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        # TODO: IMPORTANT
        return self.benchmark(str(uri))


# Register the environment for use with gym.make(...).
register(
    id="perf-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": HPCTOOLKIT_PY_SERVICE_BINARY,
        "rewards": [PerfReward()],
        "datasets": [CsmithDataset(site_data_path("llvm-v0"), sort_order=0)],
    },
)


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)

    blacklisted = [
        "generator://csmith-v0/20",
        "generator://csmith-v0/22",
        "generator://csmith-v0/60",
        "generator://csmith-v0/66",
        "generator://csmith-v0/73",
        "generator://csmith-v0/81",
        "generator://csmith-v0/88",
        "generator://csmith-v0/112",
        "generator://csmith-v0/114",
        "generator://csmith-v0/118",
        "generator://csmith-v0/123",
        "generator://csmith-v0/124",
        "generator://csmith-v0/126",
        "generator://csmith-v0/134",
        "generator://csmith-v0/137",
        "generator://csmith-v0/145",
        "generator://csmith-v0/146",
        "generator://csmith-v0/148",
        "generator://csmith-v0/162",
        "generator://csmith-v0/163",
        "generator://csmith-v0/165",
        "generator://csmith-v0/169",
        "generator://csmith-v0/191",
        "generator://csmith-v0/195",
        "generator://csmith-v0/197",
        "generator://csmith-v0/203",
        "generator://csmith-v0/206",
        "generator://csmith-v0/207",
        "generator://csmith-v0/218",
        "generator://csmith-v0/247",
        "generator://csmith-v0/280",
        "generator://csmith-v0/292",
        "generator://csmith-v0/295",
        "generator://csmith-v0/298",
        "generator://csmith-v0/305",
    ]
    last_blacklisted = 0 #int(blacklisted[-1].split("/")[-1])

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("perf-v0") as env:
        inc = 0
        for bench in blacklisted: 
        
            inc += 1


            try:
                env.reset(benchmark=bench)
            except ServiceError:
                print("AGENT: Timeout Error Reset")
                continue


            print("********************* Train on ", bench, "*********************")
            for i in range(1):
                print("Main: step = ", i)
                try:
                    observation, reward, done, info = env.step(
                        action=env.action_space.sample(),
                        observations=["perf"],
                        rewards=["perf"],
                    )
                except ServiceError:
                    print("AGENT: Timeout Error Step")
                    continue
                    

                print(reward)
                # print(observation)
                print(info)
                perf_dict = pickle.loads(observation[0])
                print(perf_dict)

                # pdb.set_trace()

                if isnan(reward[0]):
                    print(bench, " Failed with Nan reward")
                    return

                if done:                    
                    continue


        print("I run %d benchmarks." % inc)


if __name__ == "__main__":
    main()
