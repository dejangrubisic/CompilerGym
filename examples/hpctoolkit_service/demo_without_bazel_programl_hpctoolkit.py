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
import subprocess
from pathlib import Path
from typing import Iterable

import gym
import hatchet as ht
import pandas as pd

from compiler_gym.datasets import Benchmark, Dataset  # , BenchmarkUri
from compiler_gym.envs.llvm.datasets import (
    CBenchDataset,
    CBenchLegacyDataset,
    CBenchLegacyDataset2,
)
from compiler_gym.envs.llvm.llvm_benchmark import get_system_includes
from compiler_gym.spaces import Reward
from compiler_gym.third_party import llvm
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

pd.set_option("display.max_columns", None)

collected_metrics = ["REALTIME (sec) (I)", "REALTIME (sec) (E)"]


HPCTOOLKIT_PY_SERVICE_BINARY: Path = Path(
    "hpctoolkit_service/service_py/example_service.py"
)
assert HPCTOOLKIT_PY_SERVICE_BINARY.is_file(), "Service script not found"

# BENCHMARKS_PATH: Path = runfiles_path("examples/hpctoolkit_service/benchmarks")
BENCHMARKS_PATH: Path = (
    "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks"
)

HPCTOOLKIT_HEADER: Path = Path(
    "/home/dx4/tools/CompilerGym/compiler_gym/third_party/hpctoolkit/header.h"
)


class ProgramlHPCToolkitReward(Reward):
    """An example reward that uses changes in the "programl" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="programl_hpctoolkit",
            observation_spaces=["programl_hpctoolkit"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        print("Reward ProgramlHPCToolkit: reset")
        del benchmark  # unused
        unpickled_cct = observation_view["programl_hpctoolkit"]
        g = pickle.loads(unpickled_cct)
        self.baseline_runtime = g.nodes[0]["features"]["dynamic"][0]

    def update(self, action, observations, observation_view):
        print("Reward ProgramlHPCToolkit: update")
        del action
        del observation_view
        g = pickle.loads(observations[0])
        new_runtime = g.nodes[0]["features"]["dynamic"][
            0
        ]  # nodes[0] is root, get features, dynamic = ['REALTIME (sec) (I)','REALTIME (sec) (E)']

        return float(self.baseline_runtime - new_runtime) / self.baseline_runtime


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

    # def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
    #     # TODO: IMPORTANT
    #     return self.benchmark(str(uri))


# Register the environment for use with gym.make(...).
register(
    id="hpctoolkit-llvm-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": HPCTOOLKIT_PY_SERVICE_BINARY,
        "rewards": [ProgramlHPCToolkitReward()],
        # "datasets": [HPCToolkitDataset(), CBenchLegacyDataset2(site_data_path("llvm-v0"))],
        "datasets": [HPCToolkitDataset()],
    },
)


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("hpctoolkit-llvm-v0") as env:
        # env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/offsets1")
        env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/conv2d")
        # env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/nanosleep")

        # env.reset(benchmark="benchmark://cbench-v1/qsort")

        for i in range(2):
            print("Main: step = ", i)
            observation, reward, done, info = env.step(
                action=env.action_space.sample(),
                observations=["programl_hpctoolkit"],
                rewards=["programl_hpctoolkit"],
            )
            print(reward)
            print(info)
            g = pickle.loads(observation[0])
            g_df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
            f_df = pd.DataFrame.from_dict(dict(g_df["features"]), orient="index")

            df = pd.concat([g_df, f_df], axis=1)
            df.drop("features", axis=1, inplace=True)

            print(df[["full_text", "dynamic"]])
            # g_df.to_csv( self.working_dir / "programl.csv", index=False)

            pdb.set_trace()
            if done:
                env.reset()


if __name__ == "__main__":
    main()
