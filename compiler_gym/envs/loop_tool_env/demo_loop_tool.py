# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python example_compiler_gym_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
from pathlib import Path
from typing import Iterable
import pdb
# import gym

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

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


import loop_tool_service
from loop_tool_service.paths import LOOP_TOOL_SERVICE_PY
from loop_tool_service.service.datasets import simple_dataset



class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="flops",
            observation_spaces=["flops"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]

        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


# class ExampleDataset(Dataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             name="benchmark://example-v0",
#             license="MIT",
#             description="An example dataset",
#         )
#         self._benchmarks = {
#             "/foo": Benchmark.from_file_contents(
#                 "benchmark://example-v0/foo", "Ir data".encode("utf-8")
#             ),
#             "/bar": Benchmark.from_file_contents(
#                 "benchmark://example-v0/bar", "Ir data".encode("utf-8")
#             ),
#         }

#     def benchmark_uris(self) -> Iterable[str]:
#         yield from (f"benchmark://example-v0{k}" for k in self._benchmarks.keys())

#     def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
#         if uri.path in self._benchmarks:
#             return self._benchmarks[uri.path]
#         else:
#             raise LookupError("Unknown program name")


def register_env():
    register(
        id="loop_tool-v0",
        entry_point=loop_tool_service.LoopToolEnv,
        kwargs={
            "service": LOOP_TOOL_SERVICE_PY,
            "rewards": [RuntimeReward()],
            "datasets": [
                simple_dataset.Dataset(),
                CBenchDataset(site_data_path("llvm-v0"))],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)
    register_env()

    # Create the environment using the regular gym.make(...) interface.
    with loop_tool_service.make_env("loop_tool-v0") as env:
        pdb.set_trace()
        for bench in env.datasets["benchmark://cbench-v1"]:
        # for bench in env.datasets["benchmark://loop_tool_simple-v0"]:
            pdb.set_trace()
            try:
                env.reset(benchmark=bench)
                # env.send_param("timeout_sec", "1")
            except ServiceError:
                print("AGENT: Timeout Error Reset")
                continue
                            
                try:
                    observation, reward, done, info = env.step(
                        action=env.action_space.sample(),
                        observation_spaces=["flops"],
                        reward_spaces=["simple"],
                    )
                except ServiceError:
                    print("AGENT: Timeout Error Step")
                    continue
                
                print(reward)
                # print(observation)
                print(info)

                if type(observation[0]) == np.ndarray:
                    print(observation[0])
                else:
                    perf_dict = pickle.loads(observation[0])
                    print(perf_dict)  


if __name__ == "__main__":
    main()
