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
import os
import pdb
import pickle
import sys
from pathlib import Path
from typing import Iterable

import gym
import loop_tool as lt

# import gym
import numpy as np
from service.datasets import loop_tool_dataset
from service.rewards import flops_reward, runtime_reward

import compiler_gym
from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
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
from compiler_gym.service.connection import ServiceError
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path


def register_env():
    register(
        id="loop_tool-v0",
        entry_point=compiler_gym.envs.loop_tool.LoopToolCompilerEnv,
        kwargs={
            "service": compiler_gym.envs.loop_tool.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [runtime_reward.Reward(), flops_reward.Reward()],
            "datasets": [
                loop_tool_dataset.Dataset(),
                CBenchDataset(site_data_path("llvm-v0")),
            ],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)
    register_env()

    actions = ["down", "down", "split_64", "down", "split_16", "swap_down"]

    with compiler_gym.envs.loop_tool.make_env("loop_tool-v0") as env:
        for bench in env.datasets["benchmark://loop_tool_simple-v0"]:
            try:
                env.reset(benchmark=bench)
                # env.send_param("timeout_sec", "1")
            except ServiceError:
                print("AGENT: Timeout Error Reset")
                continue

            for a in actions:
                try:
                    observation, reward, done, info = env.step(
                        action=env.action_space.from_string(a),
                        observation_spaces=["loop_tree_ir"],
                        reward_spaces=["flops"],
                    )
                except ServiceError:
                    print("AGENT: Timeout Error Step")
                    continue

                print(f"{reward}\n")
                print(f"{info}\n")

                try:
                    tensor = lt.Tensor()
                    tensor.set(lt.deserialize(observation[0]))
                    print(tensor.loop_tree)
                except:
                    print(f"{observation}\n")

                pdb.set_trace()


if __name__ == "__main__":
    main()
