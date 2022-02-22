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

from compiler_gym.datasets import Benchmark, Dataset, BenchmarkUri
from compiler_gym.envs.llvm.llvm_benchmark import get_system_includes
from compiler_gym.spaces import Reward
from compiler_gym.third_party import llvm
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.service.connection import ServiceError
import utils

from agent_py.rewards import programl_reward
from agent_py.datasets import hpctoolkit_dataset


def register_env():
    register(
        id="hpctoolkit-llvm-v0",
        entry_point="compiler_gym.envs:CompilerEnv",
        kwargs={
            "service": utils.HPCTOOLKIT_PY_SERVICE_BINARY,
            "rewards": [ programl_reward.Reward()],
            "datasets": [hpctoolkit_dataset.Dataset()],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)
    register_env()

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("hpctoolkit-llvm-v0") as env:
        try:
            # env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/offsets1")
            env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/conv2d")
            # env.reset(benchmark="benchmark://hpctoolkit-cpu-v0/nanosleep")
        except ServiceError:
            print("AGENT: Timeout Error Reset")

        for i in range(2):
            print("Main: step = ", i)
            try:
                observation, reward, done, info = env.step(
                    action=env.action_space.sample(),
                    observations=["programl"],
                    rewards=["programl"],
                )
            except ServiceError:
                print("AGENT: Timeout Error Step")
                continue      
                  
            print(reward)
            print(info)
            g = pickle.loads(observation[0])
            print(g.nodes())
            

            pdb.set_trace()
            if done:
                env.reset()


if __name__ == "__main__":
    main()
