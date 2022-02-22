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
import pathlib 
import utils

from agent_py.rewards import perf_reward



def register_env():
    # Register the environment for use with gym.make(...).
    register(
        id="perf-v0",
        entry_point="compiler_gym.envs:CompilerEnv",
        kwargs={
            "service": utils.HPCTOOLKIT_PY_SERVICE_BINARY,
            "rewards": [perf_reward.Reward()],
            "datasets": [CsmithDataset(site_data_path("llvm-v0"), sort_order=0)],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)
    register_env()

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
    last_blacklisted = int(blacklisted[-1].split("/")[-1])

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("perf-v0") as env:
        inc = 0
        for bench in env.datasets["generator://csmith-v0"]:
            # for i in range(last_blacklisted, 10000):
            # bench = 'generator://csmith-v0/' + str(i)
            inc += 1

            if bench in blacklisted:
                continue

            try:
                env.reset(benchmark=bench)
                # env.send_param("timeout_sec", "1")
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
