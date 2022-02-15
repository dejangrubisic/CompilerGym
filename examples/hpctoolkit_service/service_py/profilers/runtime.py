import numpy as np
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


    def get_observation(self) -> Observation:
        avg_exec_time = self.runtime_get_average()
        return Observation(scalar_double=avg_exec_time)


    def runtime_get_average(self) -> float:
        # TODO: add documentation that benchmarks need print out execution time
        # Running 5 times and taking the average of middle 3
        exec_times = []
        
        with open('/dev/null', 'w') as f:

            for _ in range(5):
                # stdout = benchmark_builder.run_command_stdout_redirect(
                #     ['time'] + self.benchmark.run_cmd,
                #     timeout=self.timeout_sec,
                #     output_file=f
                # )
                # print(stdout)
                exec_time = 1 # TODO: Figure out how to parse time to int and to direct output to /dev/null

                try:
                    exec_times.append(exec_time)
                except ValueError:
                    raise ValueError(
                        f"Error in parsing execution time from output of command\n"
                        f"Please ensure that the source code of the benchmark measures execution time and prints to stdout\n"
                        f"Stdout of the program: {stdout}"
                    )

        exec_times = np.sort(exec_times)
        avg_exec_time = np.mean(exec_times[1:4])
        return avg_exec_time