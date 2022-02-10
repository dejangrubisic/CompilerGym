import pandas as pd
import pickle
import pdb
from typing import Dict, List, Optional, Tuple
from compiler_gym.util.commands import run_command

from compiler_gym.service.proto import (
    Observation,
)


class Profiler:
    def __init__(self, run_cmd, src_path=None):
        self.run_cmd = run_cmd



    def get_observation(self) -> Observation:
        perf_dict = self.perf_get_dict()            
        pickled = pickle.dumps(perf_dict)
        return Observation(binary_value=pickled)


    def perf_get_dict(self) -> Dict:

        # perf stat -o metric_out.csv -d -d -d -x ',' ./benchmark.exe 1125000
        # perf stat -d -d -d -x ',' ./benchmark.exe 1125000 # much faster
        metric_file_name = "metrics.csv"
        perf_cmd = ['perf', 'stat', '-o', metric_file_name, '-d', '-d', '-d', '-x', ','] + self.run_cmd

        stdout = run_command(
            perf_cmd,
            timeout=30,                
        )
        return self.perf_parse_to_dict(metric_file_name)


    def perf_parse_to_dict(self, csv_name: str) -> Dict:

        # if 'r' there will be column for std +- var
        column_names = ['counter_value', 'counter_unit', 'event_name', 'counter_runtime', 'counter_runtime_perc', 'metric_value', 'metric_unit']
        df = pd.read_csv(csv_name, names=column_names)
        assert(len(column_names) == df.shape[1])

        df = df[df['counter_value'] != '<not supported>'][df['event_name'].notnull()]

        return dict(zip(df['event_name'], df['metric_value']))         