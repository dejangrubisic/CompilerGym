import pickle
from compiler_gym.service.proto import (
    Observation,
)




class Perf:
    def __init__(self, run_cmd):
        self.run_cmd = run_cmd

    def get_observation(self) -> Observation:
        perf_dict = self.perf_get_dict()            
        pickled = pickle.dumps(perf_dict)
        return Observation(binary_value=pickled)