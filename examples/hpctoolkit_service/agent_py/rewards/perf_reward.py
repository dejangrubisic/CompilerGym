from compiler_gym.spaces import Reward
import pickle



class Reward(Reward):
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
        # print("Reward Perf: reset reward = ", self.baseline_cycles)

    def update(self, action, observations, observation_view):
        perf_dict = pickle.loads(observations[0])
        new_cycles = int(perf_dict["cycles"])

        # print("Reward Perf: update reward = ", new_cycles)

        return float(self.baseline_cycles - new_cycles) / self.baseline_cycles
