from compiler_gym.spaces import Reward
import pickle



class Reward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """
    reward_metric = "REALTIME (sec) (I)"  # "time (inc)"

    def __init__(self):
        super().__init__(
            id="hpctoolkit",
            observation_spaces=["hpctoolkit"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )

        self.baseline_cct = None
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        print("Reward HPCToolkit: reset")
        del benchmark  # unused
        unpickled_cct = observation_view["hpctoolkit"]
        gf = pickle.loads(unpickled_cct)
        self.baseline_cct = gf
        self.baseline_runtime = gf.dataframe[self.reward_metric][0]

    def update(self, action, observations, observation_view):
        print("Reward HPCToolkit: update")
        del action
        del observation_view

        gf = pickle.loads(observations[0])
        new_runtime = gf.dataframe[self.reward_metric][0]
        return float(self.baseline_runtime - new_runtime) / self.baseline_runtime

