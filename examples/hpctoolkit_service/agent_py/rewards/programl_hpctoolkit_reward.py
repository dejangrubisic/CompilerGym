from compiler_gym.spaces import Reward
import pickle



class Reward(Reward):
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
