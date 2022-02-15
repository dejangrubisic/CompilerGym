import hatchet as ht
import pandas as pd
import programl as pg
import pickle
import pdb
from typing import Dict, List, Optional, Tuple
from compiler_gym.util.commands import run_command

from compiler_gym.service.proto import (
    Observation,
)


from profilers import (
    hpctoolkit,
    programl
)


class Profiler:
    def __init__(self, run_cmd, timeout_sec, src_path=None):
        self.run_cmd = run_cmd
        self.timeout_sec = timeout_sec
        self.llvm_path = src_path
        self.hpctoolkit = hpctoolkit.Profiler(run_cmd, timeout_sec, src_path)
        self.programl = programl.Profiler(run_cmd, timeout_sec, src_path)

        # TODO: Get rid of this
        # List of metrics collected from observation space
        self.features_hpcrun = [
            "-e",
            "REALTIME@100",
        ]  # TODO: Use this in hpcrun command
        self.features_hatchet = [
            "REALTIME (sec) (I)",
            "REALTIME (sec) (E)",
        ]  # TODO: Use this from hatchat dataframe


    def get_observation(self) -> Observation:
        g_hatchet = self.hpctoolkit.hatchet_get_graph()
        g_programl = self.programl.programl_get_graph(self.llvm_path)
        g_programl = self.programl_add_features(
            g_programl, g_hatchet, self.features_hatchet
        )
        pickled = pickle.dumps(g_programl)
        return Observation(binary_value=pickled)

    def programl_get_graph(self, ll_path: str) -> pg.ProgramGraph:

        with open(ll_path, "r") as f:
            code_str = f.read().rstrip()
            g_programl = pg.from_llvm_ir(code_str)
            g_programl = pg.to_networkx(g_programl)

        return g_programl


    def programl_add_features(
        self, g_programl: pg.ProgramGraph, g_hatchet: ht.GraphFrame, feature_names: list
    ) -> pg.ProgramGraph:

        df = g_hatchet.dataframe.sort_values(by=["line"])

        # The node 0 carries the information about <program root>
        hatchet_root = df[df["name"] == "<program root>"]
        g_programl.nodes[0]["features"] = {
            "dynamic": [sum(hatchet_root[fn]) for fn in feature_names]
        }

        ins_line = 1
        for n_id in list(g_programl.nodes())[1:]:
            node = g_programl.nodes[n_id]

            if node["type"] == 0 and "features" in node and node["features"]["full_text"][0] != '':  # instruction
                hatchet_row = df[
                    df["line"] == ins_line
                ]  # if there is multiple sum them

                # print(ins_line, node["features"]["full_text"][0]) # Important for debug
                if node["features"]["full_text"][0] == '':
                    pdb.set_trace()

                ins_line += 1

                if hatchet_row.empty == True:
                    node["features"]["dynamic"] = [0] * len(feature_names)
                else:
                    node["features"]["dynamic"] = [
                        sum(hatchet_row[fn]) for fn in feature_names
                    ]

                    # Assert with pdb.set_trace()
                    if hatchet_row["llvm_ins"][0].split('!')[0] != node["features"]["full_text"][0].split('!')[0]:
                        print("\n", "ERROR: hatchat llvm_ins different from programl llvm_ins", "\n")
                        print(hatchet_row["llvm_ins"], node["features"]["full_text"][0])

                        # Debug save files
                        g_df = pd.DataFrame.from_dict(dict(g_programl.nodes(data=True)), orient="index")
                        f_df = pd.DataFrame.from_dict(dict(g_df["features"]), orient="index")

                        g_df = pd.concat([g_df, f_df], axis=1)
                        g_df.drop("features", axis=1, inplace=True)
                        g_df.to_csv("programl.csv", index=False)
                        df.to_csv("hatchet.csv", index=False)
                        pdb.set_trace()

                        print(df[["full_text", "dynamic"]])

                # print(node['features']['dynamic'])

        return g_programl        