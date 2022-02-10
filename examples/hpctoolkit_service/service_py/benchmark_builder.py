# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import subprocess
import os
import logging

from pathlib import Path

from copy import deepcopy as deepcopy
import compiler_gym.third_party.llvm as llvm
from compiler_gym.util.commands import run_command, Popen
from compiler_gym.service.proto import Benchmark
from signal import Signals
from typing import List, Optional, Tuple
import utils



## Build benchmarks
class BenchmarkBuilder:
    
    def __init__(self, working_directory: Path, benchmark: Benchmark):
        # pdb.set_trace()
        print(benchmark.uri)

        self.clang = str(llvm.clang_path())
        self.llvm_dis = str(llvm.llvm_dis_path())
        self.llc = str(llvm.llc_path())
        self.llvm_diff = str(llvm.llvm_diff_path())
        self.opt = str(llvm.opt_path())

        self.working_dir = working_directory
        self.llvm_path = str(self.working_dir / "benchmark.ll")
        self.llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self.bc_path = str(self.working_dir / "benchmark.bc")
        # vi3: Used only if the benchmark is downloaded in .bc format.
        self.bc_download_path = str(self.working_dir / "benchmark-downloaded.bc")
        self.exe_path = str(self.working_dir / "benchmark.exe")

        # TODO: Temporary
        self.pre_run_cmd = []
        self.run_cmd = [self.exe_path]

        self.compile_ll = {
            'opt':  [self.opt, "--debugify", "-o", self.bc_path, self.llvm_path],
            'cmp':  [self.clang, self.bc_path, "-o", self.exe_path, '-lm'], # "-nostartfiles"
            'save': [self.llvm_dis, "-o", self.llvm_path, self.bc_path]
        }

        self.prepare_benchmark(benchmark)
        

    def prepare_benchmark(self, benchmark: Benchmark):
        self.save_to_ll(benchmark)        
        self.set_build_run_cmd(benchmark)
        self.execute_pre_run_cmd()
        self.apply_action("-O0")


    def save_to_ll(self, benchmark: Benchmark):
        print(benchmark.uri)

        if benchmark.program.contents.startswith(b'BC'):        
            _src_path = str(self.working_dir / "benchmark-downloaded.bc")
            with open(_src_path, 'wb') as f:
                f.write(benchmark.program.contents)
            
            compile_to_ll = [
                self.llvm_dis, 
                "-o", 
                self.llvm_path, 
                _src_path
            ]

        else: # If Benchmark is in C
            _src_path = str(self.working_dir / "benchmark.c")
            with open(_src_path, "wb") as f:
                f.write(benchmark.program.contents)

            compile_to_ll = [
                self.clang,
                "-o",
                self.llvm_path,
                "-S",
                "-emit-llvm",
                _src_path,
            ]     


        # Transform input code representation to ll format
        run_command(
            compile_to_ll,
            timeout=30,
        )


    def set_build_run_cmd(self, benchmark):

        if hasattr(benchmark, 'dynamic_config'):
            if hasattr(benchmark.dynamic_config, 'build_cmd'):                
                self.prepare_build_cmd(benchmark.dynamic_config.build_cmd)
    
            if hasattr(benchmark.dynamic_config, 'pre_run_cmd'):
                self.prepare_pre_run_cmd(benchmark.dynamic_config.pre_run_cmd)
        
            if hasattr(benchmark.dynamic_config, 'run_cmd'):
                self.prepare_run_cmd(benchmark.dynamic_config.run_cmd)




    def execute_pre_run_cmd(self):
        # The stdout may be redirected to a file.
        for cmd in self.pre_run_cmd:
            pre_run_cmd_with_redirection(cmd, self.working_dir)

    def pre_run_cmd_with_redirection(cmd, working_dir):
        """
        Pre run command might redirect the stdout to a file.
        Try to recignize this.
        """
        # FIXME: This function assumes that the last argument may contain the files
        #   to which the stdout should be redirected.
        if cmd[-1].startswith('>'):
            # Remove '>' from file name and prepend working_dir path
            file_path = working_dir / cmd[-1][1:]
            with open(file_path, 'w') as f:
                # Remove the stdout redirection from the cmd.
                cmd = cmd[:-1]
                utils.run_command_stdout_redirect(cmd, timeout=30, output_file=f)
        else:
            # Execute the command with no redirection
            run_command(cmd, timeout=30)
            

    def apply_action(self, opt: str):        
        compile_ll = deepcopy(self.compile_ll)
        compile_ll["opt"].insert(1, opt)

        for cmd in compile_ll.values():
            run_command(
                cmd,
                timeout=30,
            )

    
    # Prepare build, pre_run and run commands
    def prepare_build_cmd(self, build_cmd):
        """
        build_cmd is in the following format.
        build_cmd {
          argument: "$CC"
          argument: "$IN"
          argument: "-lm"
          timeout_seconds: 60
          outfile: "a.out"
        }
        Only argument list with the following form: ["$CC", "$IN", "-lm"] is processed.
        """
        compile_cmd = utils.proto_buff_container_to_list(build_cmd.argument)
        if compile_cmd:
            # Just for the debugging purposes.
            assert compile_cmd[0] == "$CC"
            assert compile_cmd[1] == "$IN"
            # Replace $CC with the llvm_clang.
            compile_cmd[0] = self.clang
            # Replace $IN with path to the .bc files
            compile_cmd[1] = self.bc_path
            # Append the output file of the compile command.
            compile_cmd.extend(["-o", self.exe_path])
            # Add the last compile command.
            self.compile_ll['cmp'] = compile_cmd


    def prepare_pre_run_cmd(self, pre_run_cmd):
        """
        pre_run_cmd is in the following format
        pre_run_cmd {
          argument: "echo"
          argument: "1"
          argument: ">_finfo_dataset"
          timeout_seconds: 30
        }
        Only argument is considered for now.
        """
        for cmd in pre_run_cmd:
            pre_run_cmd_list = utils.proto_buff_container_to_list(cmd.argument)
            if pre_run_cmd_list:
                # append pre_run_cmd list to the pre_run_cmd
                self.pre_run_cmd.append(pre_run_cmd_list)


    def prepare_run_cmd(self, run_cmd):
        """
        run_cmd is in the following format:
        run_cmd {
          argument: "./a.out"
          argument: "/home/vi3/.local/share/compiler_gym/llvm-v0/cbench-v1-runtime-data/runtime_data/automotive_qsort_data/1.dat"
          timeout_seconds: 300
          infile: "a.out"
          infile: "_finfo_dataset"
          outfile: "sorted_output.dat"
        }
        Only argument is considered for now.
        """
        print("________ Run command: \n", run_cmd)
        run_cmd = utils.proto_buff_container_to_list(run_cmd.argument)
        if run_cmd:
            # Just for the debugging purposes.
            assert run_cmd[0] == './a.out'
            # Replace ./a.out with the exe_path
            run_cmd[0] = self.exe_path
            self.run_cmd = run_cmd
