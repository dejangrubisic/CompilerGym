import pdb
import subprocess
import signal

from signal import Signals
from typing import List, Optional, Tuple

from compiler_gym.util.commands import Popen, run_command



class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def run_command_stdout_redirect(cmd: List[str], timeout: int, output_file):
    with Popen(
        cmd,
        stdout=output_file,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ) as process:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            returncode = process.returncode
            try:
                # Try and decode the name of a signal. Signal returncodes
                # are negative.
                returncode = f"{returncode} ({Signals(abs(returncode)).name})"
            except ValueError:
                pass
            raise OSError(
                f"Compilation job failed with returncode {returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr.strip()}"
            )


def proto_buff_container_to_list(container):
    # Copy proto buff container to python list.
    compile_cmd = [el for el in container]
    # NOTE: if run_command function can work with proto buf containers,
    # then generating the list can be omitted. Uncomment the following statement instead.
    # compile_cmd = build_cmd.argument
    return compile_cmd


def print_list(cmd):
    depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1

    d = 0
    if len(cmd):
        d = depth(cmd)

    if d == 1:
        print(*cmd)
    elif d == 2:
        for x in cmd:
            print(*x, sep=" ")
    else:
        print(cmd)

    print("\n")
