"""Bounded readiness and cleanup for processes owned by this campaign only."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def listening(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


class Services:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.children = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def start(self, name, argv, cwd, env, port, timeout=180, host='127.0.0.1'):
        if listening(host, port):
            raise RuntimeError(f'{name}: {host}:{port} already in use; choose another port')
        log = (self.directory / (name + '.log')).open('w')
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.children.append((process, log))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited ({process.returncode}); see {log.name}')
            if listening(host, port):
                return process
            time.sleep(0.2)
        raise TimeoutError(f'{name} did not become ready; see {log.name}')

    def __exit__(self, *exc):
        # Reverse dependency order; never signal processes found through ports or names.
        for process, log in reversed(self.children):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except ProcessLookupError:
                    pass
            log.close()
