from __future__ import annotations

import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import RESULT_DIR


@dataclass(frozen=True)
class ReportServer:
    process: subprocess.Popen
    host: str
    port: int
    root: Path

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


def start_report_server(host: str = "0.0.0.0") -> ReportServer:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    port = find_free_port(host)
    cmd = [sys.executable, "-m", "http.server", str(port), "--bind", host, "--directory", str(RESULT_DIR)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ReportServer(process=proc, host=host, port=port, root=RESULT_DIR)


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
