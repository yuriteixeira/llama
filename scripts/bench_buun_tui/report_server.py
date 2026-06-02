from __future__ import annotations

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


def start_report_server(host: str = "0.0.0.0", port: int = 1234) -> ReportServer:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "http.server", str(port), "--bind", host, "--directory", str(RESULT_DIR)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ReportServer(process=proc, host=host, port=port, root=RESULT_DIR)
