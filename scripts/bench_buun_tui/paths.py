from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "versions/buun/src/buun-llama-cpp/build/bin/llama-bench"
SERVER = ROOT / "versions/buun/src/buun-llama-cpp/build/bin/llama-server"
CONFIG_DIR = ROOT / "bench-configs/buun"
LAST_CONFIG = CONFIG_DIR / ".last-config"
RESULT_DIR = ROOT / "bench-results/buun-tui"
GGUF_PY = ROOT / "versions/buun/src/buun-llama-cpp/gguf-py"
PACKAGE_DIR = Path(__file__).resolve().parent
HTML_REPORT_TEMPLATE = PACKAGE_DIR / "index.html"
D3_SCRIPT = PACKAGE_DIR / "d3.v7.min.js"
