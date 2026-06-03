# Benchmarks

This directory contains benchmark notes and raw outputs for the local `llama-server.service`, which runs the `versions/buun` build.

The current primary benchmarking workflow is the buun TUI runner:

```sh
scripts/bench-buun-tui.py
```

Use it to configure and run `llama-bench` comparisons, then rank and apply best results with:

```sh
scripts/rank-buun-results.py
scripts/update-buun-model-presets.py
```
