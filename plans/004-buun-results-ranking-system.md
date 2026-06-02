# Plan: Buun Benchmark Results Ranking System

## Goal

Build a ranking system that scans all finalized benchmark runs under `./bench-results/buun-tui/` and reports the winning profile configuration based on paired prompt-processing (`pp`) and token-generation (`tg`) tokens per second.

The output should make it clear which configuration wins overall and which configuration wins per context depth.

## Data Source

Read every completed run directory matching:

```txt
bench-results/buun-tui/*/summary.csv
```

Ignore:

```txt
bench-results/buun-tui/bench-buun-tui.log
bench-results/buun-tui/*/summary.live.csv
```

Use `summary.csv` because it represents finalized benchmark rows.

## Relevant Fields

Each summary row already contains the fields needed for ranking:

```csv
profile, model, model_path, cache_strategy, test, n_depth,
n_batch, n_ubatch, n_threads, n_cpu_moe, n_gpu_layers,
flash_attn, type_k, type_v, avg_ts, stddev_ts
```

Where:

- `test=pp512` means prompt-processing throughput.
- `test=tg128` means token-generation throughput.
- `avg_ts` is tokens per second.

## Profile Config Identity

Do not rank only by `profile_a` or `profile_b`, because those labels are local to each benchmark run.

Use a stable configuration key built from:

```txt
model_path
cache_strategy
n_batch
n_ubatch
n_threads
n_cpu_moe
n_gpu_layers
flash_attn
type_k
type_v
```

Optionally include these derived fields when present:

```txt
model_layers
cpu_moe_layers
remaining_moe_layers
```

This makes equivalent profile configurations comparable across multiple runs.

## Pair PP and TG Results

Group rows by:

```txt
run_dir
profile
profile_config_key
n_depth
```

Then pair rows with:

```txt
pp512 avg_ts -> pp_tps
tg128 avg_ts -> tg_tps
```

A ranking candidate is valid only when both PP and TG results exist for the same run, profile config, and `n_depth`.

Example internal candidate:

```json
{
  "run": "20260602-165120-bench",
  "profile": "profile_b",
  "model_path": "models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
  "n_depth": 8192,
  "pp_tps": 638.38,
  "tg_tps": 54.17,
  "config": {
    "cache_strategy": "turbo",
    "n_batch": 1024,
    "n_ubatch": 256,
    "n_threads": 12,
    "n_cpu_moe": 19,
    "n_gpu_layers": 41,
    "flash_attn": true,
    "type_k": "turbo4",
    "type_v": "turbo3"
  }
}
```

## Ranking Score

Use a balanced score so prompt-processing throughput does not dominate token-generation throughput numerically.

Recommended default:

```txt
normalized_pp = pp_tps / best_pp_tps_for_same_depth
normalized_tg = tg_tps / best_tg_tps_for_same_depth
score = sqrt(normalized_pp * normalized_tg)
```

This creates a fair combined score per `n_depth`.

Alternative simple score:

```txt
score = sqrt(pp_tps * tg_tps)
```

However, normalized scoring is preferred because PP TPS is usually much larger than TG TPS.

## Ranking Views

### 1. Overall Winner

Find the best aggregate configuration across all depths and runs.

Aggregate per stable profile config using median or mean:

```txt
overall_score = median(score across paired samples)
```

Median is recommended for stability.

The overall winner should show:

```txt
rank
model
model_path
cache_strategy
n_batch
n_ubatch
n_threads
n_cpu_moe
n_gpu_layers
flash_attn
type_k
type_v
average or median PP TPS
average or median TG TPS
overall score
paired sample count
```

### 2. Winners by Context Depth

For each `n_depth`, show the top-ranked paired candidates.

Expected common depths:

```txt
0
4096
8192
16384
32768
```

Columns:

```txt
n_depth
rank
model
config summary
pp_tps
tg_tps
score
run
profile
```

### 3. Best PP-only and TG-only

Also show single-metric winners because one profile may dominate prompt processing while another dominates generation.

Generate:

```txt
best_pp_by_depth
best_tg_by_depth
```

Each entry should include:

```txt
n_depth
model
config summary
pp_tps or tg_tps
run
profile
```

## Output Files

Generate reports under:

```txt
bench-results/buun-tui/ranking.txt
bench-results/buun-tui/ranking.md
```

Optional future output:

```txt
bench-results/buun-tui/ranking.html
```

## Suggested Implementation

Add a new script:

```txt
scripts/rank-buun-results.py
```

Suggested CLI:

```bash
./scripts/rank-buun-results.py \
  --input ./bench-results/buun-tui \
  --output ./bench-results/buun-tui/ranking.md
```

Optional arguments:

```txt
--min-samples 2
--score normalized-geomean
--include-test-runs
--latest-only
```

## Report Shape

Example Markdown report:

```md
# Buun Benchmark Ranking

## Overall Winner

Model: Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Config:
- cache_strategy: turbo
- n_cpu_moe: 19
- n_gpu_layers: 41
- type_k/type_v: turbo4/turbo3
- batch/ubatch: 1024/256

Median PP TPS: 638.2
Median TG TPS: 54.7
Score: 0.94
Samples: 5

## How to run the same params on llama-server?

Run this:

```sh
<add the equivalent llama-server command with the equivalent params and values here>
```

## Winners by Depth

| Depth | Rank | Model | PP TPS | TG TPS | Score | Run |
|---:|---:|---|---:|---:|---:|---|
| 0 | 1 | ... | ... | ... | ... | ... |
| 4096 | 1 | ... | ... | ... | ... | ... |
```

## Recommended Default

Use normalized geometric mean by depth:

```txt
score = sqrt(normalized_pp * normalized_tg)
```

This rewards configurations that are strong at both PP and TG instead of allowing very high PP TPS to hide weak TG TPS.

