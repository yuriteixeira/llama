# PLAN.md — Best-Known Model Params in Buun Benchmark TUI

## Goal

Add a TUI workflow that can:

1. Scan all GGUF models under `./models`.
2. Research each model by name/metadata using `pi -p`.
3. Save researched, best-known benchmark/server parameter recommendations to an INI file matching that model.
4. Let users load those recommendations while editing a profile via an option like **“Load best-known params for this model”**.

## Current TUI touchpoints

- Entry point: `scripts/bench-buun-tui.py`
- TUI screens/actions: `scripts/bench_buun_tui/tui.py`
- Model discovery and GGUF metadata: `scripts/bench_buun_tui/models.py`
- Profile schema/defaults/descriptions: `scripts/bench_buun_tui/profile.py`
- Config read/write: `scripts/bench_buun_tui/config_store.py`
- Benchmark command rendering: `scripts/bench_buun_tui/commands.py`
- Equivalent `llama-server` command rendering should likely live in `commands.py` too, next to `profile_command()`.

## Proposed files

- `bench-configs/buun/best-known-params.ini`
  - Generated/updated by the scan/research workflow.
  - Stores one section per discovered model or model fingerprint.
- `scripts/bench_buun_tui/best_known.py`
  - New module for loading/saving best-known params, matching current model to a recommendation, and applying params to a `Profile`.
- Optional: `scripts/research-best-known-params.py`
  - Non-curses helper that performs the scan/research/generate step from CLI.
  - The TUI can call the same logic or shell out to it.

## Proposed INI format

Example:

```ini
[meta]
created_at = 2026-06-02T00:00:00
source = <replace by full pi command that was run to research about the model>
schema_version = 1

[model:models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf]
model_name = Qwen3.6-35B-A3B
architecture = qwen3moe
quantization = IQ4_NL
size_bytes = 123456789
research_summary = MoE model, about 3B active params; best known for RTX 3070 uses turbo KV and CPU MoE offload.
confidence = medium
cache_strategy = turbo
depths = 0,4096,8192,16384,32768
batch_size = 1024
ubatch_size = 256
threads = 12
n_cpu_moe = 33
gpu_layers = unset
prompt = 512
gen = 128
repetitions = 1
warmup = false
device = CUDA0
flash_attn = 1
no_mmap = unset
references = https://...
```

Notes:

- Sections should use the relative model path for exact matching.
- Store metadata like architecture, GGUF name, quantization, if it is MTP, REAP, supports MoE, active params, and file size to help detect stale recommendations.
- Only keys that already exist on `Profile` should be applied automatically.
- Unknown or research-only fields should be displayed but not applied.

## Research workflow

1. Discover models via the existing `find_models()` logic.
2. For each model, read local GGUF metadata with `model_metadata()`:
   - `general.name`
   - `general.architecture`
   - `*.block_count`
   - filename-derived quantization/model family
3. Build a research prompt for `pi -p` including:
   - model path and filename
   - GGUF metadata
   - hardware target from `AGENTS.md` / README: RTX 3070 8GB VRAM, 12 CPU threads, 16+GB RAM
   - available TUI fields and valid values
   - requirement to return strict INI-compatible key/value recommendations
4. Execute `pi -p` once per model or as one batch prompt for all models.
5. Parse and validate output:
   - accept only known `Profile` fields for auto-apply
   - normalize cache recommendations to existing `CACHE_STRATEGIES`
   - ensure numeric/list fields remain compatible with `llama-bench`
   - preserve a human-readable summary/references
6. Save to `bench-configs/buun/best-known-params.ini`.

## TUI additions

### Main menu

Add an action near existing config actions:

- **Scan models and research best-known params**

Behavior:

1. Run the scan/research workflow immediately; no confirmation screen is needed.
2. Show progress/status while `pi -p` is running, because it may take time.
3. Show success/failure and the path of `best-known-params.ini`.
4. Refresh any in-memory best-known recommendation cache.

### Profile editor

Add action/fields at the top or bottom of profile editing:

- **Load best-known params for this model**
- **Show equivalent llama-server command**

Load best-known params behavior:

1. Match `profile.model` against `best-known-params.ini`.
2. If no exact match, offer fuzzy matches by basename/model name.
3. Show a preview of changes:
   - current value
   - recommended value
   - confidence/research summary
4. User confirms before applying.
5. Apply only supported `Profile` fields, not `model` unless explicitly desired.

Equivalent `llama-server` command behavior:

1. Render a command using the current profile values and the buun server binary/path conventions.
2. Map profile fields to server flags where possible, for example:
   - `model` → `--model`
   - `threads` → `--threads` and probably `--threads-batch`
   - `batch_size` → `--batch-size`
   - `ubatch_size` → `--ubatch-size`
   - `n_cpu_moe` → `--n-cpu-moe`
   - `gpu_layers` → `--n-gpu-layers` if set
   - `cache_strategy` → `--cache-type-k` / `--cache-type-v`
   - `device` → `--device`
   - `flash_attn` → `--flash-attn on/off` or server-compatible equivalent
   - `no_mmap` → `--no-mmap` when true
3. Include server defaults not present in the benchmark profile when useful, such as host/port and context size.
4. Show the command in a scrollable/copyable text screen, and optionally offer to save it to a file or copy it if the TUI has a safe mechanism for that.
5. Clearly label fields that do not map cleanly from benchmark to server, such as `depths`, `prompt`, `gen`, `repetitions`, and `warmup`.

## Matching strategy

Recommended order:

1. Exact relative path match.
2. Exact basename match.
3. GGUF `general.name` match.
4. Filename stem similarity match.

If multiple matches are found, ask the user to choose.

## Validation rules

- Reject unknown profile fields for auto-apply.
- Keep `model` unchanged when loading recommendations from inside profile editor.
- Validate `cache_strategy` against `CACHE_STRATEGIES`.
- Validate boolean-like fields (`warmup`, `flash_attn`, `no_mmap`) against current accepted values.
- Validate comma-separated numeric fields enough to avoid malformed commands.
- If a value is questionable, keep it in the INI but do not apply without manual confirmation.

## Implementation phases

### Phase 1 — Data model and loader

- Add `best_known.py`.
- Implement:
  - path constant for `best-known-params.ini`
  - INI read/write helpers
  - recommendation lookup for a model
  - validation and safe application to `Profile`
- Add small unit-testable functions where possible.

### Phase 2 — Research generator

- Add scan/research function and optional CLI helper.
- Use existing `find_models()` and `model_metadata()`.
- Construct deterministic `pi -p` prompts.
- Parse strict INI output and save recommendations.
- Add dry-run or prompt-print mode if useful for debugging.

### Phase 3 — TUI integration

- Add main menu action to trigger scan/research.
- Add profile editor action to load best-known params.
- Add profile editor action to show the equivalent `llama-server` command.
- Add preview/confirmation screen before applying recommendations.
- Add an inspect/copy-friendly command display screen for generated server commands.
- Add error messages for missing INI, no model selected, no match, or invalid recommendation.

### Phase 4 — Docs

- Update README with:
  - how to run the TUI
  - how to generate best-known params
  - where the generated INI lives
  - caveat that recommendations are researched heuristics and should be benchmarked

### Phase 5 — Manual verification

- Run TUI smoke test.
- Generate recommendations for current `./models` contents.
- Load recommendation into `profile_a` and verify saved config contains applied params.
- Open the equivalent `llama-server` command view and verify the flags match the selected profile.
- Run a small benchmark config if approved.

## Open questions before implementation

1. Should the research command be available only in the TUI, or also as a standalone CLI script?
2. Should `pi -p` run once per model for better prompts, or once for all models for speed/cost control?
3. Should best-known params target benchmarking only, server serving only, or both? Current `Profile` fields are benchmark-oriented, but the TUI should also render an equivalent `llama-server` command from them.
4. Should recommendations be hardware-specific to this RTX 3070 setup, or should the INI support multiple hardware profiles?
5. Should applying best-known params overwrite all fields or only fields that differ from defaults?
