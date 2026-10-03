# Llama.cpp - My experiments with models & presets

## Testing Workflow

1. Choose a model
2. Set-up a preset for that model and let `./scripts/download-preset-models.sh` to download and save them in the `./models/` folder.
3. Run `./manage-server.sh restart && ./manage-server.sh logs`
4. Use Pi's `/llama` to load a model (if it fails to load, logs will tell why, probably OOM or unsupported args)
5. Test it out!

## About versions

We have for now `upstream` (vanilla's llama-cpp) and some forks like `turboquant` & `buun`. Another one can always be added for test purposes, but for now, we're focusing on regular llama-cpp.

## Qwen3.8 preset tests

The new presets are in `model-presets.upstream.ini`. Existing presets remain unchanged.
All new presets use medium reasoning and one slot. MTP drafts several tokens before the main model checks them; this can improve speed.

| Preset | Context tokens | GPUs | Cache | Notes |
| --- | ---: | --- | --- | --- |
| `baseline-qwen3.8-27b-iq4-32k-q8-mtp2` | 32768 | 5060 Ti | Q8 | Baseline for speed and quality tests |
| `compare-qwen3.8-27b-iq4-32k-q8-mtp3` | 32768 | 5060 Ti | Q8 | Compare MTP depth 3 with the baseline |
| `memory-qwen3.8-27b-iq4-64k-q4-mtp3` | 65536 | 5060 Ti | Q4 | Lower cache memory use; check layer placement and quality |
| `dual-gpu-qwen3.8-27b-iq4-64k-q8-mtp3` | 65536 | Both | Q8 | Uses both GPUs; may improve speed |
| `capacity-qwen3.8-27b-iq3-64k-q4-mtp3` | 65536 | 5060 Ti | Q4 | Smaller weights; check answer quality |
| `long-context-qwen3.8-27b-iq3-128k-q4-mtp3` | 131072 | 5060 Ti | Q4 | For long prompts; check answer quality |

The IQ4 presets use your existing model. The IQ3 presets use smaller weights, which can affect answer quality.
Download the missing IQ3 file from the repository root before loading either IQ3 preset:

```bash
curl --fail --location --retry 3 --continue-at - \
  --output models/unsloth/Qwen3.8-27B-UD-IQ3_XXS.gguf \
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_XXS.gguf
```

The existing `scripts/download-preset-models.sh` script registers local files; it does not download models.
Restart with `./manage-server.sh restart`, then choose a preset in Pi with `/llama`.
For 64K, compare Q4 cache on one GPU against Q8 cache on both GPUs. Try IQ3 at 128K only after its 64K preset works.

Context includes input, reasoning, and output. A successful load does not prove that a long prompt works.
Automatic fitting can put layers on the CPU when GPU memory is insufficient. Check layer placement and actual context size in the logs.
The preset for both GPUs requests all layers on GPUs and can fail if memory is insufficient.
Use equal prompts and output limits when comparing speed and task success. Client request values can override preset settings.

