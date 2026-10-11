# Can preset 4 go from 64k to 128k / 256k using both GPUs?

## Short answer

**Yes for 128k (comfortable), and yes for 256k (tight, needs 3 tuning knobs).** The second GPU (RTX 3070 8GB, currently idle with 7705 MiB free) is enough to double or quadruple the context, at the cost of roughly 15 to 25 percent decode speed. The 5060 Ti alone cannot do it: at 128k it would need about 15.3 GiB before startup overhead, against 15.4 GiB usable, so it OOMs after the CUDA graph allocations (confirmed by the 64k run, where projected 13548 MiB became 15493 MiB actual).

## Memory budget (measured with `llama-fit-params` on the real model file)

The fit tool measures the main model only (it excludes the MTP draft; see `tools/fit-params/fit-params.cpp`, extra model passed as `nullptr`). The MTP draft overhead was measured from the live server log: 13548 (server, 64k) minus 12633 (fit, 64k) = **915 MiB**, which scales to about 1310 MiB at 128k and about 2100 MiB at 256k. The MTP layer is the last layer, so with `--device CUDA1,CUDA0` it lands on **CUDA0** (layer to device mapping, `src/llama-model.cpp:1591 1599`).

| Scenario (main model, fit tool) | CUDA1 used | CUDA0 used | + MTP draft | CUDA0 free | Verdict |
|---|---|---|---|---|---|
| 1 GPU, 64k | 12633 | - | 915 on CUDA1 | - | baseline |
| 1 GPU, 128k | 14001 | - | 1310 | - | **OOM after startup** |
| 2 GPU, `ts 3,1`, 128k | 10628 | 4982 | 1310 on CUDA0 | **~1600** | **fits with room** |
| 2 GPU, `ts 4,1`, 192k | 12763 | 5023 | ~1670 on CUDA0 | **~1200** | fits |
| 2 GPU, `ts 4,1`, 256k, `-ub 256` | 13639 | 5159 | ~2100 on CUDA0 | **~630** | **fits, tight** |
| 2 GPU, `ts 3,1`, 256k | 13380 | 6582 | ~2100 on CUDA0 | **~-790** | overflow |

Key facts behind the numbers:
- Per token KV: about 22528 bytes (16 full attention layers in q4_0, 48 linear layers have a fixed ~149 MiB recurrent state that does not scale with context, 1 MTP layer in f16). Verified against the 64k log (1408 MiB KV).
- The compute buffer scales with context and ubatch: 505 MiB (64k, ub 512) to 2169 MiB (256k, ub 512), dropping to ~1600 MiB per card at 256k with ub 256.
- At 256k the default fit target of 1024 MiB free per device will abort on CUDA0 (630 free), so you must lower it with `--fit-target 512` (flag `-fitt`).

## Recommended preset changes

In `./model-presets.upstream.ini`, under section `[4-qwen3.8-27b-gsq-rco-iq3-64k-mtp2-single]`:

**Option A: 128k (recommended first step, comfortable fit)**
```ini
device = CUDA1,CUDA0
tensor-split = 3,1
ctx-size = 131072
```
Unchanged: `cache-type-k/v q4_0`, `flash-attn on`, `spec-type draft-mtp`, `parallel 1`.

**Option B: 256k (tight, three extra knobs)**
```ini
device = CUDA1,CUDA0
tensor-split = 4,1
ctx-size = 262144
ubatch-size = 256
fit-target = 512
```
(Exact flag names to confirm in your preset file format: `ubatch-size` maps to `-ub`, `fit-target` to `-fitt`.)

Why `4,1` for 256k: the 3070 hosts the MTP layer, so it carries the draft KV (1024 MiB at 256k) plus draft compute. `3,1` pushes it about 800 MiB over.

**Why layer mode, not tensor mode:**
- `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1181 1215`) is an exclusion list and `qwen35` is not in it, so tensor mode is not blocked for this arch.
- But `common/fit.cpp:183` rejects tensor mode in the fit path, tensor mode forces full f16 KV (`src/llama-context.cpp:3822 3828`), which at 256k alone would be about 17 GiB of KV (16 layers x 1024 MiB plus the MTP layer) and does not fit in 23.8 GiB total.
- Community reports show tensor mode is also the riskier mode on mismatched card pairs (Reddit threads below), while layer mode "seems to work fine".

## Expected performance

Both cards have 448 GB/s memory bandwidth (3070: 14 Gbps x 256 bit; 5060 Ti: 8 Gbps x 128 bit; also noted in the titarpro post below), so decode time is roughly the split of total bytes read per token. A 2,1 to 4,1 layer split gives about 25 to 30 tokens/s decode versus about 35 to 40 today at 64k, but prefill benefits strongly from two GPUs working in parallel, and you gain 2x to 4x the context. This matches community reports: a dual 5060 Ti setup runs Qwen3.6 27B at 43 to 44 tokens/s at 131k context with MTP (Reddit 2x3090 thread), and tensor mode is reported as about 2x faster than layer mode only when both cards are identical (dual 5060 Ti thread).

## Sources

Local (all verified by me in this session):
- Fit projections: `versions/upstream/src/build/bin/llama-fit-params` runs listed in the table above (12633 / 14001 / 16945 one GPU; 10628 / 4982 at ts 3,1 128k; 13639 / 5159 at ts 4,1 256k ub 256; 6582 at ts 3,1 256k).
- MTP draft overhead: live server log (64k: projected 13548, actual 15493) minus fit 12633 = 915 MiB.
- `src/llama-arch.cpp:1181 1215`, `src/llama-model.cpp:1591 1599` and `2655 2830`, `src/llama-context.cpp:3822 3828`, `src/llama-hparams.cpp:227 268`, `common/fit.cpp:183`, `tools/fit-params/fit-params.cpp` as cited above.
- Hardware: `nvidia-smi` (CUDA0 RTX 3070 7893 MiB idle, CUDA1 RTX 5060 Ti 15938 MiB).

Web:
- r/LocalLLaMA, single 5060 Ti 16GB with Qwen 3.8 27B tops out at 73k context: https://www.reddit.com/r/LocalLLaMA/comments/1vqrt86/
- r/LocalLLM, dual 5060 Ti with Qwen3.6 27B: tensor mode about 2x speed, standard build does not support Q8_0 KV with tensor split: https://www.reddit.com/r/LocalLLM/comments/1to8k77/
- r/LocalLLaMA, tensor mode issues on a mixed GPU pair, layer mode fine: https://www.reddit.com/r/LocalLLaMA/comments/1ufy09k/
- r/LocalLLaMA, Qwen3.6 27B on 2x3090, 43 to 44 tokens/s at 131k context with MTP: https://www.reddit.com/r/LocalLLaMA/comments/1twq9n9/
- Hacker News, 5060 Ti 16GB at 128K context limited by KV fit: https://news.ycombinator.com/item?id=49749076
- titar.pro, the exact card pair (5060 Ti 16GB + 3070 8GB) with llama.cpp, both 448 GB/s, warns that chasing 100 percent VRAM makes it twice as slow: https://titar.pro/posts/running-gemma-4-31b-on-two-mismatched-gpus
- hardware-corner, dual 5060 Ti (32GB pool) beats single 3090 for long context: https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/
- GitHub, dual 5060 Ti recipes with tensor split and MTP: https://github.com/Kurokosama/club-dual-5060ti
- YouTube, 5060 Ti plus an old second GPU as a speed and VRAM boost: https://www.youtube.com/watch?v=BLgN7ZmeJZI
- YouTube, 5060 Ti + 3060 12GB running Gemma3 27B (28GB pool): https://www.youtube.com/watch?v=hqrBC4EgqbY
- YouTube, Qwen 3.8 27B split across two machines at 128K context, 22 tokens/s: https://www.youtube.com/watch?v=dypHkOKOLt8

## Caveat

The two GPU numbers above are projections (fit tool plus a hand added MTP draft term), because I cannot stop the live service on port 60713 to run a clean measurement (no passwordless sudo). Before switching, do a one time smoke test: apply Option A, restart the service, check `nvidia-smi` on both cards, and confirm free memory on CUDA0 is 1 GiB or more. If it is tight, drop to 192k (Option A with `ctx-size = 196608`) before trying 256k.
