# Plan 005 — Chat UI, Image Generation, and Autonomous Agent Workflows

**Date:** 2026-06-28
**Status:** Draft

## Goal

Extend this local inference stack beyond a headless OpenAI-compatible `llama-server` endpoint by adding:

1. A browser chat UI for interactive use, with web search enabled.
2. A local image-generation service exposed through a compatible API.
3. Integration notes and validation for more autonomous workflows using Hermes Agent and Paperclip.

## Current repository finding

No existing plan or `AGENTS.md` section currently covers a chat UI, image generation, Hermes Agent, or Paperclip. Existing plans focus on model inventory, preset optimization, multi-GPU/upstream migration, and larger context windows.

## Target architecture

```txt
users / tools
  ├─ chat UI
  ├─ Hermes Agent
  ├─ Paperclip
  └─ pi / other OpenAI-compatible clients
        │
        ├─ LLM:   http://llama-host:8080/v1  -> llama-server router -> GGUF models
        └─ Image: http://image-host:PORT      -> ComfyUI or Stable Diffusion WebUI -> diffusion models
```

Keep `llama-server` focused on text/code LLM inference. Run image generation as a separate service so VRAM contention can be controlled explicitly.

## Phase 1 — Chat UI

Evaluate lightweight UIs that can point at an OpenAI-compatible base URL **and support web search/RAG from search results**:

- Open WebUI, with web search enabled
- LibreChat, with a search plugin/provider enabled
- Anything already bundled by the selected `llama.cpp` fork, only if it supports web search or can be paired with a search sidecar

Minimum requirements:

- Configurable base URL: `http://<host>:8080/v1`
- Configurable model names matching `model-presets.*.ini`
- Web search support is mandatory, not optional
- Prefer self-hostable search via SearXNG; hosted APIs such as Brave Search, Tavily, Serper, or Exa are acceptable only if explicitly configured by the operator
- Search results must be cited or visibly attached to responses when the UI supports citations
- Ability to disable web search per-chat for private/offline prompts
- No hard dependency on hosted LLM providers
- Easy systemd or container deployment

Deliverables:

- `chat-ui.service` or documented container command
- `search.service` or documented SearXNG/container command if using self-hosted search
- README section with URL, default model selection, web-search setup, and troubleshooting

## Phase 2 — Image generation

Run image generation separately from `llama-server`:

- Preferred first candidate: ComfyUI for flexible workflows
- Alternative: AUTOMATIC1111 / Stable Diffusion WebUI if a simpler UI/API is preferred

Minimum requirements:

- Can be started/stopped independently from `llama-server`
- Uses explicit CUDA device selection to avoid unexpected contention with LLM workloads
- Stores models outside git, e.g. `./image-models/` or another documented path
- Has a documented API endpoint usable by agents/tools

Deliverables:

- Image service install/run script
- Optional `image-generation.service`
- Model download notes and deny-list guidance similar to text models

## Phase 3 — Hermes Agent integration

Hermes Agent appears to support local/custom OpenAI-compatible endpoints via `model.base_url`-style configuration, but this must be validated against the current release.

Validation checklist:

- Configure Hermes Agent base URL as `http://<host>:8080/v1`
- Confirm model picker/model name behavior with `llama-server --models-preset`
- Test tool-use/autonomous loops with a coding-capable preset
- Record required temperature/top-p/reasoning settings for reliable agent behavior
- Decide whether a dedicated "agent" preset should use smaller context/lower temperature than chat presets

Deliverable:

- `docs/hermes-agent.md` with exact config and known-good preset recommendations

## Phase 4 — Paperclip integration

Paperclip local LLM / OpenAI-compatible support should be verified before implementation. Treat it as a client of the existing LLM endpoint unless its runtime requires a separate adapter.

Validation checklist:

- Determine whether Paperclip supports OpenAI-compatible `base_url` directly
- If direct support exists, point it at `http://<host>:8080/v1`
- If not, evaluate an adapter layer such as LiteLLM or LocalAI
- Test session management, long-running jobs, and sandboxing expectations
- Document any missing features or upstream blockers

Deliverable:

- `docs/paperclip.md` with exact config or adapter notes

## Phase 5 — Resource scheduling

Because this host is GPU-constrained, add an operating policy:

- Default: only one heavy service active at a time on the 16 GB GPU
- Chat UI may stay always-on if it only calls `llama-server`
- Image generation should be manually started or pinned to a secondary GPU when available
- Autonomous agents should use conservative presets first to avoid OOM during long runs

Future enhancement:

- Add `manage-workloads.sh` to stop/start LLM and image services safely.

## Success criteria

- A browser chat UI can chat with every active `llama-server` preset.
- The selected chat UI can answer with web search enabled and cite/show search sources.
- Image generation can run locally without breaking the LLM service after restart.
- Hermes Agent can complete a small autonomous coding/research task through the local endpoint.
- Paperclip support is either working through the local endpoint or documented with clear blockers.
- README documents how to start/stop each user-facing workflow.
