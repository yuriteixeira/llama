# Plan 006 — Pi Web, Obsidian Memory, and the Nightly Librarian

**Date:** 2026-07-10  
**Status:** Proposed implementation guide

## Goal

Turn the LLM server into a persistent, browser-accessible AI workspace where:

- Pi is the primary agent runtime.
- Pi Web is the browser control plane.
- Existing Pi sessions remain resumable from the terminal and tmux.
- `llama-server` remains the local text/code inference backend.
- image generation runs as an independent service.
- the Obsidian `notes` repository is the human-readable source of truth for memory.
- a nightly Librarian consolidates sessions, updates notes, proposes research, and extracts reusable skills.
- all autonomous changes are reviewable and reversible through Git.

This supersedes the generic chat-UI direction in Plan 005. Open WebUI or LibreChat may still be useful for non-agent chat, but Pi Web should be the main interface for project-aware work.

---

## 1. Target architecture

```text
phone / tablet / laptop / desktop
              │
      Tailscale or SSH tunnel
              │
              ▼
          Pi Web
   browser control plane
      │       │       │
      │       │       └── workspace terminals / files / previews
      │       └────────── persistent Pi sessions
      └────────────────── Pi Web UI plugins
                          │
                          ▼
                     Pi runtime
      ┌───────────────────┼──────────────────────┐
      │                   │                      │
Pi extensions         Pi skills              session store
(agent tools)         (workflows)       ~/.pi/agent/sessions
      │                   │                      │
      └───────────────────┼──────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          │               │                   │
          ▼               ▼                   ▼
 llama-server       image service       Obsidian vault
 localhost:8080       ComfyUI         GitHub notes repo
          │               │                   │
          └───────────────┼───────────────────┘
                          ▼
                  Nightly Librarian
         consolidate → synthesize → propose
                   → review branch/PR
```

### Source-of-truth policy

| Data | Source of truth |
|---|---|
| Conversations | Pi JSONL sessions |
| Permanent knowledge | Obsidian Markdown |
| Project state | project repository |
| Reusable procedures | Pi skills in Git |
| Generated images | workspace artifact folders |
| Semantic indexes | disposable/rebuildable cache |
| Personal fact cache | optional Mem0 sidecar |

The semantic database is never authoritative. If it disappears, it must be rebuildable from Git and Pi sessions.

---

## 2. Implementation order

Do not install everything at once. Complete each phase and validate it before moving forward.

1. Stabilize the base LLM server.
2. Install Pi and connect it to `llama-server`.
3. Install Pi Web and validate browser ↔ terminal handoff.
4. Connect the Obsidian notes repository.
5. Add conservative memory retrieval.
6. Add session export and daily inbox capture.
7. Introduce the Librarian in proposal-only mode.
8. Add skill extraction.
9. Add image generation.
10. Add scheduling, observability, backups, and security hardening.

---

# Phase 0 — Prepare the host

## 0.1 Confirm assumptions

This guide assumes:

- Arch Linux with systemd.
- NVIDIA GPU and working CUDA drivers.
- the repository is cloned at a stable absolute path.
- `llama-server` already works through `manage-server.sh`.
- the machine is reachable privately through Tailscale or SSH.
- GitHub SSH authentication works for private repositories.

Suggested paths:

```bash
export AI_ROOT="$HOME/ai"
export LLAMA_ROOT="$AI_ROOT/llama"
export NOTES_ROOT="$HOME/notes"
export PI_HOME="$HOME/.pi"

mkdir -p "$AI_ROOT"
```

Persist these in a shell file such as `~/.config/ai-stack/env` rather than duplicating paths across services.

## 0.2 Update the host

```bash
sudo pacman -Syu
sudo pacman -S --needed \
  git git-lfs base-devel cmake ninja nodejs npm \
  python python-pipx jq ripgrep fd tmux tailscale

sudo systemctl enable --now tailscaled
```

Do not expose Pi Web, Pi RPC, `llama-server`, or ComfyUI directly to the public internet.

## 0.3 Validate the current inference service

```bash
cd "$LLAMA_ROOT"
./manage-server.sh status
./manage-server.sh test-start
curl -fsS http://127.0.0.1:8080/health
```

Confirm that at least one configured model can answer through the OpenAI-compatible endpoint before installing the agent layer.

---

# Phase 1 — Install Pi as the primary agent runtime

## 1.1 Install Pi

Follow the current official Pi installation method and pin a known-good version initially. Record it:

```bash
pi --version
```

Create a version file:

```bash
mkdir -p "$AI_ROOT/state"
pi --version > "$AI_ROOT/state/pi-version.txt"
```

Avoid unattended Pi upgrades until Pi Web and the critical extensions have been smoke-tested against the new version.

## 1.2 Install and configure `pi-llama-cpp`

Install the current Pi extension that integrates with this repository's llama.cpp model presets:

```bash
pi install npm:pi-llama-cpp
```

Configure it to use:

```text
http://127.0.0.1:8080/v1
```

Use model names that exactly match the selected `model-presets.*.ini` file.

## 1.3 Add a root Pi policy

Create `~/.pi/agent/AGENTS.md` with global rules that apply to all sessions:

```markdown
# Global agent policy

- Prefer reversible changes.
- Never expose services publicly without explicit approval.
- Never commit secrets, tokens, model files, session JSONL, or generated caches.
- Treat the Obsidian notes repository as the source of truth for durable personal knowledge.
- Store temporary artifacts under a project-local `.artifacts/` directory.
- Ask before destructive filesystem, Git, package-management, or service actions.
- When changing infrastructure, update the relevant README or plan.
- Run tests or validation commands before declaring a change complete.
```

Project repositories should keep their own `AGENTS.md` for project-specific instructions.

## 1.4 Validate Pi locally

From the llama repository:

```bash
cd "$LLAMA_ROOT"
pi
```

Validation task:

```text
Read README.md and report the active llama.cpp implementation,
configured API endpoint, service-management commands, and model-preset files.
Do not modify anything.
```

Confirm that Pi can:

- read the repository;
- call the local model;
- use the expected extensions;
- preserve the session under `~/.pi/agent/sessions/`;
- resume it through `pi -r` or `pi --session <id>`.

---

# Phase 2 — Install Pi Web

## 2.1 Install the persistent web control plane

Use the official package:

```bash
npm install -g @jmfederico/pi-web
pi-web install
pi-web doctor
pi-web status
pi-web version
```

The supported service installation creates a persistent session daemon and web server. For a manual fallback:

```bash
pi-web-sessiond
# second terminal
PI_WEB_PORT=8504 pi-web-server
```

## 2.2 Keep it private

Preferred access methods, in order:

1. Tailscale/private VPN address.
2. SSH local forwarding.
3. authenticated reverse proxy restricted to the private network.

SSH example:

```bash
ssh -L 8504:127.0.0.1:8504 user@llm-server
```

Then open:

```text
http://127.0.0.1:8504
```

Do not bind it indiscriminately to `0.0.0.0`.

## 2.3 Register workspaces

Add these as separate Pi Web projects:

```text
$LLAMA_ROOT              local inference infrastructure
$NOTES_ROOT              Obsidian memory vault
other coding repos       one project per repository
$AI_ROOT/operations      shared operational workspace
```

Create the operations workspace:

```bash
mkdir -p "$AI_ROOT/operations"/{bin,logs,state,artifacts}
cd "$AI_ROOT/operations"
git init
```

## 2.4 Validate browser ↔ terminal continuity

Run this exact smoke test:

1. Start a new session from Pi Web in the llama workspace.
2. Ask it to read `README.md` and create no files.
3. Record the session name and ID.
4. Close the browser.
5. In a terminal, run `pi -r` from the same working directory.
6. Select the session and continue it.
7. Send another message in the terminal.
8. Reopen Pi Web and confirm the new turn is visible.
9. Send one more browser message and verify that the same session remains resumable from the CLI.

Document any differences between a live daemon-backed session and a session reopened from JSONL. Do not assume tmux process attachment and Pi session resumption are the same thing.

## 2.5 tmux integration

Keep tmux as the desktop shell workspace, while Pi sessions remain the logical conversation state.

Example helper:

```bash
#!/usr/bin/env bash
set -euo pipefail

name="${1:?workspace name required}"
dir="${2:?workspace directory required}"

tmux new-session -Ad -s "$name" -c "$dir"
tmux attach-session -t "$name"
```

Save as:

```text
$AI_ROOT/operations/bin/workspace-tmux
```

A later Pi Web plugin can expose buttons that launch named tmux workspaces in browser terminals.

---

# Phase 3 — Connect the Obsidian Git repository

## 3.1 Clone the vault on the server

```bash
git clone git@github.com:yuriteixeira/notes.git "$NOTES_ROOT"
cd "$NOTES_ROOT"
git status
```

Use a dedicated deploy key only if the server should not inherit your full GitHub account permissions.

## 3.2 Create an AI-facing vault structure

Adapt these paths to the existing vault rather than reorganizing everything immediately:

```text
AI/
  Inbox/
  Sessions/
  Research/
  Decisions/
  Skills/
  Memory/
    Preferences/
    Projects/
    People/
    Systems/
  Librarian/
    Reports/
    Proposals/
    Rejected/
```

Recommended naming:

```text
AI/Sessions/2026/07/2026-07-10-pi-web-memory-architecture.md
AI/Librarian/Reports/2026-07-10.md
AI/Librarian/Proposals/2026-07-10-skill-project-health-assessment.md
```

## 3.3 Add vault-local agent rules

Create or extend `$NOTES_ROOT/AGENTS.md`:

```markdown
# Notes vault policy

- Markdown is the canonical durable-memory format.
- Preserve existing organization and frontmatter conventions.
- Prefer links and small updates over rewriting entire notes.
- Never silently delete notes.
- Place uncertain material in AI/Librarian/Proposals.
- Label generated claims with sources and dates.
- Do not promote a proposal to permanent memory without review.
- Keep personal facts separate from research conclusions.
- Never commit secrets, access tokens, raw private session dumps, or medical identifiers.
```

## 3.4 Establish Git workflow

The Librarian must never push autonomous changes straight to the main branch.

Initial policy:

```text
main/master       human-approved vault
librarian/YYYY-MM-DD  nightly proposal branch
```

Every run should:

1. fetch and fast-forward the base branch;
2. create a dated branch;
3. make focused commits;
4. generate a report;
5. optionally open a draft PR;
6. stop without merging.

---

# Phase 4 — Memory retrieval, vault first

Start with simple retrieval before adding a memory database.

## 4.1 Level 1: deterministic Markdown search

Provide Pi tools or a skill using:

```bash
rg --glob '*.md' 'query' "$NOTES_ROOT"
fd --extension md . "$NOTES_ROOT"
```

Create a `vault-search` Pi skill that:

1. searches exact phrases with `rg`;
2. searches title/filename variations;
3. reads only the most relevant notes;
4. cites note paths in its answer;
5. distinguishes remembered facts from inferred conclusions.

This is transparent, fast, and easy to debug.

## 4.2 Level 2: semantic index

Add semantic retrieval only after exact search works reliably.

Requirements:

- indexes Markdown without changing it;
- records source path and line/heading metadata;
- can be rebuilt from the vault;
- supports incremental updates;
- runs locally;
- exposes retrieval through MCP, a Pi extension, or a small command-line wrapper;
- never becomes the canonical store.

Candidate path:

```text
Obsidian vault → chunk/index job → local vector store → Pi retrieval tool
```

Keep index data outside Git:

```text
$AI_ROOT/state/vault-index/
```

## 4.3 GBrain evaluation gate

Before adopting GBrain, validate the current project against this checklist:

- active maintenance and clear license;
- direct support for Markdown/Obsidian paths;
- Git-friendly write behavior;
- MCP or CLI interface usable from Pi;
- source citations include note paths/headings;
- no requirement to move canonical data into a proprietary database;
- predictable behavior on renames and deleted notes;
- acceptable handling of private data;
- reproducible backup and restore.

If it passes, use GBrain as the semantic/retrieval layer. If it does not, retain the custom `rg + embeddings` stack.

## 4.4 Optional Mem0 layer

Mem0 solves a narrower problem: extraction and retrieval of compact user/session facts.

Use it only as an optional cache for facts such as:

```text
preferred tools
stable workflow preferences
recurring constraints
current long-lived projects
```

Do not store the only copy of a durable fact in Mem0. Periodically export approved facts to Markdown under:

```text
AI/Memory/Preferences/
AI/Memory/Projects/
```

Do not enable automatic write-back until extraction quality has been reviewed over multiple weeks.

---

# Phase 5 — Capture Pi sessions into the vault

## 5.1 Do not copy every raw message permanently

Pi JSONL remains the complete transcript store. The vault receives curated session summaries.

Each captured summary should contain:

```markdown
---
date: 2026-07-10
source: pi-session
session_id: <id>
workspace: /path/to/project
status: reviewed | generated | rejected
---

# Session title

## Objective

## Decisions

## Changes made

## Important findings

## Open questions

## Candidate memories

## Candidate skills

## Source references
```

## 5.2 Build a session-export command

Create a script in `$AI_ROOT/operations/bin/export-pi-session` that:

1. accepts a Pi session ID/path;
2. reads the JSONL;
3. extracts user messages, assistant conclusions, and tool results;
4. asks a configured summarization model to produce the template above;
5. writes only to `AI/Inbox/` or `AI/Sessions/`;
6. records the original session path;
7. never deletes the JSONL.

The first implementation may call Pi itself with a locked-down summarization prompt.

## 5.3 Manual validation period

For at least one week:

- export sessions manually;
- review every generated summary;
- track omissions and hallucinations;
- refine the template;
- do not run the Librarian unattended yet.

---

# Phase 6 — Nightly Librarian, proposal-only mode

## 6.1 Responsibilities

The Librarian may:

- summarize new Pi sessions;
- consolidate duplicate findings;
- update project status drafts;
- propose backlinks;
- propose research synthesis notes;
- identify contradictions or stale notes;
- suggest missing research;
- propose reusable skills;
- produce a daily report.

It may not:

- merge branches;
- delete permanent notes;
- rewrite large parts of the vault without review;
- invent sources;
- promote uncertain facts to permanent memory;
- execute internet research without recording sources and dates;
- install packages or modify services.

## 6.2 Librarian pipeline

```text
1. ingest
   ├── new Pi sessions
   ├── changed vault notes
   ├── changed project plans
   └── optional Git activity

2. normalize
   ├── deduplicate
   ├── classify
   ├── identify entities/projects
   └── separate facts from hypotheses

3. reflect
   ├── repeated patterns
   ├── contradictions
   ├── stale assumptions
   ├── missing links
   └── research gaps

4. propose
   ├── note changes
   ├── synthesis notes
   ├── candidate memories
   └── candidate skills

5. validate
   ├── links resolve
   ├── sources exist
   ├── Markdown/frontmatter valid
   ├── no secrets
   └── diff-size limits

6. publish
   ├── dated branch
   ├── focused commits
   ├── report
   └── optional draft PR
```

## 6.3 Guardrails

Set hard limits per nightly run:

```text
maximum changed files: 20
maximum new synthesis notes: 3
maximum proposed skills: 2
maximum research tasks: 5
maximum diff size: configurable, initially small
```

Abort and leave a report if limits are exceeded.

## 6.4 systemd service and timer

Create a user service after the script works manually:

```ini
# ~/.config/systemd/user/ai-librarian.service
[Unit]
Description=Nightly AI Librarian
After=network.target

[Service]
Type=oneshot
WorkingDirectory=%h/notes
EnvironmentFile=%h/.config/ai-stack/env
ExecStart=%h/ai/operations/bin/run-librarian
Nice=10
```

```ini
# ~/.config/systemd/user/ai-librarian.timer
[Unit]
Description=Run AI Librarian nightly

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true
RandomizedDelaySec=15m

[Install]
WantedBy=timers.target
```

Enable only after manual validation:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ai-librarian.timer
systemctl --user list-timers ai-librarian.timer
```

The timer should skip heavy inference when another GPU-heavy workload is active.

---

# Phase 7 — Skill extraction

## 7.1 What qualifies as a skill

Create a skill proposal only when a workflow is:

- repeated at least three times, or explicitly requested;
- stable enough to describe deterministically;
- broadly reusable;
- testable;
- safer than free-form improvisation.

Examples:

```text
open-source-project-health-assessment
research-with-primary-sources
llama-model-preset-review
obsidian-session-consolidation
safe-systemd-service-change
```

## 7.2 Skill proposal format

```markdown
---
type: pi-skill-proposal
status: proposed
created: 2026-07-10
evidence_sessions:
  - <session-id>
---

# Skill: Open-source project health assessment

## Trigger

## Inputs

## Procedure

## Required tools

## Safety constraints

## Output format

## Validation cases

## Evidence of repetition
```

## 7.3 Promotion workflow

1. Librarian writes proposal under `AI/Librarian/Proposals/`.
2. Human reviews and edits it.
3. A dedicated build task converts it to a Pi skill.
4. Tests or golden examples are added.
5. Skill is committed to a Git-backed skill repository.
6. Pi reloads the skill.
7. Librarian records whether future uses succeed.

Never let the Librarian silently overwrite an active skill.

---

# Phase 8 — Image generation

## 8.1 Keep it independent

Use ComfyUI as a separate service, as already proposed in Plan 005.

```text
Pi chat → image-generation Pi extension/tool → ComfyUI API
                                             ↓
                                      generated file
                                             ↓
                                  Pi Web file preview
```

## 8.2 Resource policy

With the current 16 GB primary GPU:

- do not assume large LLM inference and diffusion can coexist reliably;
- initially stop or unload the LLM before heavy image generation;
- if retaining the RTX 3070, pin ComfyUI to that GPU and `llama-server` to the RTX 5060 Ti;
- store image models outside Git;
- store generated images under project-local `.artifacts/images/` or a dedicated media directory;
- write prompts and generation metadata beside important outputs.

## 8.3 Pi integration

Install or build a Pi extension exposing a constrained tool such as:

```text
generate_image(prompt, workflow, width, height, output_directory)
```

The extension should:

- submit a known ComfyUI workflow;
- reject arbitrary output paths;
- wait for completion with a timeout;
- return the generated file path;
- save generation metadata;
- allow Pi Web to preview the result.

---

# Phase 9 — Pi Web plugins

Use Pi extensions for agent capabilities and Pi Web plugins for human/browser affordances.

## 9.1 First custom plugin: AI stack dashboard

Create a local Pi Web plugin that displays:

```text
llama-server status
active model
GPU memory and utilization
Pi Web version
Pi version
Librarian last run
pending Librarian PR/branch
notes repository dirty state
ComfyUI status
```

Actions:

```text
open llama logs
open Librarian report
start/stop image service
open notes workspace
open named tmux workspace
run stack smoke test
```

The plugin should call narrowly scoped scripts rather than expose an unrestricted command textbox.

## 9.2 Second plugin: memory review inbox

Panel contents:

```text
new session summaries
candidate permanent memories
proposed skills
research gaps
conflicting notes
```

Actions:

```text
open diff
mark accepted
mark rejected
open draft PR
```

Keep review state in Markdown or a small Git-backed JSON file when possible.

---

# Phase 10 — Operations and security

## 10.1 Service boundaries

Run these as separate services:

```text
llama-server.service
pi-web session daemon
pi-web server
comfyui.service                 optional/on-demand
ai-librarian.service            scheduled oneshot
vault-index.service             optional incremental indexer
```

One failure should not cascade into the others.

## 10.2 Secrets

Store secrets in protected environment files:

```text
~/.config/ai-stack/env
~/.config/pi-web/
```

Permissions:

```bash
chmod 700 ~/.config/ai-stack
chmod 600 ~/.config/ai-stack/env
```

Never give the Librarian broad access to SSH private keys, browser cookies, password stores, or unrelated home-directory data.

## 10.3 Backups

Back up:

```text
notes Git repository
Pi session JSONL
Pi skills/extensions configuration
service units and environment templates
Pi Web configuration
ComfyUI workflows
```

Do not prioritize backing up rebuildable caches such as vector indexes.

Suggested local backup:

```bash
rsync -a --delete \
  "$HOME/.pi/agent/sessions/" \
  /backup/pi-sessions/
```

Git is not a complete backup. Keep at least one offline or independently hosted copy of the vault and session store.

## 10.4 Observability

Create one smoke-test script that verifies:

```text
NVIDIA driver available
llama-server HTTP health
configured model list
Pi executable/version
Pi Web services/version
notes Git status
Librarian timer and last run
ComfyUI health when enabled
disk free space
```

Return non-zero on failures so it can later feed notifications.

---

# Phase 11 — Acceptance tests

## Core workflow

- [ ] Start a Pi session from Pi Web.
- [ ] Resume it from terminal/tmux.
- [ ] Continue it from the browser again.
- [ ] Tool and skill availability is consistent in both interfaces.
- [ ] Files created in the workspace appear in Pi Web previews.

## Local inference

- [ ] Pi uses `llama-server` through the expected base URL.
- [ ] Model selection matches the preset router.
- [ ] Server restart does not destroy Pi sessions.

## Memory

- [ ] Exact vault search returns cited note paths.
- [ ] Semantic search can be fully rebuilt.
- [ ] Session export creates a reviewable summary.
- [ ] No raw transcript is promoted automatically.
- [ ] Permanent notes remain human-readable Markdown.

## Librarian

- [ ] Manual dry run produces no direct main-branch changes.
- [ ] Every claim has a traceable source.
- [ ] Diff limits and secret scanning work.
- [ ] A report is produced even on failure.
- [ ] The timer does not overlap with heavy GPU workloads.

## Skill evolution

- [ ] Repeated workflow produces a proposal, not an active skill.
- [ ] Proposal includes evidence sessions and tests.
- [ ] Human-approved skill loads in Pi.
- [ ] Existing skills are never silently replaced.

## Image generation

- [ ] Pi can submit a constrained ComfyUI workflow.
- [ ] Output is saved in an allowed directory.
- [ ] Pi Web previews the output.
- [ ] GPU contention policy prevents accidental OOM loops.

---

# Recommended first milestone

Do only these tasks first:

1. Install Pi Web.
2. Register the llama and notes repositories.
3. Validate browser ↔ terminal session continuity.
4. Create vault-local `AGENTS.md` rules.
5. Build deterministic `vault-search` using `rg`.
6. Manually export three Pi sessions into `AI/Inbox/`.
7. Review the results before adding semantic search or automation.

This provides most of the daily value while keeping the system understandable.

---

# Recommended second milestone

After one week of regular use:

1. Add a rebuildable semantic index.
2. Implement the Librarian as a manual command.
3. Restrict it to a dated proposal branch.
4. Generate a report, candidate memories, and at most one skill proposal.
5. Review every diff.
6. Run manually for another week.

Only then schedule it nightly.

---

# Definition of the endgame

The system is complete when:

- Pi is the common runtime for coding, research, and personal-assistant sessions.
- Pi Web is a convenient remote control plane, not a separate chat silo.
- tmux remains useful without owning conversation state.
- the Obsidian vault remains the inspectable, editable, Git-backed memory source.
- retrieval databases are disposable accelerators.
- the Librarian continuously improves organization and proposes new knowledge.
- reusable behavior evolves through reviewed Pi skills.
- autonomous work remains source-linked, diffable, reversible, and under human control.
