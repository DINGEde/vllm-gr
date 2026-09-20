# AB_Tests — A/B performance testing for vllm-gr PRs

Runs an **A-B-B-A** offline beam-search benchmark (baseline → head → head → baseline) against a vllm-gr PR and produces a base-vs-head latency comparison report. **Mode B: each developer runs their own benchmark server** — the scripts are shared, the config is private.

**What it measures:** end-to-end latency only, on two canonical samples — `e2el` (cold-cache miss) and `e2el_hit` (warm-cache hit). It is a *latency* harness, not a correctness one: a "no regressions" verdict says nothing about whether the outputs are numerically identical, so a PR that changes numerics still needs its own parity check.

**How to use it, in short.** Set the server up once ([One-time setup](#one-time-setup)), then:

```bash
export AB_SERVER=my-gpu    # your ssh alias for the benchmark box
./ab_benchmark.sh 358      # A/B PR #358 on the default scenario
```

`ab_benchmark.sh` runs on your machine: it SSHes to the server, has `ab_runner.sh` check out both revisions and run the four passes, then polls for the report, fetches it and prints it. [Usage](#usage) covers scenarios and options; [Reading the results](#reading-the-results) explains how to interpret the table.

## Directory structure

| File | Where it runs | Purpose |
| --- | --- | --- |
| `ab_benchmark.sh` | **Local (developer machine)** | Entry point: resolves the PR's base/head SHAs → SSHes to the server to launch A-B-B-A → polls for the report → fetches and prints it |
| `ab_runner.sh` | Server | Runs the A-B-B-A four-pass offline benchmark (checkout → docker `run_offline_benchmark.py` → `generate_summary.py`) |
| `compare_ab.py` | Server | Compares the two sides' summaries and emits a markdown + json report |
| `benchmark.env.example` | Server | Config template (container name / GPU / paths); copy to `benchmark.env` (git-ignored) |
| `run_offline_benchmark.py` | Server (in container) | Offline beam-search benchmark engine, invoked directly by `ab_runner.sh` |
| `generate_summary.py` | Server (in container) | Aggregates `raw-result.json` → `summary.json` |
| `download_onerec_dataset.py` | Server (in container) | Verifies / downloads the OneRec dataset |
| `instrumentation/` | Server (in container) | Lightweight CPU timing probes (`sitecustomize.py` + `vllm_gr_lightweight_timing.py`), injected via `PYTHONPATH` |
| `constraints/` | Server (in container) | Constraint catalog triples doc (`video_constraint_triples.json`) for constrained beam decoding, committed with the tooling |

**This directory is self-contained**: every dependency above is committed here, so an A/B run needs nothing from `tools/daily_benchmark/`. Do not confuse the two — that other directory is the *daily* benchmark toolkit, whose source of truth is the dashboard repo (`DINGEde/vllm-gr-performance-dashboard`) and which this repo keeps only as a git-ignored local snapshot. The two are separate harnesses; AB mirrors the daily runner's engine configuration so that their E2E numbers stay comparable — see [Known limitations](#known-limitations).

## Prerequisites

**Local**: `bash`, `ssh`; [`gh`](https://cli.github.com/) (with `gh auth login`) only if you resolve a PR number.

**Server** (all four required):

1. **SSH access** — an alias in `~/.ssh/config` using your own key (never share private keys).
2. **A benchmark container with the vllm-gr image** — inside the container `/opt/vllm-gr` mounts the vllm-gr repo, and `nvidia-smi` / `torch.cuda` work.
3. **The OneRec dataset** — at `/opt/vllm-gr/data` in the container (`download_onerec_dataset.py --tasks video` downloads/verifies it).
4. **HF model access** — the container can pull `OpenOneRec/OneRec-1.7B` (`HF_ENDPOINT` can point at a mirror).

## One-time setup

### 1. Server side: `benchmark.env`

```bash
# in the vllm-gr worktree on the server
cd tools/AB_Tests
cp benchmark.env.example benchmark.env
vi benchmark.env    # set PROJECT_DIR / CONTAINER_NAME / GPU_INDEX / CONTAINER_DATA_DIR / ...
```

### 2. Local: SSH alias + env vars

```bash
# ~/.ssh/config
Host my-gpu
    HostName <your server IP>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

```bash
# optional: set defaults in your local shell so you don't pass --server every time
export AB_SERVER=my-gpu
export AB_REMOTE_DIR=/path/on/server/to/vllm-gr
```

## Usage

```bash
# A/B PR #358 with the default scenario (beam 128, input 1024)
./ab_benchmark.sh 358

# pick server / scenario
./ab_benchmark.sh 358 --server my-gpu --scenario 256:1024

# compare any two commits (no gh needed)
./ab_benchmark.sh --base <base_sha> --head <head_sha>

# also download the report locally
./ab_benchmark.sh 358 --save-dir ./reports

# A/B a PR whose base predates beam_search_v1
./ab_benchmark.sh 358 --beam-api legacy
```

The **scenario** defaults to `128:1024` (`beam_width:input_tokens`; beam width is restricted to 64/128/256). Tuning such as the number of prompts and warmup requests lives in the server's `benchmark.env`. `DIAGNOSTIC_PROMPTS` there re-enables the phase decomposition, which is off by default and is for ad-hoc debugging only — see [Known limitations](#known-limitations).

`--beam-api` is passed through to both sides and overrides the server's `BEAM_API`; omit it and the server default (`v1`) applies. It must match on both sides — see [Known limitations](#known-limitations).

A PR's four A-B-B-A passes (each reloads the model + 100 requests) take roughly **10–15 minutes**; the script polls until the report is generated or it times out (default 1 hour).

## Reading the results

The report is a `base→head (delta%)` table; **positive delta% = slower (regression), negative = faster (improvement)**. Noise thresholds: mean/p50 +/-2%, p90 +/-3%, p99 +/-5%; values beyond threshold are flagged ⚠️.

Pitfalls to know:

- **The baseline is the merge base (fork point) of the PR head and `origin/decode_graph`, not the current `origin/decode_graph` tip** — the tip may already include other merged PRs and would mix in unrelated changes. `ab_runner.sh` computes `git merge-base origin/decode_graph <head>` server-side; `ab_benchmark.sh` passes only the head, so a stale "Base: xxx" from the PR body is never used.
- **A p50 ⚠️ is often just the hit/miss bimodal boundary**: `e2el`'s p50 can land in the fast (hit) sample band while mean/p90/p99 land in the slow (miss) band. Judge a real regression by whether **mean/p90/p99 move in the same direction**; a p50-only move is usually not real.
- The report has exactly two rows: `e2el` (cold-cache miss) and `e2el_hit` (warm-cache hit). They are the same canonical samples the dashboard's E2E trend uses, but the phase decomposition behind them is not measured here.

## Known limitations

- **Both sides must run the same beam entry point.** `--beam-api` (client) / `BEAM_API` (server `benchmark.env`) selects `v1` (`beam_search_v1`, the default) or `legacy` (`GRLLM.beam_search()`). If the older of the two commits predates `beam_search_v1`, set `BEAM_API=legacy` — the base side cannot run a path it does not have, and a flat `v1`-only PR result is expected when comparing against a pre-`v1` base.
- **AB measures E2E only** (2026-09-18). The Prefill/Decode phase decomposition is off by default (`--diagnostic-prompts 0`) and no longer appears in the report, so a change that only moves work between phases will read as flat. Set `DIAGNOSTIC_PROMPTS>0` in `benchmark.env` to collect it again; those samples still run after canonical sampling and never contribute to the E2E numbers.
- **The retained phase decomposition is stale — do not compare it with the dashboard.** It is frozen at `phase_definition.version = vllm-gr-beam-search-v1-cuda-ready-v4`, while the dashboard has since moved to `...-gpu-compute-v5` (it split device-idle out of the stage spans). Re-enabling `DIAGNOSTIC_PROMPTS` therefore yields numbers on a caliber the dashboard no longer publishes; treat them as a local debugging aid, never as a trend point.
- **AB and daily numbers are only comparable from 2026-09-18 on.** `build_engine_kwargs()` used to run AB's own long-standing tuning (`max_logprobs=1024` / `max_num_seqs=1024` / `max_num_batched_tokens=16384` / `gpu_memory_utilization=0.90` / `max_model_len=max(8192, …)`). It now mirrors the live daily runner — `tools/daily_benchmark/run_offline_benchmark.py` in the dashboard repo, `DINGEde/vllm-gr-performance-dashboard` (`max_logprobs=beam_width` / `max_num_seqs=1` under `v1` / `max_num_batched_tokens=40960` / `gpu_memory_utilization=0.5` / `max_model_len=40960`, with prefix caching and chunked prefill on), so a delta measured across that date mixes the config change in. A git-ignored copy of that runner sits under this repo's `tools/daily_benchmark/` for local convenience and is kept in sync with the dashboard repo, but it is **not** authoritative — read the configuration from the dashboard repo, not from here.
- The report is only saved/printed; it is **not** auto-commented to the PR — whether to post it is up to you.
- Server-side `ab_runner.sh` requires the worktree to be on a clean `decode_graph` branch, and it runs `git fetch origin decode_graph` and `git fetch origin pull/<N>/head` (the server must be able to reach GitHub).

## How it survives `git checkout`

`ab_runner.sh` switches the worktree between base and head with `git checkout --detach <sha>`. Because the benchmark engine now lives in the *tracked* `tools/AB_Tests/` directory, checking out a base/head commit that predates this directory would delete those files from the working tree mid-run. To avoid that, `ab_runner.sh` first copies the engine (`run_offline_benchmark.py`, `generate_summary.py`, `download_onerec_dataset.py`, `compare_ab.py`, `instrumentation/`, and `constraints/`) into the git-ignored `results/ab/.tooling/` directory and runs everything from there. That staging dir sits under `results/`, which is git-ignored and holds only untracked files, so `git checkout` never touches it — the A/B can run against any base/head commit regardless of when this tooling was added.

## Working on this directory

`tools/AB_Tests/` is tracked, so every file here goes through the repository's pre-commit gate. Two hooks matter when porting code in from the dashboard, whose copy of the same engine is **not** under this gate:

- **`check-torch-cuda-call`** rejects new `torch.cuda.*` calls (`synchronize`, `Event`, `current_device`, …) and its `ALLOWED_FILES` allowlist is empty. Use the device-agnostic spellings: `torch.accelerator.synchronize()`, `torch.accelerator.current_stream(device)`, `torch.Event(enable_timing=True)`.
- **`check-characters`** rejects non-English characters outside its allowlist. `→`, `⚠️`, `—` and `…` pass; the Greek capital delta does not, which is why the report spells out `delta%`.

On a Windows checkout `core.autocrlf=true` keeps the working tree in CRLF while git stores LF, so `scp`-ing the working tree to a server leaves CRLF shell scripts that die with `set: pipefail: invalid option name`. Strip them after copying with `sed -i 's/\r$//'`.
