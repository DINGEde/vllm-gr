# Accuracy Benchmark

## Overview

This guide explains how to run the accuracy benchmark using the [`one_rec_acc_test.py`](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/open_one_rec/one_rec_acc_test.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../../getting_started/installation/cuda.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Accuracy Mode (Simple, Local)

Prepare the matching model snapshot and a canonical V1 configuration with a
compiled Constraint Table. Follow the [artifact and engine setup](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md#constraint-table-artifact).
Use the same model and tokenizer snapshot for evaluation and artifact
compilation. Set `--n` to match the first entry of `gr.beam_width` in the config. An enabled
Constraint Table fixes generation to exactly three generated SID tokens,
including the token sampled during Prefill; with both OneRec boundary tokens
configured that is public `max_tokens=5`. The video task below uses three. The
unconstrained path accepts one to three, subject to the decode-step capacity
limit.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA="$PWD/data/benchmark_data"
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$ONEREC_MODEL" \
  --dataset-path "$ONEREC_DATA" \
  --task-type video \
  --config ./vllm_gr_config.yaml \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

### 2. Online, API Server Mode (Production-like)

#### Terminal 1 — Prepare the full catalog and start the server

Run these commands from the repository root in Bash on CUDA. Set
`ONEREC_MODEL` to your local model snapshot, prepare the
[dataset](README.md#get-dataset), and apply the
[CLI dispatch patch](../../cli/README.md#enable-gr-dispatch).
The video benchmark requires exactly 128 choices per request. Build the
serving table from the full dataset's video SID mapping.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA="$PWD/data/benchmark_data"
export ONEREC_WORKDIR="$PWD/results/onerec-accuracy"
mkdir -p "$ONEREC_WORKDIR"

python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA" \
  --output-path "$ONEREC_WORKDIR/catalog.json" &&
python -m examples.online_serving.catalog.prepare_serving_config \
  --model "$ONEREC_MODEL" \
  --workdir "$ONEREC_WORKDIR" \
  --raw-catalog "$ONEREC_WORKDIR/catalog.json"
```

After preparation succeeds, start the server with the generated YAML.
The accuracy client uses `--model` as both the API model name and the
tokenizer source. Use the same model snapshot path in both terminals.
The client sends the source chat turns and the task's thinking setting.
`--chat-template-kwargs` can override that setting for an online run.

```bash
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" vllm serve "$ONEREC_MODEL" --gr \
  --host 127.0.0.1 --port 8000 \
  --served-model-name "$ONEREC_MODEL" \
  --config "$ONEREC_WORKDIR/vllm_gr_config.yaml" \
  --chat-template benchmarks/open_one_rec/qwen3_soft_switch.jinja2 \
  --trust-remote-code \
  --max-logprobs 128 --max-num-seqs 128 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --async-scheduling --no-enable-chunked-prefill
```

#### Terminal 2 — Run the accuracy benchmark against the server

Once the server is ready, run the accuracy client from the same checkout.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA="$PWD/data/benchmark_data"
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --endpoint /v1/chat/completions \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path "$ONEREC_DATA" \
  --task-type video \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

For a one-request smoke run, set `--num-prompts 1` and keep `--n 128`.

For an authenticated deployment, pass the configured key with `--api-key`.
Use `--request-timeout` to override the default 180-second HTTP connect and
read timeout. HTTP failures, malformed responses and unexpected choice counts
fail the run. Each choice needs non-blank answer or reasoning text. The client
prefers `reasoning` and falls back to `reasoning_content` for older servers. With
`--result-dir`, the client saves affected sample IDs and errors to
`<task>-failed.json`.

The `label_pred` task requires label-token probabilities. This entry point
currently exits before inference for that task. The evaluator reports an
undefined AUC as `null` with valid and invalid sample counts and a reason
when zero or one class has usable probabilities. Judge-enabled tasks also
fail when the Judge is unavailable or cannot produce valid scores. The
recommendation-reason Judge requires a score for every selected sample and
fails on empty predictions. WIP matching must account for every occurrence
of each reference and model point. Saved Judge results are for diagnostics
and are evaluated afresh on subsequent runs.

For `rec_reason`, set `llm_judge_model` in the task's `evaluation_config` to
the model name exposed by the Judge service. `vllm_endpoint` and `vllm_api_key`
configure its OpenAI-compatible API. `vllm_async=True` uses the asynchronous
SDK inside scoring workers. Each sample score must be an integer from 1 to 5;
the reported mean can be fractional.

## Usage

### Command Line Arguments

#### Core Arguments

| Argument  | Type | Default                | Description              |
|-----------|------|------------------------|--------------------------|
| `--model` | str  | OpenOneRec/OneRec-1.7B | Model ID or path to load |

#### Benchmark related parameters

| Argument           | Type  | Default                 | Description                                                               |
| ------------------ | ----- | ----------------------- | ------------------------------------------------------------------------- |
| `--dataset-name`   | str   | onerec                  | Must be `onerec` for OneRec benchmark                                     |
| `--dataset-path`   | str   | `./data/benchmark_data` | Directory containing the task folders and catalog mappings                |
| `--num-prompts`    | int   | task-specific           | Positive number of prompts to sample                                      |
| `--task-type`      | str   | -                       | See [task support](README.md#open-one-rec-benchmark-dataset).             |
| `--result-dir`     | str   | -                       | Directory to save debug evaluation files                                  |
| `--debug`          | -     | -                       | Enable debug mode for evaluators                                          |
| `--config`         | path  | -                       | Offline engine YAML. Use `--gr` for JSON overrides.                       |
| `--batch-size`     | int   | 1                       | Offline V1 batch size (1, 2 or 4)                                         |
| `--n`              | int   | dataset-specific        | Override the dataset default beam width (number of beams) for beam search |
| `--temperature`    | float | dataset-specific        | Override the dataset default temperature                                  |

#### Arguments for Online benchmark

| Argument     | Type | Default       | Description                                                            |
|--------------|------|---------------|------------------------------------------------------------------------|
| `--endpoint` | str  | -             | API endpoint URL for the online benchmark (e.g., /v1/chat/completions) |
| `--api-key`  | str  | -             | API Key for OpenAI-compatible API                                      |

## Expected Output

The benchmark prints accuracy metrics summarizing model performance.
Example:

```text

...
...
--- Evaluation Results ---
{
  "total_samples": 100,
  "pass@1": 0.03,
  "position1_pass@1": 0.0,
  "recall@1": 0.004523809523809524,
  "pass@32": 0.16,
  "position1_pass@32": 0.04,
  "recall@32": 0.024341269841269844,
  "pid_pass@1": 0.03,
  "pid_position1_pass@1": 0.0,
  "pid_recall@1": 0.004523809523809524,
  "pid_pass@32": 0.16,
  "pid_position1_pass@32": 0.04,
  "pid_recall@32": 0.024341269841269844,
  "select_k_strategy": "first_k",
  "evaluation_mode": "both",
  "sid_to_pid_strategy": "most_popular_after_downsampling"
}
```
