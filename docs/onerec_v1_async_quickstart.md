# OneRec-1.7B V1 asynchronous scheduling quickstart

This guide runs the new EngineCore-owned asynchronous Beam Search V1 path
with OneRec-1.7B. One offline request exercises Prefill, asynchronous Decode,
GPU Beam decisions, and the CUDA Constraint Table. Keep the values in this
guide unchanged for the first run.

> The V1 reproduction entry point is offline `GRLLM.beam_search_v1()`.
> The `vllm-gr serve` and OpenAI Chat API examples are separate online and
> compatibility paths. Do not add the V1 configuration from this guide to
> those commands.

## 1. Check the environment

Use a CUDA GPU, Python 3.11 or later, vLLM 0.22.1, and the matching
`vllm-gr` installation. See [quickstart](quickstart.md) for installation.
From the repository root, verify the environment:

```bash
nvidia-smi
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -c 'from vllm_gr.entrypoints.gr import GRLLM; print("v1 import OK")'
```

Set local paths for the model, dataset, and generated files:

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA=/absolute/path/to/OpenOneRec-RecIF
export ONEREC_WORKDIR="$PWD/results/onerec-v1"
mkdir -p "$ONEREC_WORKDIR"
```

`ONEREC_DATA` must contain `sid2pid.json`. See the
[OneRec dataset instructions](benchmarks/README.md#open-one-rec-benchmark-dataset).
Use the same model and tokenizer version when building the Constraint Table.

If the model and dataset are not available locally, accept the required access
terms on Hugging Face and sign in first. Then download them, reserving enough
disk space for the dataset:

```bash
huggingface-cli download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF "$ONEREC_DATA"
```

## 2. Build the V1 Constraint Table inputs

V1 uses a CUDA Constraint Table rather than a plain catalog JSON. Convert
`sid2pid.json` to a SID catalog, then create the triples document:

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA" \
  --output-path "$ONEREC_WORKDIR/video_catalog.json"

python tools/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/video_catalog.json" \
  --output "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --begin '<|sid_begin|>' \
  --end '<|sid_end|>'
```

Rebuild these files only when the catalog changes. The standalone example
compiles a temporary `.safetensors` artifact from the triples document and the
model tokenizer. To create a reusable artifact for an integration, run:

```bash
python tools/build_constraint_table.py \
  --triples "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --model "$ONEREC_MODEL" \
  --output "$ONEREC_WORKDIR/video_constraint_table.safetensors" \
  --trust-remote-code
```

The command prints `artifact_digest` and `tokenizer_digest`. Use both values
with the artifact absolute path in an integration's `constraint_table`
configuration. The standalone example in the next section deliberately builds
a temporary artifact and does not read this persisted file. See the
[artifact configuration reference](../examples/offline_inference/beam_search/README.md#constraint-table-artifact)
for the full configuration.

## 3. Run the minimal V1 reproduction

Create a UTF-8 prompt file. The following ASCII-only input is enough to check
the scheduling path:

```text
Video history: <|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>. Recommend the next video.
```

Save it as `$ONEREC_WORKDIR/prompt.txt`, then run:

```bash
USE_TF=0 \
python examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$ONEREC_MODEL" \
  --catalog "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --input-len 1024 \
  --generated-steps 3 \
  --beam-width 128
```

The first run loads the model, compiles the temporary table, and captures the
fixed-width CUDA graph, so it is slower than later runs. A successful run
prints `returned_beams=128` followed by the first five candidates. Each
`sid_tokens` value contains a begin marker, three SID components, and an end
marker. Scores and ranking can differ across GPU, CUDA, weights, and prompts;
do not use one fixed score as the pass condition.

## 4. Why these values are fixed

The new scheduler has an execution contract, rather than being a performance
switch for arbitrary requests. Keep these values for the first reproduction:

| Setting | First-run value | Reason |
| --- | ---: | --- |
| Scheduling | `async_scheduling=True` | EngineCore submits the next stage early. |
| Beam width | 128 | Startup allocates the KV pool and captures a fixed-shape graph; request width must match exactly. |
| Generation steps `N` | 3 | N includes the first token from Prefill; only 1, 2, and 3 are supported. |
| EOS | `ignore_eos=True` | Legal EOS candidates are retained; finish after N outputs. |
| Prefill | `enable_chunked_prefill=False` | The complete remaining prompt must fit the native token budget. |
| Dispatch capacity `Q` | 2 | Two stages may be in flight for the same logical owner; this does not make B=2. |
| `max_tokens` | 5 | The example has begin and end positions, so `N = 5 - 2 = 3`. |
| Constraint | CUDA Constraint Table | Produces catalog-valid SIDs. |
| Parallelism and requests | TP/PP/DP/DCP/PCP are 1; one request | One execution owner holds the physical dispatch slots. |

The example script sets `execution_mode: v1`, `worker_decision: true`,
`graph_enabled: true`, Prefix Cache, and `max_num_seqs == beam_width`.
When changing the Beam width, also update the request, `beam.max_width`,
Constraint Table `max_top_k`, and engine capacity. Do not change only the
client `beam_width`.

### Batching several prompts in one HTTP request

The V1 completions route renders a prompt **list** once and partitions it into
contiguous execution batches of 4, 2, or 1. Each batch runs one session; the
next batch is submitted only after the previous call completes successfully.
The engine retains one execution owner. For example:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "OpenOneRec/OneRec-1.7B",
    "prompt": ["<prompt 0>", "<prompt 1>", "<prompt 2>", "<prompt 3>"],
    "use_beam_search": true,
    "n": 128,
    "temperature": 0.0,
    "max_tokens": 5,
    "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"}
  }'
```

Any nonempty list length can be submitted, subject to normal request limits.
With compatible parameters and sufficient capacity, 3 prompts run as `2+1`,
5 as `4+1`, 6 as `4+2`, and 8 as `4+4`. One prompt keeps the original route.
The planner preserves input order; it does not reorder prompts to fill batches.

Each execution batch must fit `max_num_seqs` and `max_num_batched_tokens`,
including the begin token appended to every prompt. If configured, a smaller
`max_num_scheduled_tokens` also limits the batch. Insufficient capacity causes
smaller sub-batches, without rendering again. An individually oversized prompt
is a request error, even if previous prompts would fit. Engine KV admission
remains authoritative; the frontend plan does not guarantee device capacity.

V1 requires Chunked Prefill to be **off**. Enabling it is refused when the engine
starts (`beam_search_v1 requires enable_chunked_prefill=False`), not split:
there is no partial Prefill on this path, and the complete prompt has to fit the
token budget.

A batch runs one `max_tokens`, so adjacent prompts with different effective
generation budgets are split. No prompt's budget is reduced to match another.

The response is a single object whose `choices` are flat-indexed prompt-major,
so `k` prompts × `n` beams give `k × n` choices, and `usage.prompt_tokens` sums
across the entire HTTP request. Results are assembled once after all sub-batches
succeed. A failed sub-batch stops subsequent submissions and returns an error;
completed work is not replayed. Cancellation cleans up the current session and
prevents later submissions. Completed work cannot be rolled back. Independent
HTTP requests are not coalesced, and other requests may run between sub-batches.

Each claimed request emits an INFO-level `Beam batch plan` log with its request
ID, input count, sub-batch count, and sizes. For example, `sizes=[1, 1, 1, 1]`
makes singleton-only execution visible. This describes the plan, not successful
execution, and does not log prompt contents or change the response schema.

At the completions adapter boundary, cancellation during child execution returns
the same `Client disconnected` error response as upstream vLLM 0.22.1 after
iterator cleanup. It intentionally does not re-raise there; lower-level
engine/client cancellation behavior is unchanged.

Preserving order and generation budgets does not guarantee identical beam
rankings to separate B=1 calls. GPU batch shapes and prefix-cache execution
paths can produce different candidates or scores. Validate model quality for
the batch sizes you deploy; the split/merge system test compares against manual
partitions with matching batch shapes and cache conditions.

#### Execution flow and ownership

The HTTP list is a response boundary, not a GPU execution-batch boundary:

```text
one HTTP request: [p0, p1, p2, p3, p4, p5, p6, p7, p8]
  -> render all inputs once; resolve each generation budget
  -> validate individual inputs and plan contiguous batches
  -> session [p0..p3] -> close -> session [p4..p7] -> close -> session [p8]
  -> merge outputs in input order -> one CompletionResponse
```

At each input position, the planner chooses the largest supported batch that
fits sequence/token capacity and has one shared generation budget. This is a
greedy, order-preserving policy, not a global bin-packing optimizer. All lengths,
budgets, and beam parameters are checked before submitting the first session;
runtime resource admission and failures still belong to the engine.

`GRExecutionCapacity.active_sessions` remains 1. Each session fixes its members
before it starts, and the frontend waits for its iterator and cleanup before
submitting the next one. No members join or leave a running batch. Single owner
therefore does not mean that one HTTP request can contain only one batch, nor
does it give the whole HTTP request exclusive ownership across all sub-batches.

This behavior applies to multi-prompt, non-streaming V1 beam completions. Single
prompts, non-beam requests, streaming, embedded prompts, and non-V1 engines keep
the original handler. It is not cross-request coalescing or continuous batching,
and it introduces no new endpoint or client-side batch-size requirement.

## 5. Validate and troubleshoot

With real model weights, run the system test for actual offline and online V1
calls, N=1/2/3, Prefix Cache, graph replay, cancellation/reuse and release at return.
Select an available GPU using the environment's existing device allocation rules:

```bash
USE_TF=0 \
VLLM_GR_TEST_MODEL="$ONEREC_MODEL" \
python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py
```

For the completions route, run `tests/st/test_onerec_v1_completion_batch_st.py`
with the same environment. It checks 3/5/6-prompt splitting, actual session
dispatches, response ordering/usage, and cancellation followed by reuse.

| Symptom | Resolution |
| --- | --- |
| `async_scheduling=True` is required | Use the V1 example and keep this `GRLLM` initialization option. |
| Width or generation-step error | Keep request width equal to `max_width` and use 1 through 3 effective generation steps. |
| Constraint Table digest or tokenizer mismatch | Rebuild the artifact from the current model directory and triples document. |
| CUDA graph or out-of-memory error | Start with width 128 and the default `--gpu-memory-utilization 0.5`; check for competing GPU processes. |
| Online V1 validation is needed | The system test calls `OpenAIServing.beam_search_v1()` with a real asynchronous multiprocess engine. Legacy HTTP gates remain separate. |
| EOS or Prefill admission error | Set `ignore_eos=True`, disable Chunked Prefill and allow the complete prompt in `max_num_batched_tokens`. |
| A prompt list uses small execution batches | Check sequence/token capacity and equal effective generation budgets; all three constrain sub-batch size. |

## Further reading

- [Asynchronous scheduling execution contract](async_scheduling.md): scheduling, Prefill/Decode overlap, and retirement.
- [Offline Beam Search V1 reference](../examples/offline_inference/beam_search/README.md#beam-search-v1): full configuration fields and script arguments.
- [Asynchronous scheduling validation](async_scheduling_validation.md): historical precision and graph-replay results.
- [OneRec benchmark](benchmarks/README.md): dataset, online benchmarks, and accuracy evaluation.
