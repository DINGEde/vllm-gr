# OneRec benchmarks

Use the same model/tokenizer, dataset, constraint artifacts and runtime when
comparing runs. Installation requirements are maintained in the
[CUDA](../../getting_started/installation/cuda.md) and [Ascend](../../getting_started/installation/ascend.md) guides.

Apply the [CLI dispatch patch](../../cli/README.md#enable-gr-dispatch) to use
`vllm bench ... --gr`. Run `vllm bench --gr --list` to list workloads and
`vllm bench qwen-v1 --gr --help` to see an example workload's options. Run these
commands from the matching source checkout or one of its subdirectories.

## Choose a benchmark

| Goal | Guide |
| --- | --- |
| Offline Beam Search V1 latency and throughput | [V1 benchmark](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/open_one_rec/V1_BENCHMARK.md) |
| General offline and online throughput | [Throughput](throughput.md) |
| Offline and online business accuracy | [Accuracy](accuracy.md) |
| Standalone latency and timeline profiling | [Qwen benchmark](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/qwen/README.md) |
| CI Video-100 accuracy acceptance | [Video-100 gate](parent_score_gate.md) |
| CI five-task Beam V1 Recall acceptance | [Recall gate](beam_v1_recall_gate.md) |
| Dataset and catalog preparation utilities | [OneRec tools](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/open_one_rec/tools/README.md) |

## Open One Rec Benchmark Dataset

### Get Dataset

Obtain access to [OpenOneRec-RecIF](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)
before downloading it. A Git LFS checkout can be prepared with:

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
export ONEREC_DATA="$PWD/data/benchmark_data"
test -f "$ONEREC_DATA/video/video_test.parquet"
test -f "$ONEREC_DATA/sid2pid.json"
```

The accuracy CLI and catalog tools default to `./data/benchmark_data`.
The examples pass `--dataset-path "$ONEREC_DATA"` explicitly. For a custom
dataset, set `ONEREC_DATA` to the directory that directly contains the task
folders and catalog mappings. Tasks use files such as
`video/video_test.parquet` and `interactive/interactive_test.parquet` beneath it.
Parquet rows contain `messages` (chat messages as JSON or a list) and `metadata`
(answer/context metadata as JSON or a mapping).

| Task | Purpose | Accuracy CLI support |
| --- | --- | --- |
| `rec_reason` | Recommendation explanation | Online long-text inference with a configured Judge |
| `item_understand` | Caption generation from a video SID | Online long-text inference with a configured Judge |
| `ad` | Ad engagement prediction | Online and offline SID recommendation scoring |
| `product` | Product recommendation | Online and offline SID recommendation scoring |
| `label_cond` | Conditional label prediction | Online and offline SID recommendation scoring |
| `video` | Video recommendation | Online and offline SID recommendation scoring |
| `interactive` | Interactive engagement prediction | Online and offline SID recommendation scoring |
| `label_pred` | Engagement classification | Requires label logprobs. The accuracy CLI exits before inference. |

Business evaluation also needs the catalog mappings required by the selected
task. A triples file used to build a Constraint Table does not replace those
mappings. Follow the selected accuracy or gate guide for its exact inputs.

## Interpreting results

Keep raw outputs, input identities, effective configuration and timing boundaries
with each result. Separate startup, warmup, steady-state inference and profiling
overhead. Record actual graph replay and fallback behavior; enabling graphs is
not proof that a particular input used one.

Different batch shapes and BF16 execution can change candidate order. Use the
benchmark's business metrics for acceptance and report numerical comparisons
separately. CI thresholds come from
[the benchmark policy](https://github.com/JiusiServe/vllm-gr/blob/main/.github/benchmark-gate.yaml); see
[policy selection](../../contributing/ci/benchmark-policy-approvals.md) for how a run resolves it.
Historical results do not establish that a new checkout passes those gates.
