# Benchmarks

Use a [source checkout that matches the installed runtime](../docs/cli/README.md#benchmarks).
Follow the [GR dispatch setup](../docs/cli/README.md#enable-gr-dispatch) to run
`vllm bench --gr --list` and invoke the registered scenarios.

| Directory | Workloads | Guides |
| --- | --- | --- |
| `beam_search/` | `beam-performance`, `beam-performance-ab`, `beam-recall` | [Benchmark guides](../docs/usage/benchmarks/README.md#choose-a-benchmark), [Recall gate](../docs/usage/benchmarks/beam_v1_recall_gate.md) |
| `open_one_rec/` | `open-one-rec` throughput, OneRec accuracy, dataset and catalog tools | [OneRec benchmarks](../docs/usage/benchmarks/README.md), [Accuracy](../docs/usage/benchmarks/accuracy.md), [Catalog tools](open_one_rec/tools/README.md) |
| `qwen/` | `qwen-v1` offline Beam performance and profiling | [Qwen benchmark](qwen/README.md) |

[`metrics.py`](metrics.py) contains shared percentile calculations for the Beam
latency gate and Qwen benchmark.
