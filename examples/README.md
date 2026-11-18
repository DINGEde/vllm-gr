# Examples

## Run OneRec offline

Install the [CUDA runtime](../docs/quickstart.md) or
[Ascend runtime](../docs/quickstart-ascend.md). Download
[OneRec-1.7B and its tokenizer](../docs/onerec_v1_async_quickstart.md#1-check-the-environment).
Run the command for your device from the repository root.

### CUDA

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model /path/to/OneRec-1.7B
```

### Ascend

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model /path/to/OneRec-1.7B --constraint-backend npu --execution eager
```

### Expected output

The script prints `request=0 returned_beams=128`, followed by the tokens and
scores of the first five candidates. It uses `Recommend the next video.` by
default. Pass `--prompt /path/to/prompt.txt` to use a UTF-8 file at its
natural token length.

Full-vocabulary search can return invalid SIDs. Use a catalog to restrict the
output to known items.

- [Generate SIDs from your catalog](offline_inference/beam_search/README.md#run-with-a-constraint-table)
- [Serve OneRec over HTTP and verify a response](online_serving/beam_search/README.md)
- [Serve OneRec over gRPC](online_serving/grpc/README.md)
- [Measure offline Beam Search latency](offline_inference/beam_search/README.md#measure-latency-without-profiling)
- [Measure offline or online throughput](../docs/benchmarks/throughput.md)
- [Run offline Pangu inference](offline_inference/pangu1b/README.md)

## Run Sidecar hot updates

Copy the [configuration template](sidecar/config.py). Set each task's HDFS
address and a cache directory that its target instances can read. Set each
gRPC instance's management address to use its `VLLM_GR_ADMIN_PORT`. Run
`vllm-gr-sidecar --config /path/to/config.py`. See the
[deployment guide](../docs/weight_update_sidecar_design.md#user-startup-and-publication-guide)
for checkpoint rolling updates and online Trie switching.

## Choose a ModelRunner

Direct offline Beam Search, HTTP serving, and gRPC serving default to
ModelRunner V2 on CUDA. To select V1, set the variable below before startup.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

Use `VLLM_USE_V2_MODEL_RUNNER=1`, an empty value, or an unset variable to
select the CUDA example default. Restart the process after changing the
setting. The Ascend examples and gRPC tiny-model smoke test use V1. The
library, general CLI, and offline thinking modes default to V1 when the
variable is unset.

`beam.execution_mode: v1` selects the same Beam Search API for both runners.
CUDA Prefill and Decode capture policies differ by runner. See the
[V2 integration guide](../docs/modelrunner_v2_multibatch.md) and
[Prefill capture guide](../docs/prefill_graph_capture.md).
