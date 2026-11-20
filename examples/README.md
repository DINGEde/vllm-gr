# Examples

## Run OneRec offline

Install the [CUDA runtime](../docs/getting_started/installation/cuda.md) or
[Ascend runtime](../docs/getting_started/installation/ascend.md). Run these commands from the
repository root in Bash. The first run needs only a local OneRec-1.7B model
snapshot with its tokenizer.

If you already have the snapshot, set `ONEREC_MODEL` to its directory.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
```

To download the snapshot with the Hugging Face CLI, use a directory outside
the checkout.

```bash
export ONEREC_MODEL="$HOME/.cache/vllm-gr/models/OneRec-1.7B"
hf download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
```

### CUDA

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

### Ascend

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --constraint-backend npu --execution eager
```

### Expected output

With the defaults, `request=0 returned_beams=128` means the first request
returned 128 candidates. The script shows the five highest-ranked candidates
with their `beam_tokens` and `cumulative_logprob`. It uses
`Recommend the next video.` by default. Pass `--prompt /path/to/prompt.txt`
to use a UTF-8 file at its natural token length.

Full-vocabulary search can return candidates outside your catalog.
[Try a small catalog](offline_inference/beam_search/README.md#run-with-a-constraint-table)
to restrict the output to known items.

## Run OneRec over HTTP

Use the same local model and apply the
[CLI dispatch patch](../docs/cli/README.md#enable-gr-dispatch). On Linux
with Bash and curl 7.76 or later, run:

```bash
python examples/online_serving/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

On Ascend, append `--constraint-backend npu`. The command starts a local
server, sends one prompt, prints the returned SIDs, verifies them against the
two-entry sample catalog, and stops the server. If both SIDs are returned, the
last verification line reads `SUCCESS: All 2 generated SIDs exist in the catalog.`

Both quickstarts accept `--prompt FILE` for a natural-length UTF-8 prompt
and `--catalog FILE` for constraint triples. The offline default searches the
full vocabulary. The HTTP default searches the sample catalog. To compare
the two, supply the same model, prompt and triples to both. See the
[HTTP guide](online_serving/beam_search/README.md) for custom inputs and
instructions for keeping a server running.

## Run gRPC

- [Check the CUDA gRPC installation](online_serving/grpc/README.md#check-the-grpc-installation)
  with two RPCs against a locally generated tiny Llama.
- [Serve a real OneRec model over gRPC](online_serving/grpc/README.md#serve-onerec-over-grpc)
  with the same catalog and server configuration as HTTP.

## Run Sidecar hot updates

Follow the [Sidecar example](sidecar/README.md) to configure checkpoint and
Trie updates for running gRPC instances and check the version on each instance.

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

Both runners use the same Beam Search API and `gr` configuration.
CUDA Prefill and Decode capture policies differ by runner. See the
[V2 integration guide](../docs/features/cuda-graphs.md#modelrunner-v2-multi-batch-execution) and
[Prefill capture guide](../docs/features/cuda-graphs.md#prefill-capture-configuration).

## Run Pangu offline

Follow the [Pangu evaluation guide](../docs/contributing/tools/pangu1b.md#run-the-offline-accuracy-evaluation) to run GSM8K-mini with the local checkpoint.

## Measure performance

- [Measure offline Beam Search latency](offline_inference/beam_search/README.md#measure-latency-without-profiling)
- [Measure offline or online throughput](../docs/usage/benchmarks/throughput.md)
