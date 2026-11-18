# Prepare a Constraint Table for OneRec serving

Run the commands from the repository root in Bash. Generate the catalog
from the OneRec dataset or supply your own OneRec SID catalog.

## Set the model and dataset paths

Set the paths to a local `OpenOneRec/OneRec-1.7B` snapshot with its tokenizer
and to the dataset root containing `benchmark_data/sid2pid.json`. If needed,
[download the model and mapping](../../../docs/onerec_v1_async_quickstart.md#1-check-the-environment).

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA=/absolute/path/to/OpenOneRec-RecIF
export ONEREC_WORKDIR="$PWD/results/onerec-serving"
export ONEREC_BACKEND=cuda  # Set to npu on Ascend.
mkdir -p "$ONEREC_WORKDIR"
```

## Compile the Constraint Table

If you already have a catalog of OneRec SIDs with their boundary markers,
save it as `$ONEREC_WORKDIR/catalog.json` and begin with
`tools/build_constraint_triples.py`. For conversion options and input formats,
see the [converter documentation](../../../benchmarks/open_one_rec/tools/README.md).

Compile the table with the tokenizer from the serving model snapshot. The
compiler writes the artifact and prints the two digests used in the server
configuration.

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA/benchmark_data" \
  --output-path "$ONEREC_WORKDIR/catalog.json" \
&& python tools/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/catalog.json" \
  --output "$ONEREC_WORKDIR/constraint_triples.json" \
  --begin '<|sid_begin|>' --end '<|sid_end|>' \
&& python tools/build_constraint_table.py \
  --triples "$ONEREC_WORKDIR/constraint_triples.json" \
  --model "$ONEREC_MODEL" \
  --output "$ONEREC_WORKDIR/constraint_table.safetensors" \
  --trust-remote-code > "$ONEREC_WORKDIR/digests.txt" \
&& cat "$ONEREC_WORKDIR/digests.txt"
```

## Write the server configuration

Generate the server YAML using the printed digest values.

```bash
ARTIFACT_DIGEST=$(awk '$1 == "artifact_digest:" {print $2}' "$ONEREC_WORKDIR/digests.txt")
TOKENIZER_DIGEST=$(awk '$1 == "tokenizer_digest:" {print $2}' "$ONEREC_WORKDIR/digests.txt")
cat > "$ONEREC_WORKDIR/vllm_gr_config.yaml" <<EOF
schema_version: 1
attention_backend: CUSTOM
beam:
  execution_mode: v1
  graph_enabled: true
  worker_decision: true
  max_width: 128
  max_decode_steps: 3
constraint_table:
  enabled: true
  backend: $ONEREC_BACKEND
  max_top_k: 128
  path: "$ONEREC_WORKDIR/constraint_table.safetensors"
  format: constraint_table_v1
  artifact_digest: "${ARTIFACT_DIGEST:?Build the Constraint Table first}"
  tokenizer_digest: "${TOKENIZER_DIGEST:?Build the Constraint Table first}"
EOF
```

## Change the beam width

Set the request width to match `beam.max_width`. For the example width of
128, omitting `constraint_table.candidate_width` makes it resolve to 128.
For widths up to 256 on CUDA, the candidate width can follow the beam width.
Set `constraint_table.max_top_k` to cover the candidate width. Set
`--max-logprobs` to at least the candidate width and `--max-num-seqs` to at
least the beam width for these server examples.

For a CUDA beam width of 512, change these fields in the generated YAML.
The per-beam candidate limit is 256.

```yaml
beam:
  max_width: 512
constraint_table:
  candidate_width: 256
  max_top_k: 256
```

With this configuration, use `--max-logprobs 256` or higher and
`--max-num-seqs 512` or higher in the example server command. Restart the
server, then request width 512 with HTTP `-b 512` or gRPC `--width 512`.

A constrained request can return fewer beams if the catalog has too few
reachable SIDs. Pass `--expected-count` when you know the expected result
count. Omit it to check SID format and membership for a variable count.

## Verify a saved response

From the repository root, verify the saved response. The verifier accepts
OneRec three-layer SIDs with begin and end markers.

```bash
python examples/online_serving/catalog/verify_catalog.py \
  "$ONEREC_WORKDIR/result.json" "$ONEREC_WORKDIR/catalog.json" --expected-count 128
```

The verifier reads `choices[].text` from a Completions response. It also
accepts saved Chat Completions responses with SIDs in
`choices[].message.content`. Server errors, empty choices, incomplete SIDs
and SIDs outside the catalog cause a nonzero exit.
