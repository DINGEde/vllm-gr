# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -euo pipefail

width=128
prompt_file=
while getopts ":b:p:h" opt; do
  case "$opt" in
    b) width="$OPTARG" ;;
    p) prompt_file="$OPTARG" ;;
    h) echo "Usage: bash client.sh [-b beam_width] [-p prompt_file]"; exit 0 ;;
    *) echo "Usage: bash client.sh [-b beam_width] [-p prompt_file]" >&2; exit 2 ;;
  esac
done
shift "$((OPTIND - 1))"
if [[ $# -ne 0 || ! "$width" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected a positive integer beam width and no positional arguments." >&2
  exit 2
fi

model="${ONEREC_SERVED_MODEL_NAME:-onerec}"
base_url="${ONEREC_BASE_URL:-http://127.0.0.1:8000}"
python_bin=
for candidate in python3 python; do
  if "$candidate" -c 'import sys; sys.exit(sys.version_info.major != 3)' >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "Python 3 is required to encode the request JSON." >&2
  exit 2
fi
payload="$("$python_bin" - "$width" "$prompt_file" "$model" <<'PY'
import json
import sys
from pathlib import Path

width, prompt_file, model = sys.argv[1:]
prompt = "Recommend the next video."
if prompt_file:
    try:
        with Path(prompt_file).open(encoding="utf-8", newline="") as source:
            prompt = source.read()
    except (OSError, UnicodeError) as exc:
        sys.exit(f"Could not read prompt file: {exc}")

print(json.dumps({
    "model": model,
    "prompt": prompt,
    "use_beam_search": True,
    "n": int(width),
    "temperature": 0.0,
    "max_tokens": 5,
    "ignore_eos": True,
    "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
}))
PY
)"

headers=(-H "Content-Type: application/json")
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  headers+=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

exec curl --silent --show-error --fail-with-body --connect-timeout 5 --max-time 300 \
  "${headers[@]}" --data-binary "$payload" "${base_url%/}/v1/completions"
