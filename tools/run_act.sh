#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -x
set -euo pipefail

source "./tools/devices/resource.sh"
idle_gpu=$(get_gpu 4)


if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi

# Lock the GPU to prevent other scripts from using it
lock_pid=$(lock_gpu "$idle_gpu")
if [ -z "$lock_pid" ]; then
    echo "Failed to lock GPU $idle_gpu"
    exit 1
fi

secret_file=""
cleanup() {
    if [[ -n "$secret_file" ]]; then
        rm -f -- "$secret_file"
    fi
    echo "Unlocking GPU $idle_gpu (PID: $lock_pid)..."
    unlock_gpu "$lock_pid"
}
trap cleanup EXIT

cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
token=()
# Keep the token out of the trace for both the check and the file write.
set +x
if [[ -n "${HF_TOKEN:-}" ]]; then
    secret_file=$(mktemp)
    printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > "$secret_file"
    token=("--secret-file" "$secret_file")
fi
set -x
act "${token[@]}"  --container-options "--runtime=nvidia --gpus device=$cuda_dev -v ${HOME}/.cache/act-cache:/data/actions/cache" \
--env NVIDIA_DRIVER_CAPABILITIES=all --artifact-server-path "${HOME}"/artifacts \
-P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest
