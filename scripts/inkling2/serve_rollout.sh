#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
#
# Start the official Inkling 2 vLLM server on one GPU for training.drafter.traces_remote.

set -euo pipefail

# vllm/vllm-openai:inkling-2-mini as pulled on 2026-11-20: vllm 0.1.dev25855+g7cb5dae22.cu130,
# torch 2.14.0, Python 3.14.10, Model Runner V2 only.
readonly IMAGE="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
readonly DEFAULT_REPO="thinkingmachines/Inkling-2-Mini"
readonly DEFAULT_REVISION="1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
readonly VARLOOP_REPO="thinkingmachines/TML-d24-varloop-research"

usage() {
  cat <<EOF
usage: serve_rollout.sh [--dry-run] [--repo ORG/NAME] [--revision SHA] [--name NAME] [--hf-home DIR]
                        [--max-model-len N] [--wait SECONDS] [--hopper-patches DIR]
                        GPU PORT [vllm serve options]

Starts the OpenAI-compatible server of the official Inkling 2 vLLM image, pinned by digest, on
one GPU as the detached container inkling2-rollout-gpuGPU, listening on 127.0.0.1:PORT, and waits
until it answers /health. It needs a GPU of compute capability 9.0 or higher (H100 or newer):
the image's attention kernels use Hopper hardware features and fail on older cards.

  GPU               index of the one GPU to use (as nvidia-smi lists it); refused when it has any
                    compute process.
  PORT              local port of the server.
  further arguments go to vllm serve.

  --dry-run         print the docker command and exit.
  --repo            Hugging Face repository of the checkpoint (default ${DEFAULT_REPO}).
  --revision        its snapshot (default ${DEFAULT_REVISION}).
  --name            model name the server answers to, which the clients ask for: by default the registry name
                    of the repository (inkling-2-mini, or inkling-2-mini-varloop for ${VARLOOP_REPO});
                    required for any other repository. One checkpoint, one name: traces are saved under it.
  --hf-home         Hugging Face home that holds hub/models--ORG--NAME (default \$HF_HOME, else
                    ~/.cache/huggingface). The model's cache directory is mounted read-only.
  --max-model-len   prompt + output tokens of one request (default 16384, the trace budget).
  --wait            seconds to wait for the server (default 900).
  --hopper-patches DIR
                    needed on an H100: mount the three patched files of
                    serve_official.sh --make-hopper-patches DIR over the image's (its --help says why).

The container uses the host network but binds 127.0.0.1 only, is offline (Hugging Face and usage
statistics), and runs as the calling user with every capability dropped.

  follow   docker logs -f inkling2-rollout-gpuGPU
  stop     docker rm -f inkling2-rollout-gpuGPU

example (four GPUs, then the clients):
  for g in 0 1 2 3; do scripts/inkling2/serve_rollout.sh \$g \$((8100 + g)); done
  python -m training.drafter.traces_remote launch --out TRACES --model inkling-2-mini \\
      --prompts PROMPTS_V2 --servers http://127.0.0.1:8100 http://127.0.0.1:8101 \\
      http://127.0.0.1:8102 http://127.0.0.1:8103
EOF
}

fail() {
  echo "serve_rollout.sh: $1" >&2
  exit "${2:-2}"
}

dry_run=0
repo="${DEFAULT_REPO}"
revision="${DEFAULT_REVISION}"
served=""
hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
max_model_len=16384
wait_seconds=900
hopper=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --repo|--revision|--name|--hf-home|--max-model-len|--wait|--hopper-patches)
      [[ $# -ge 2 && -n "$2" ]] || fail "$1 needs a value"
      case "$1" in
        --repo) repo="$2" ;;
        --revision) revision="$2" ;;
        --name) served="$2" ;;
        --hf-home) hf_home="$2" ;;
        --max-model-len) max_model_len="$2" ;;
        --wait) wait_seconds="$2" ;;
        --hopper-patches) hopper="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) usage >&2; fail "unknown option $1" ;;
    *) break ;;
  esac
done
if [[ $# -lt 2 ]]; then
  usage >&2
  fail "GPU and PORT are required"
fi
gpu="$1"
port="$2"
shift 2

[[ "${gpu}" =~ ^[0-9]+$ ]] || fail "GPU must be one device index, got '${gpu}'"
[[ "${port}" =~ ^[0-9]+$ ]] || fail "PORT must be a number, got '${port}'"
[[ "${max_model_len}" =~ ^[0-9]+$ ]] || fail "--max-model-len must be a number"
[[ "${wait_seconds}" =~ ^[0-9]+$ ]] || fail "--wait must be a number of seconds"
[[ "${repo}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || fail "--repo must be ORG/NAME, got '${repo}'"
[[ "${revision}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--revision must be a snapshot name, got '${revision}'"
if [[ -z "${served}" ]]; then
  case "${repo}" in
    "${DEFAULT_REPO}") served="inkling-2-mini" ;;
    "${VARLOOP_REPO}") served="inkling-2-mini-varloop" ;;
    *) fail "--name is required for ${repo}: the model name its clients ask for" ;;
  esac
fi
[[ "${served}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--name must be a model name, got '${served}'"

# The files serve_official.sh --make-hopper-patches writes, and where the image keeps them.
readonly SITE="/usr/local/lib/python3.14/dist-packages"
readonly PATCHED=(vllm/models/inkling2/nvidia/ops/conv_mix.py
  vllm/model_executor/layers/fused_moe/moe_align_block_size.py
  vllm/_custom_ops.py)
if [[ -n "${hopper}" ]]; then
  hopper="$(realpath -m -- "${hopper}")"
  for file in "${PATCHED[@]}"; do
    [[ -f "${hopper}/$(basename "${file}")" ]] \
      || fail "${hopper}/$(basename "${file}") is missing (serve_official.sh --make-hopper-patches DIR)"
  done
fi

cache_name="models--${repo//\//--}"
hub="$(realpath -m -- "${hf_home}/hub")"
name="inkling2-rollout-gpu${gpu}"

command=(docker run --detach --init --pull missing
  --name "${name}"
  --gpus "device=${gpu}"
  --network host
  --cap-drop ALL
  --security-opt no-new-privileges
  --user "$(id -u):$(id -g)"
  --shm-size 8g
  --workdir /tmp
  -e HOME=/tmp
  -e USER=vllm
  -e LOGNAME=vllm
  -e HF_HOME=/hf
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1
  -e DO_NOT_TRACK=1
  -e CUDA_VISIBLE_DEVICES=0
  -v "${hub}/${cache_name}:/hf/hub/${cache_name}:ro")
# A cache that keeps large files in a shared blob directory links into it.
if [[ -d "${hub}/blobs" ]]; then
  command+=(-v "${hub}/blobs:/hf/hub/blobs:ro")
fi
if [[ -n "${hopper}" ]]; then
  for file in "${PATCHED[@]}"; do
    command+=(-v "${hopper}/$(basename "${file}"):${SITE}/${file}:ro")
  done
fi
command+=(--entrypoint python3 "${IMAGE}"
  -m vllm.entrypoints.cli.main serve "/hf/hub/${cache_name}/snapshots/${revision}"
  --served-model-name "${served}"
  --max-model-len "${max_model_len}"
  --host 127.0.0.1
  --port "${port}"
  "$@")

if [[ "${dry_run}" -eq 1 ]]; then
  printf '%q' "${command[0]}"
  printf ' %q' "${command[@]:1}"
  printf '\n'
  exit 0
fi

[[ -d "${hub}/${cache_name}/snapshots/${revision}" ]] \
  || fail "no snapshot ${revision} of ${repo} under ${hub} (set --hf-home or HF_HOME; download with: hf download ${repo} --revision ${revision})"

if ! compute_apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader)"; then
  fail "nvidia-smi could not query GPU ${gpu}; not starting" 3
fi
compute_apps="$(tr -s '[:space:]' ' ' <<<"${compute_apps}" | sed 's/^ //; s/ $//')"
if [[ -n "${compute_apps}" ]]; then
  fail "GPU ${gpu} is busy (compute processes: ${compute_apps}); not starting" 3
fi

"${command[@]}" >/dev/null
echo "${name}: started, waiting for http://127.0.0.1:${port}/health"
deadline=$((SECONDS + wait_seconds))
while (( SECONDS < deadline )); do
  if curl -sf -m 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "${name}: ready on http://127.0.0.1:${port}"
    exit 0
  fi
  if [[ -z "$(docker ps -q --filter "name=^${name}$")" ]]; then
    docker logs --tail 40 "${name}" >&2 || true
    fail "${name} exited during startup (log above; remove it with: docker rm ${name})" 1
  fi
  sleep 5
done
fail "${name} did not answer within ${wait_seconds} s (docker logs ${name})" 1
