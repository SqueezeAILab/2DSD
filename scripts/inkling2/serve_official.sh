#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
#
# Run scripts/inkling2/native_baseline.py in the official Inkling 2 vLLM image on one GPU.

set -euo pipefail

# vllm/vllm-openai:inkling-2-mini as pulled on 2026-11-20: vllm 0.1.dev25855+g7cb5dae22.cu130,
# torch 2.14.0, Python 3.14.10, transformers 5.23.0, Model Runner V2 only.
readonly IMAGE="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
readonly DEFAULT_REPO="thinkingmachines/Inkling-2-Mini"
readonly DEFAULT_REVISION="1fb4134b9a74f9681054f4ef82e4bf1f458a2337"

usage() {
  cat <<EOF
usage: serve_official.sh [--dry-run] [--repo ORG/NAME] [--revision SHA] [--hf-home DIR]
                         [--cpuset CPUS] [--hopper-patches DIR]
                         GPU PROMPTS.jsonl OUT.jsonl [native_baseline.py options]
       serve_official.sh --make-hopper-patches DIR

Measures native AR or native DSpark decoding of Inkling 2 at batch size one with
scripts/inkling2/native_baseline.py inside the official vLLM image, pinned by digest:
  ${IMAGE}

  GPU            index of the one GPU to use (as nvidia-smi lists it). The run is refused when
                 that GPU has any compute process.
  PROMPTS.jsonl  prompts file of scripts/build_prompts.py; its directory is mounted read-only.
  OUT.jsonl      results journal; its directory is created and mounted writable. A rerun resumes.
  further arguments go to native_baseline.py, e.g. --method dspark --limit 64 (see its --help).

  --dry-run      print the docker command and exit: nothing is started and the GPU is not checked.
  --repo         Hugging Face repository of the checkpoint (default ${DEFAULT_REPO}).
  --revision     its snapshot (default ${DEFAULT_REVISION}).
  --hf-home      Hugging Face home that holds hub/models--ORG--NAME (default \$HF_HOME, else
                 ~/.cache/huggingface). Only that model's cache directory is mounted, read-only.
  --cpuset       CPUs the container may use, e.g. 24-47 (default: no restriction).
  --hopper-patches DIR
                 needed on an H100: the image's native kernels are built for SM10x, and three that the
                 Hopper path calls have no sm_90 code (inkling2_conv_mix, the alignment kernel of
                 moe_align_block_size and, for --method dspark, reshape_and_cache_flash). DIR holds the
                 image's conv_mix.py, moe_align_block_size.py and _custom_ops.py with
                 scripts/inkling2/hopper_patches applied, which send Hopper through the image's own Triton
                 kernels and a static-shape PyTorch alignment; they are mounted read-only over the
                 image's files and the run identity records it.
  --make-hopper-patches DIR
                 write those three files to DIR (copied out of the pinned image, then patched) and exit.

The container has no network (--network none, offline Hugging Face and no usage statistics), runs
as the calling user with every capability dropped and no-new-privileges, and sees only the model
cache, the prompts directory, this script directory (all read-only) and the output directory.

exit status: 2 for a usage error or a missing input and 3 when the GPU is busy or cannot be
queried, both before anything is started; otherwise that of docker run (native_baseline.py also
exits 2 on its own usage errors).

examples:
  serve_official.sh 3 prompts/inkling-2-mini/core.prompts.jsonl runs/mini-core-ar.jsonl --limit 64
  serve_official.sh 4 prompts/inkling-2-mini/core.prompts.jsonl runs/mini-core-dspark.jsonl \\
      --method dspark --limit 64
  serve_official.sh --repo thinkingmachines/TML-d24-varloop-research \\
      --revision 657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2 3 PROMPTS.jsonl OUT.jsonl
EOF
}

fail() {
  echo "serve_official.sh: $1" >&2
  exit "${2:-2}"
}

dry_run=0
repo="${DEFAULT_REPO}"
revision="${DEFAULT_REVISION}"
hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
cpuset=""
hopper=""
make_hopper=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --repo|--revision|--hf-home|--cpuset|--hopper-patches|--make-hopper-patches)
      [[ $# -ge 2 && -n "$2" ]] || fail "$1 needs a value"
      case "$1" in
        --repo) repo="$2" ;;
        --revision) revision="$2" ;;
        --hf-home) hf_home="$2" ;;
        --cpuset) cpuset="$2" ;;
        --hopper-patches) hopper="$2" ;;
        --make-hopper-patches) make_hopper="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) usage >&2; fail "unknown option $1" ;;
    *) break ;;
  esac
done
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SITE="/usr/local/lib/python3.14/dist-packages"
readonly PATCHED=(vllm/models/inkling2/nvidia/ops/conv_mix.py
  vllm/model_executor/layers/fused_moe/moe_align_block_size.py
  vllm/_custom_ops.py)

if [[ -n "${make_hopper}" ]]; then
  mkdir -p "${make_hopper}"
  container="$(docker create --pull never "${IMAGE}")"
  trap 'docker rm "${container}" > /dev/null' EXIT
  for file in "${PATCHED[@]}"; do
    name="$(basename "${file}")"
    docker cp "${container}:${SITE}/${file}" "${make_hopper}/${name}"
    patch --quiet "${make_hopper}/${name}" < "${script_dir}/hopper_patches/${name%.py}.patch"
  done
  echo "Hopper-patched files of ${IMAGE} are in ${make_hopper}"
  exit 0
fi

if [[ $# -lt 3 ]]; then
  usage >&2
  fail "GPU, PROMPTS.jsonl and OUT.jsonl are required"
fi
gpu="$1"
prompts="$2"
out="$3"
shift 3

[[ "${gpu}" =~ ^[0-9]+$ ]] || fail "GPU must be one device index, got '${gpu}'"
[[ "${repo}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || fail "--repo must be ORG/NAME, got '${repo}'"
[[ "${revision}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--revision must be a snapshot name, got '${revision}'"
[[ -f "${prompts}" ]] || fail "prompts file ${prompts} does not exist"

[[ -f "${script_dir}/native_baseline.py" ]] || fail "native_baseline.py is not next to this script"
image_label="${IMAGE}"
if [[ -n "${hopper}" ]]; then
  hopper="$(realpath -m -- "${hopper}")"
  for file in "${PATCHED[@]}"; do
    [[ -f "${hopper}/$(basename "${file}")" ]] \
      || fail "${hopper}/$(basename "${file}") is missing (serve_official.sh --make-hopper-patches DIR)"
  done
  image_label="${IMAGE} with the Hopper patches of scripts/inkling2/hopper_patches"
fi
prompts="$(realpath -e -- "${prompts}")"
out="$(realpath -m -- "${out}")"
cache_name="models--${repo//\//--}"
model_cache="$(realpath -m -- "${hf_home}/hub/${cache_name}")"

# The calling user has no passwd entry in the image: HOME, USER and LOGNAME are set as the image's
# own non-root entrypoint (docker/entrypoints/vllm-nonroot-entrypoint.sh) sets them, so the caches
# under ~/.cache are writable and getpass.getuser() does not look the UID up.
command=(docker run --rm --init --pull never
  --name "inkling2-native-gpu${gpu}"
  --gpus "device=${gpu}"
  --network none
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
  -e OMP_NUM_THREADS=4
  -e MKL_NUM_THREADS=4
  -e TOKENIZERS_PARALLELISM=false
  -e PYTHONDONTWRITEBYTECODE=1
  -e "INKLING2_IMAGE=${image_label}"
  -v "${model_cache}:/hf/hub/${cache_name}:ro"
  -v "$(dirname "${prompts}"):/prompts:ro"
  -v "${script_dir}:/opt/inkling2:ro"
  -v "$(dirname "${out}"):/out")
if [[ -n "${hopper}" ]]; then
  for file in "${PATCHED[@]}"; do
    command+=(-v "${hopper}/$(basename "${file}"):${SITE}/${file}:ro")
  done
fi
if [[ -n "${cpuset}" ]]; then
  command+=(--cpuset-cpus "${cpuset}")
fi
command+=(--entrypoint python3 "${IMAGE}"
  /opt/inkling2/native_baseline.py
  --model "/hf/hub/${cache_name}/snapshots/${revision}"
  --prompts "/prompts/$(basename "${prompts}")"
  --out "/out/$(basename "${out}")"
  "$@")

if [[ "${dry_run}" -eq 1 ]]; then
  printf '%q' "${command[0]}"
  printf ' %q' "${command[@]:1}"
  printf '\n'
  exit 0
fi

[[ -d "${model_cache}/snapshots/${revision}" ]] \
  || fail "no snapshot ${revision} of ${repo} under ${hf_home}/hub (set --hf-home or HF_HOME)"

# One measuring process per GPU: never start next to another compute process.
if ! compute_apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader)"; then
  fail "nvidia-smi could not query GPU ${gpu}; not starting" 3
fi
compute_apps="$(tr -s '[:space:]' ' ' <<<"${compute_apps}" | sed 's/^ //; s/ $//')"
if [[ -n "${compute_apps}" ]]; then
  fail "GPU ${gpu} is busy (compute processes: ${compute_apps}); not starting" 3
fi

mkdir -p "$(dirname "${out}")"
exec "${command[@]}"
