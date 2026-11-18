#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VLLM_ATTENTION_BACKEND=FLASHINFER
if [ -z "${CUDA_HOME:-}" ]; then
  for OPSD_CUDA_CANDIDATE in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8; do
    if [ -x "$OPSD_CUDA_CANDIDATE/bin/nvcc" ]; then
      export CUDA_HOME="$OPSD_CUDA_CANDIDATE"
      break
    fi
  done
fi
if [ ! -x "${CUDA_HOME:-}/bin/nvcc" ]; then
  echo "CUDA toolkit not found; set CUDA_HOME to a toolkit containing bin/nvcc." >&2
  exit 1
fi
export PATH="$PWD/.env-vllm/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=12.0
OPSD_PYTHON_INCLUDE="$PWD/cache/python-dev/root/usr/include"
if [ -f "$OPSD_PYTHON_INCLUDE/python3.10/Python.h" ]; then
  export CPATH="$OPSD_PYTHON_INCLUDE/python3.10:$OPSD_PYTHON_INCLUDE${CPATH:+:$CPATH}"
fi
OPSD_LIBRARY_PATH="$PWD/cache/runtime-libs:$PWD/.env/lib/python3.10/site-packages/torch/lib"
for OPSD_LIBRARY_DIR in "$PWD"/.env/lib/python3.10/site-packages/nvidia/*/lib; do
  OPSD_LIBRARY_PATH="$OPSD_LIBRARY_PATH:$OPSD_LIBRARY_DIR"
done
export LD_LIBRARY_PATH="$OPSD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
