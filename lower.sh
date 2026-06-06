#!/usr/bin/env bash
set -euo pipefail

mkdir -p tmp_output
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOM2TTKERNEL_BUILD_DIR="${LOOM2TTKERNEL_BUILD_DIR:-${SCRIPT_DIR}/build}"
CMAKE_CACHE="${LOOM2TTKERNEL_BUILD_DIR}/CMakeCache.txt"

if [[ -z "${TTMLIR_BUILD_DIR:-}" && -f "${CMAKE_CACHE}" ]]; then
  TTMLIR_BUILD_DIR="$(sed -n 's/^TTMLIR_BUILD_DIR:PATH=//p' "${CMAKE_CACHE}" | tail -n 1)"
fi

if [[ -z "${TTMLIR_BUILD_DIR:-}" ]]; then
  echo "error: TTMLIR_BUILD_DIR is not set and was not found in ${CMAKE_CACHE}" >&2
  echo "hint: run ./build.sh first or export TTMLIR_BUILD_DIR=/path/to/tt-mlir/build" >&2
  exit 1
fi

TILELOOM_TO_TTKERNEL_OPT="${TILELOOM_TO_TTKERNEL_OPT:-${LOOM2TTKERNEL_BUILD_DIR}/bin/tileloom_to_ttkernel_opt}"
TTMLIR_OPT="${TTMLIR_OPT:-${TTMLIR_BUILD_DIR}/bin/ttmlir-opt}"
TTMLIR_TRANSLATE="${TTMLIR_TRANSLATE:-${TTMLIR_BUILD_DIR}/bin/ttmlir-translate}"
PYTHON="${PYTHON:-python3}"
REPLACE_PY="${REPLACE_PY:-${SCRIPT_DIR}/replace.py}"
SPLIT_KERNEL_PY="${SPLIT_KERNEL_PY:-${SCRIPT_DIR}/split_kernel.py}"

usage() {
  echo "usage: $0 input.mlir [func_index]" >&2
  echo "       $0 func_index" >&2
  echo "" >&2
  echo "func_index is 1-based. With only func_index, input defaults to" >&2
  echo "LOWER_INPUT or ${SCRIPT_DIR}/../../test/matmul_2Dmesh/IRs/p03_bufferized.mlir." >&2
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

INPUT_MLIR=""
FUNC_INDEX=""
DEFAULT_INPUT="${LOWER_INPUT:-${SCRIPT_DIR}/../../test/matmul_2Dmesh/IRs/p03_bufferized.mlir}"

if [[ $# -eq 1 ]]; then
  if is_positive_int "$1"; then
    FUNC_INDEX="$1"
    INPUT_MLIR="${DEFAULT_INPUT}"
  else
    INPUT_MLIR="$1"
  fi
else
  if is_positive_int "$1" && ! is_positive_int "$2"; then
    FUNC_INDEX="$1"
    INPUT_MLIR="$2"
  elif ! is_positive_int "$1" && is_positive_int "$2"; then
    INPUT_MLIR="$1"
    FUNC_INDEX="$2"
  else
    usage
    exit 1
  fi
fi

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input MLIR not found: ${INPUT_MLIR}" >&2
  exit 1
fi

TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel"
if [[ -n "${TILELOOM_TO_TTKERNEL_OPTIONS:-}" ]]; then
  TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel=${TILELOOM_TO_TTKERNEL_OPTIONS}"
fi

"${TILELOOM_TO_TTKERNEL_OPT}" "${TILELOOM_PASS_ARG}" "${INPUT_MLIR}" -o ./tmp_output/kernel_ttkernel.mlir

"${PYTHON}" "${REPLACE_PY}" tmp_output/kernel_ttkernel.mlir

"${TTMLIR_OPT}" -canonicalize -cse -canonicalize --convert-ttkernel-to-emitc -canonicalize -cse -canonicalize -sccp -canonicalize tmp_output/kernel_ttkernel.mlir -o tmp_output/kernel_emitc.mlir

# Fold EmitC SSA temporaries (especially cast chains like i32<->ui32/ptrdiff_t)
# into expressions so generated C++ is substantially cleaner.
"${TTMLIR_OPT}" --form-expressions tmp_output/kernel_emitc.mlir -o tmp_output/kernel_emitc_formexpr.mlir
mv tmp_output/kernel_emitc_formexpr.mlir tmp_output/kernel_emitc.mlir

"${TILELOOM_TO_TTKERNEL_OPT}" --loom-post-emitc-host-signature tmp_output/kernel_emitc.mlir -o tmp_output/kernel_emitc_hostsig.mlir

"${TTMLIR_TRANSLATE}" --ttkernel-to-cpp tmp_output/kernel_emitc_hostsig.mlir -o tmp_output/kernel.cpp

SPLIT_KERNEL_ARGS=(tmp_output/kernel.cpp)
if [[ -n "${SPLIT_KERNEL_OUTPUT_DIR:-}" ]]; then
  SPLIT_KERNEL_ARGS+=(--output-dir "${SPLIT_KERNEL_OUTPUT_DIR}")
fi
if [[ -n "${FUNC_INDEX}" ]]; then
  SPLIT_KERNEL_ARGS+=(--func-index "${FUNC_INDEX}")
fi

"${PYTHON}" "${SPLIT_KERNEL_PY}" "${SPLIT_KERNEL_ARGS[@]}"
