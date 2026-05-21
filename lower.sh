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

if [[ $# -lt 1 ]]; then
  echo "usage: $0 input.mlir" >&2
  exit 1
fi

TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel"
if [[ -n "${TILELOOM_TO_TTKERNEL_OPTIONS:-}" ]]; then
  TILELOOM_PASS_ARG="--loom-tileloom-to-ttkernel=${TILELOOM_TO_TTKERNEL_OPTIONS}"
fi

"${TILELOOM_TO_TTKERNEL_OPT}" "${TILELOOM_PASS_ARG}" "$1" -o ./tmp_output/kernel_ttkernel.mlir

"${PYTHON}" "${REPLACE_PY}" tmp_output/kernel_ttkernel.mlir

"${TTMLIR_OPT}" -canonicalize -cse -canonicalize --convert-ttkernel-to-emitc -canonicalize -cse -canonicalize -sccp -canonicalize tmp_output/kernel_ttkernel.mlir -o tmp_output/kernel_emitc.mlir

# Fold EmitC SSA temporaries (especially cast chains like i32<->ui32/ptrdiff_t)
# into expressions so generated C++ is substantially cleaner.
"${TTMLIR_OPT}" --form-expressions tmp_output/kernel_emitc.mlir -o tmp_output/kernel_emitc_formexpr.mlir
mv tmp_output/kernel_emitc_formexpr.mlir tmp_output/kernel_emitc.mlir

"${TILELOOM_TO_TTKERNEL_OPT}" --loom-post-emitc-host-signature tmp_output/kernel_emitc.mlir -o tmp_output/kernel_emitc_hostsig.mlir

"${TTMLIR_TRANSLATE}" --ttkernel-to-cpp tmp_output/kernel_emitc_hostsig.mlir -o tmp_output/kernel.cpp

"${PYTHON}" "${SPLIT_KERNEL_PY}" tmp_output/kernel.cpp
