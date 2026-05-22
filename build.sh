#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

cmake -S . -B build -G Ninja \
  -DLOOM_SOURCE_DIR=../loom-dataflow \
  -DLOOM_BUILD_DIR=../loom-dataflow/build \
  "$@"
cmake --build "${SCRIPT_DIR}/build" --target tileloom_to_ttkernel_opt
