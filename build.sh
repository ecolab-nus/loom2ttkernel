#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -G Ninja "$@"
cmake --build "${SCRIPT_DIR}/build" --target tileloom_to_ttkernel_opt
