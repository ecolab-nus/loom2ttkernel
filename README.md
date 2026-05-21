# Loom2TTKernel

Compiler stack: **Loom** (bufferized dataflow MLIR) → **TTKernel** (`tileloom_to_ttkernel_opt`) → **tt-mlir** (EmitC + `ttkernel-to-cpp`) → **tt-metal**-style C++ (`kernel.cpp`, then `split_kernel.py`).

**Build:** `./build.sh` — CMake+Ninja into `build/`, builds `tileloom_to_ttkernel_opt`.

**Codegen:** `./lower.sh <bufferized.mlir>` — requires a prior build and a tt-mlir build; `TTMLIR_BUILD_DIR` is taken from `build/CMakeCache.txt` when set there, else export it (see `lower.sh`). Input must be Loom-produced bufferized MLIR, e.g. `loom/test/matmul/wormhole/M256_N256_K256/IRs/p03_bufferized.mlir`.
