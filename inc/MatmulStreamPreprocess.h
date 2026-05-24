/**
 * @file MatmulStreamPreprocess.h
 * @brief Pre-lowering matmul subblock and stream-output annotation helpers.
 */

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir::loom {

struct MatmulSubblockInfo {
  int64_t rt = 1;
  int64_t ct = 1;
};

std::optional<MatmulSubblockInfo>
chooseMatmulSubblock(int64_t rt, int64_t ct, int64_t capacity);

FailureOr<MatmulSubblockInfo> getMatmulSubblockAttrs(Operation *op);
void setMatmulSubblockAttrs(Operation *op, MatmulSubblockInfo subblock);

LogicalResult preprocessMatmulSubblocksAndStreams(
    ModuleOp module, bool enableStreamOutputSubblocks);

LogicalResult prepareMatmulStreamOutputLowering(ModuleOp module);

} // namespace mlir::loom
