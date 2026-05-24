/**
 * @file MatmulStreamPreprocess.cpp
 * @brief Pre-lowering matmul subblock and stream-output annotation helpers.
 */

#include "MatmulStreamPreprocess.h"
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <array>
#include <tuple>

#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomInterfaces.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;

namespace {

struct MatmulStreamCandidate {
  Operation *op = nullptr;
  Value out;
};

std::optional<MatmulStreamCandidate> getBatchMatmulCandidate(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
    if (linalgOp.getOutputs().size() != 1)
      return std::nullopt;
    return MatmulStreamCandidate{op, linalgOp.getOutputs()[0]};
  }

  if (auto loomOp = dyn_cast<::loom::BatchMatmulOp>(op))
    return MatmulStreamCandidate{op, loomOp.getOuts()};

  return std::nullopt;
}

std::optional<std::pair<int64_t, int64_t>>
getOutputTilePlane(Value out) {
  auto shaped = dyn_cast<ShapedType>(out.getType());
  if (!shaped || !shaped.hasStaticShape() || shaped.getRank() < 2)
    return std::nullopt;

  ArrayRef<int64_t> shape = shaped.getShape();
  auto rt = ceilDiv32(shape[shape.size() - 2]);
  auto ct = ceilDiv32(shape[shape.size() - 1]);
  if (!rt || !ct)
    return std::nullopt;
  return std::make_pair(*rt, *ct);
}

bool hasBatchSizeOne(Value out) {
  auto shaped = dyn_cast<ShapedType>(out.getType());
  if (!shaped || !shaped.hasStaticShape())
    return false;
  if (shaped.getRank() < 3)
    return true;
  return shaped.getShape().front() == 1;
}

static bool isLocalL1Copy(::loom::CopyOp op) {
  return !isDramToL1Copy(op) && !isL1ToDramCopy(op);
}

FailureOr<::loom::CopyOp> findSingleLocalBridge(func::FuncOp func,
                                                Operation *matmulOp,
                                                Value matmulOut) {
  Value outBase = stripViewLikeWrappers(matmulOut);
  SmallVector<::loom::CopyOp, 2> bridgeCopies;
  func.walk([&](::loom::CopyOp copyOp) {
    if (!isLocalL1Copy(copyOp))
      return;
    if (stripViewLikeWrappers(copyOp.getSource()) == outBase)
      bridgeCopies.push_back(copyOp);
  });

  if (bridgeCopies.size() != 1) {
    return matmulOp->emitOpError()
           << "expected exactly one local L1 bridge copy for streamed matmul, "
           << "found " << bridgeCopies.size();
  }
  return bridgeCopies.front();
}

FailureOr<::loom::CopyOp> findSingleStoreFromBridge(func::FuncOp func,
                                                    Operation *matmulOp,
                                                    ::loom::CopyOp bridgeCopy) {
  Value bridgeDst = stripViewLikeWrappers(bridgeCopy.getDestination());
  SmallVector<::loom::CopyOp, 2> stores;
  func.walk([&](::loom::CopyOp copyOp) {
    if (!isL1ToDramCopy(copyOp))
      return;
    if (stripViewLikeWrappers(copyOp.getSource()) == bridgeDst)
      stores.push_back(copyOp);
  });

  if (stores.size() != 1) {
    return matmulOp->emitOpError()
           << "expected exactly one L1-to-DRAM store for streamed matmul "
           << "bridge, found " << stores.size();
  }
  return stores.front();
}

void annotateStreamParticipant(Operation *op, MatmulSubblockInfo subblock,
                               StringRef streamAttrName) {
  setMatmulSubblockAttrs(op, subblock);
  OpBuilder builder(op->getContext());
  op->setAttr(streamAttrName, builder.getUnitAttr());
}

Value getLoomMatmulOutput(Operation *op) {
  if (auto matmul = dyn_cast<::loom::MatmulOp>(op))
    return matmul.getOuts();
  if (auto matmul = dyn_cast<::loom::BatchMatmulOp>(op))
    return matmul.getOuts();
  return {};
}

Value getStreamMatmulOutput(Operation *op) {
  if (auto matmul = dyn_cast<linalg::BatchMatmulOp>(op)) {
    if (matmul.getOutputs().size() == 1)
      return matmul.getOutputs()[0];
    return {};
  }
  return getLoomMatmulOutput(op);
}

bool setLoomMatmulOutput(Operation *op, Value newOutput) {
  if (isa<::loom::MatmulOp, ::loom::BatchMatmulOp>(op)) {
    op->setOperand(2, newOutput);
    return true;
  }
  return false;
}

static bool attrContains(Attribute attr, StringRef needle) {
  if (!attr)
    return false;
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  os.flush();
  return StringRef(text).contains(needle);
}

std::optional<scf::ForOp> getStaticSequentialKLoop(Operation *op) {
  for (scf::ForOp loop = op->getParentOfType<scf::ForOp>(); loop;
       loop = loop->getParentOfType<scf::ForOp>()) {
    if (!attrContains(loop->getAttr("loom.iter_type"), "sequential"))
      continue;

    std::optional<int64_t> lower = evaluateConstInt(loop.getLowerBound());
    std::optional<int64_t> upper = evaluateConstInt(loop.getUpperBound());
    std::optional<int64_t> step = evaluateConstInt(loop.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper - *lower <= 1)
      continue;
    return loop;
  }
  return std::nullopt;
}

void moveBridgeDestinationToDominate(Operation *anchor,
                                     ::loom::CopyOp bridgeCopy) {
  auto semTake =
      bridgeCopy.getDestination().getDefiningOp<::loom::SemaphoreTakeOp>();
  if (!semTake)
    return;

  if (Operation *sourceDef = semTake.getSource().getDefiningOp()) {
    if (sourceDef->getBlock() == anchor->getBlock() &&
        anchor->isBeforeInBlock(sourceDef))
      sourceDef->moveBefore(anchor);
  }
  if (semTake->getBlock() == anchor->getBlock() &&
      anchor->isBeforeInBlock(semTake))
    semTake->moveBefore(anchor);
}

MemRefType getBatch1Type(MemRefType srcTy) {
  int64_t dim0 =
      srcTy.isDynamicDim(0) ? ShapedType::kDynamic : srcTy.getDimSize(0);
  int64_t dim1 =
      srcTy.isDynamicDim(1) ? ShapedType::kDynamic : srcTy.getDimSize(1);
  return MemRefType::get({1, dim0, dim1}, srcTy.getElementType(), AffineMap(),
                         srcTy.getMemorySpace());
}

FailureOr<Value> makeOutputTypeCompatible(Operation *matmulOp, Value bridgeDst,
                                          Type targetType) {
  if (bridgeDst.getType() == targetType)
    return bridgeDst;

  auto dstTy = dyn_cast<MemRefType>(bridgeDst.getType());
  auto targetMemrefTy = dyn_cast<MemRefType>(targetType);
  if (!dstTy || !targetMemrefTy || dstTy.getRank() != 2 ||
      targetMemrefTy.getRank() != 3) {
    return matmulOp->emitError()
           << "streamed loom matmul bridge destination is not type-compatible "
           << "with matmul output";
  }

  MemRefType batchDstTy = getBatch1Type(dstTy);
  if (batchDstTy != targetMemrefTy) {
    return matmulOp->emitError()
           << "streamed loom matmul batch-1 bridge type does not match output";
  }

  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
  OpBuilder builder(matmulOp);
  return builder
      .create<memref::ExpandShapeOp>(matmulOp->getLoc(), batchDstTy, bridgeDst,
                                     reassociation)
      .getResult();
}

} // namespace

std::optional<MatmulSubblockInfo>
mlir::loom::chooseMatmulSubblock(int64_t rt, int64_t ct, int64_t capacity) {
  if (rt <= 0 || ct <= 0 || capacity <= 0)
    return std::nullopt;

  static constexpr std::array<std::tuple<int64_t, int64_t>, 20>
      subblockHWChoices = {{
          {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
          {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
          {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
      }};

  for (auto subblockHW : subblockHWChoices) {
    int64_t outSubblockW = std::get<0>(subblockHW);
    int64_t outSubblockH = std::get<1>(subblockHW);
    int64_t area = outSubblockH * outSubblockW;
    if (area > capacity)
      continue;
    if (rt % outSubblockH == 0 && ct % outSubblockW == 0)
      return MatmulSubblockInfo{outSubblockH, outSubblockW};
  }

  return std::nullopt;
}

FailureOr<MatmulSubblockInfo>
mlir::loom::getMatmulSubblockAttrs(Operation *op) {
  if (!op)
    return failure();
  auto rowsAttr = op->getAttrOfType<IntegerAttr>(kMatmulSubblockRowsAttrName);
  auto colsAttr = op->getAttrOfType<IntegerAttr>(kMatmulSubblockColsAttrName);
  if (!rowsAttr || !colsAttr || rowsAttr.getInt() <= 0 ||
      colsAttr.getInt() <= 0)
    return failure();
  return MatmulSubblockInfo{rowsAttr.getInt(), colsAttr.getInt()};
}

void mlir::loom::setMatmulSubblockAttrs(Operation *op,
                                        MatmulSubblockInfo subblock) {
  OpBuilder builder(op->getContext());
  op->setAttr(kMatmulSubblockRowsAttrName,
              builder.getI64IntegerAttr(subblock.rt));
  op->setAttr(kMatmulSubblockColsAttrName,
              builder.getI64IntegerAttr(subblock.ct));
}

LogicalResult mlir::loom::preprocessMatmulSubblocksAndStreams(
    ModuleOp module, bool enableStreamOutputSubblocks) {
  LogicalResult status = success();

  module.walk([&](func::FuncOp func) {
    if (failed(status))
      return;

    SmallVector<Operation *, 4> matmuls;
    func.walk([&](Operation *op) {
      if (getBatchMatmulCandidate(op))
        matmuls.push_back(op);
    });

    for (Operation *op : matmuls) {
      if (failed(status))
        return;

      std::optional<MatmulStreamCandidate> candidate =
          getBatchMatmulCandidate(op);
      if (!candidate)
        continue;

      auto tilePlane = getOutputTilePlane(candidate->out);
      if (!tilePlane) {
        status = op->emitOpError()
                 << "failed to derive static output tile plane for matmul";
        return;
      }

      std::optional<MatmulSubblockInfo> subblock =
          chooseMatmulSubblock(tilePlane->first, tilePlane->second, 8);
      if (!subblock) {
        status = op->emitOpError()
                 << "failed to choose matmul subblock for output tile plane "
                 << tilePlane->first << "x" << tilePlane->second;
        return;
      }
      setMatmulSubblockAttrs(op, *subblock);

      if (!enableStreamOutputSubblocks)
        continue;

      if (!hasBatchSizeOne(candidate->out)) {
        status = op->emitOpError()
                 << "streamed matmul output only supports batch size 1";
        return;
      }

      FailureOr<::loom::CopyOp> bridgeCopy =
          findSingleLocalBridge(func, op, candidate->out);
      if (failed(bridgeCopy)) {
        status = failure();
        return;
      }

      FailureOr<::loom::CopyOp> store =
          findSingleStoreFromBridge(func, op, *bridgeCopy);
      if (failed(store)) {
        status = failure();
        return;
      }

      annotateStreamParticipant(op, *subblock,
                                kMatmulStreamOutputSubblocksAttrName);
      annotateStreamParticipant(bridgeCopy->getOperation(), *subblock,
                                kMatmulStreamBridgeAttrName);
      annotateStreamParticipant(store->getOperation(), *subblock,
                                kMatmulStreamStoreAttrName);
      if (isa<linalg::BatchMatmulOp>(op) && getStaticSequentialKLoop(op)) {
        OpBuilder builder(op->getContext());
        op->setAttr(kMatmulStreamFinalPackAttrName, builder.getUnitAttr());
        bridgeCopy->getOperation()->setAttr(kMatmulStreamFinalPackAttrName,
                                            builder.getUnitAttr());
      }
    }
  });

  return status;
}

LogicalResult mlir::loom::prepareMatmulStreamOutputLowering(ModuleOp module) {
  LogicalResult status = success();
  SmallVector<func::FuncOp, 8> funcs;
  module.walk([&](func::FuncOp func) { funcs.push_back(func); });
  for (func::FuncOp func : funcs) {
    if (failed(status) || !func.getName().ends_with("__compute"))
      continue;

    SmallVector<::loom::CopyOp, 4> bridgeCopies;
    func.walk([&](::loom::CopyOp copyOp) {
      if (copyOp->hasAttr(kMatmulStreamBridgeAttrName))
        bridgeCopies.push_back(copyOp);
    });

    for (::loom::CopyOp bridgeCopy : bridgeCopies) {
      if (failed(status))
        break;

      Value bridgeSrc = stripViewLikeWrappers(bridgeCopy.getSource());
      Operation *matmulOp = nullptr;
      func.walk([&](Operation *op) {
        if (matmulOp ||
            !isa<linalg::BatchMatmulOp, ::loom::MatmulOp,
                 ::loom::BatchMatmulOp>(op) ||
            !op->hasAttr(kMatmulStreamOutputSubblocksAttrName))
          return;
        if (stripViewLikeWrappers(getStreamMatmulOutput(op)) == bridgeSrc)
          matmulOp = op;
      });

      if (!matmulOp)
        continue;

      Operation *anchor = matmulOp;
      if (auto kLoop = getStaticSequentialKLoop(matmulOp))
        anchor = kLoop->getOperation();
      moveBridgeDestinationToDominate(anchor, bridgeCopy);

      if (isa<linalg::BatchMatmulOp>(matmulOp)) {
        if (matmulOp->hasAttr(kMatmulStreamFinalPackAttrName))
          bridgeCopy->setAttr(kMatmulStreamFinalPackAttrName,
                              OpBuilder(matmulOp->getContext()).getUnitAttr());
        continue;
      }

      FailureOr<Value> newOutput = makeOutputTypeCompatible(
          matmulOp, bridgeCopy.getDestination(),
          getLoomMatmulOutput(matmulOp).getType());
      if (failed(newOutput)) {
        status = failure();
        break;
      }
      if (!setLoomMatmulOutput(matmulOp, *newOutput)) {
        status = matmulOp->emitError()
                 << "failed to retarget streamed loom matmul output";
        break;
      }

      SmallVector<::loom::SemaphoreGiveOp, 4> givesToErase;
      func.walk([&](::loom::SemaphoreGiveOp giveOp) {
        if (stripViewLikeWrappers(giveOp.getSource()) == bridgeSrc)
          givesToErase.push_back(giveOp);
      });
      for (::loom::SemaphoreGiveOp giveOp : llvm::reverse(givesToErase))
        giveOp.erase();
      bridgeCopy.erase();
    }
  }
  return status;
}
