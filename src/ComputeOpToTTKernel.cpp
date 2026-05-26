/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Conversion patterns for compute ops to TTKernel.
 */

#include "ComputeOpToTTKernel.h"
#include "FuncOpToTTKernel.h"
#include "MatmulStreamPreprocess.h"
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

// Loom dialect headers for Loom ops used in compute lowering.
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomInterfaces.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using mlir::loom::CompileArgTracker;

namespace {

enum class FlashAttentionGenericKind {
  Reduction,
  FusedMulBcastAdd,
  InplaceMulBcastCols,
  AddInplace,
  SimpleBinaryTile,
  Elementwise
};

/// Elementwise broadcast categories accepted by the generic lowering.
enum class ElementwiseBroadcastKind {
  /// Broadcast over the last tensor dimension.
  RowBroadcast,
  /// Broadcast over the second-last tensor dimension.
  ColBroadcast,
  /// Broadcast over a dimension other than the last two.
  BatchBroadcast
};

/// Metadata describing one elementwise broadcast input.
struct ElementwiseBroadcastInfo {
  /// Input operand index in the generic op.
  unsigned inputIdx = 0;
  /// True when input indexing map is scalar (`()->`), i.e. broadcast all tiles.
  bool isScalar = false;
  /// Output-loop dimension that is broadcast (dropped in input indexing map).
  unsigned droppedDim = 0;
  /// Normalized broadcast category.
  ElementwiseBroadcastKind kind = ElementwiseBroadcastKind::BatchBroadcast;
  /// Tile-count extent for the dropped dimension.
  int64_t droppedDimTiles = 1;
  /// Product of tile-count extents after @p droppedDim.
  int64_t suffixTiles = 1;
};

struct ElementwiseAnalysis {
  Value yieldValue;
  int64_t outTiles = 0;
  SmallVector<std::optional<ElementwiseBroadcastInfo>, 4> broadcastsByInput;
  SmallVector<int64_t, 4> inputWaitTiles;
  llvm::SmallBitVector usedInputs;

  bool needsBinopWithScalar = false;
  bool needsSubBinary = false;
  bool needsAddBinary = false;
  bool needsMulBinary = false;
  bool needsBinaryMax = false;
  bool needsPowBinary = false;
  bool needsRecip = false;
  bool needsLog = false;
  bool needsExp = false;
};

static std::optional<int64_t> getAnnotatedVecTilesFromInput(Value value) {
  Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      if (auto tilesAttr = sem->getAttrOfType<IntegerAttr>(kVecTilesAttrName)) {
        int64_t tiles = tilesAttr.getInt();
        if (tiles > 0)
          return tiles;
      }
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return std::nullopt;
}

static Value stripElementwiseInputWrappers(Value value) {
  Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t>
getScalarSiteIdForGenericInput(linalg::GenericOp op, unsigned inputIdx) {
  if (inputIdx >= op.getNumDpsInputs())
    return std::nullopt;

  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return std::nullopt;

  Value inputRoot = stripElementwiseInputWrappers(op.getDpsInputs()[inputIdx]);
  if (auto blockArg = dyn_cast<BlockArgument>(inputRoot)) {
    if (blockArg.getOwner() == &parentFunc.front()) {
      if (auto siteAttr = dyn_cast_or_null<IntegerAttr>(parentFunc.getArgAttr(
              blockArg.getArgNumber(), kScalarSiteIdAttrName)))
        return siteAttr.getInt();
    }
  }

  std::optional<int64_t> siteId;
  parentFunc.walk([&](::loom::CopyOp copyOp) {
    if (siteId)
      return;
    auto siteAttr = copyOp->getAttrOfType<IntegerAttr>(kScalarSiteIdAttrName);
    if (!siteAttr)
      return;
    Value destinationRoot = stripElementwiseInputWrappers(copyOp.getDestination());
    if (destinationRoot != inputRoot)
      return;
    siteId = siteAttr.getInt();
  });
  return siteId;
}

struct ScalarRuntimeListDimUse {
  Value iv;
  int64_t dimOrdinal = 0;
  int64_t lb = 0;
  int64_t step = 1;
  int64_t extent = 1;
};

static SmallVector<ScalarRuntimeListDimUse, 4>
collectScalarRuntimeListDimsForSite(Operation *anchor, int64_t siteId) {
  SmallVector<ScalarRuntimeListDimUse, 4> dims;
  for (Operation *parent = anchor ? anchor->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp)
      continue;

    auto attr =
        forOp->getAttrOfType<DenseI64ArrayAttr>(kScalarSiteListDimsAttrName);
    if (!attr)
      continue;

    ArrayRef<int64_t> values = attr.asArrayRef();
    for (size_t idx = 0; idx + 4 < values.size(); idx += 5) {
      if (values[idx] != siteId)
        continue;
      dims.push_back(ScalarRuntimeListDimUse{
          forOp.getInductionVar(), values[idx + 1], values[idx + 2],
          values[idx + 3], values[idx + 4]});
    }
  }

  std::stable_sort(dims.begin(), dims.end(),
                   [](const ScalarRuntimeListDimUse &a,
                      const ScalarRuntimeListDimUse &b) {
                     return a.dimOrdinal < b.dimOrdinal;
                   });
  return dims;
}

static FailureOr<Value>
buildScalarRuntimeListIndex(linalg::GenericOp op, int64_t siteId,
                            ConversionPatternRewriter &rewriter,
                            Location loc) {
  SmallVector<ScalarRuntimeListDimUse, 4> dims =
      collectScalarRuntimeListDimsForSite(op.getOperation(), siteId);
  if (dims.empty())
    return op.emitOpError()
           << "missing scalar runtime list dim metadata for site " << siteId;

  Value index = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  for (const ScalarRuntimeListDimUse &dim : dims) {
    if (dim.step <= 0 || dim.extent <= 0)
      return op.emitOpError()
             << "invalid scalar runtime list dim metadata for site " << siteId;

    Value iv = toI32(rewriter, loc, dim.iv);
    if (!iv)
      return op.emitOpError()
             << "failed to convert scalar runtime list IV to i32 for site "
             << siteId;

    Value lb = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(),
                                                     dim.lb);
    Value step = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), dim.step);
    Value extent = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), dim.extent);

    Value shifted =
        rewriter.create<arith::SubIOp>(loc, iv, lb);
    Value digit =
        rewriter.create<arith::DivSIOp>(loc, shifted, step);
    Value scaled =
        rewriter.create<arith::MulIOp>(loc, index, extent);
    index = rewriter.create<arith::AddIOp>(loc, scaled, digit);
  }

  return index;
}

static Value selectScalarRuntimeListElement(ConversionPatternRewriter &rewriter,
                                            Location loc, Value listIndex,
                                            ArrayRef<Value> scalarArgs) {
  if (scalarArgs.empty())
    return {};
  if (scalarArgs.size() == 1)
    return scalarArgs.front();

  Value selected = scalarArgs.front();
  for (size_t elementIndex = 1; elementIndex < scalarArgs.size(); ++elementIndex) {
    Value indexValue = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), static_cast<int64_t>(elementIndex));
    Value matches = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, listIndex, indexValue);
    selected = rewriter.create<arith::SelectOp>(
        loc, matches, scalarArgs[elementIndex], selected);
  }
  return selected;
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis);
static bool isSupportedElementwiseGeneric(linalg::GenericOp op) {
  ElementwiseAnalysis analysis;
  return succeeded(analyzeElementwiseGeneric(op, analysis));
}

static bool isSupportedFusedMulBcastAddGeneric(linalg::GenericOp op);
static bool isSupportedInplaceMulBcastColsGeneric(linalg::GenericOp op);
static bool isSupportedAddInplaceGeneric(linalg::GenericOp op);
static bool isSupportedSimpleBinaryTileGeneric(linalg::GenericOp op);

template <typename OpTy>
static bool bodyHasOp(linalg::GenericOp op) {
  for (Operation &inner : op.getRegion().front().without_terminator())
    if (isa<OpTy>(inner))
      return true;
  return false;
}

static std::optional<FlashAttentionGenericKind>
classifyFlashAttentionGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return std::nullopt;

  if (llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::reduction;
      }))
    return FlashAttentionGenericKind::Reduction;

  if (isSupportedFusedMulBcastAddGeneric(op))
    return FlashAttentionGenericKind::FusedMulBcastAdd;

  if (isSupportedInplaceMulBcastColsGeneric(op))
    return FlashAttentionGenericKind::InplaceMulBcastCols;

  if (isSupportedAddInplaceGeneric(op))
    return FlashAttentionGenericKind::AddInplace;

  if (isSupportedSimpleBinaryTileGeneric(op))
    return FlashAttentionGenericKind::SimpleBinaryTile;

  if (isSupportedElementwiseGeneric(op))
    return FlashAttentionGenericKind::Elementwise;

  return std::nullopt;
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          llvm::DenseMap<Value, int64_t> &waitState);

enum class TileReductionCombineOp {
  Sum,
  Max
};

struct TileReductionAnalysis {
  TileReductionCombineOp combineOp = TileReductionCombineOp::Sum;
};

/**
 * @brief Distinguishes tile-generic body block arguments by semantic role.
 */
enum class TileGenericBodyOperand {
  Input,
  Accumulator
};

static LogicalResult analyzeTileReductionGeneric(
    linalg::GenericOp op, TileReductionAnalysis &analysis);

/**
 * @brief Check for the [reduction, parallel, parallel] iterator pattern.
 */
static bool isTileGenericOp(linalg::GenericOp op) {
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() < 3)
    return false;
  if (iteratorTypes[0] != utils::IteratorType::reduction)
    return false;
  for (utils::IteratorType type : llvm::drop_begin(iteratorTypes))
    if (type != utils::IteratorType::parallel)
      return false;
  return true;
}

/**
 * @brief Entry-point for [reduction, parallel, parallel] generic lowering.
 *
 * @details Keeps the partial accumulator in DST registers for the complete
 *          reduction loop. Only the final accumulated tile is packed to the
 *          output CB, avoiding repeated BF16 round-trips through CB storage.
 */
static LogicalResult tileGenericOp(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
    return failure();

  Value inCb = adaptor.getInputs().front();
  Value outCb = adaptor.getOutputs().front();
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
    return op.emitOpError()
           << "tile generic lowering requires CB-typed input and output";

  TileReductionAnalysis analysis;
  if (failed(analyzeTileReductionGeneric(op, analysis)))
    return op.emitOpError() << "unsupported tile reduction body";

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return failure();

  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  if (inShape.size() != outShape.size() + 1 || inShape[0] <= 0)
    return failure();
  for (size_t dim = 0; dim < outShape.size(); ++dim) {
    if (inShape[dim + 1] != outShape[dim])
      return failure();
  }

  int64_t outTiles = 1;
  unsigned outRank = outShape.size();
  unsigned tiledStartDim = outRank > 2 ? outRank - 2 : 0;
  for (unsigned dim = 0; dim < outRank; ++dim) {
    int64_t dimSize = outShape[dim];
    if (dimSize <= 0)
      return failure();

    if (dim >= tiledStartDim) {
      auto dimTiles = ceilDiv32(dimSize);
      if (!dimTiles || *dimTiles <= 0)
        return failure();
      outTiles *= *dimTiles;
    } else {
      outTiles *= dimSize;
    }
  }
  int64_t reduceSlices = inShape[0];
  int64_t inputTiles = reduceSlices * outTiles;

  Location loc = op.getLoc();
  auto i32 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
  };
  Value zeroI32 = i32(0);
  Value oneI32 = i32(1);
  Value outTilesV = i32(outTiles);
  Value reduceSlicesV = i32(reduceSlices);

  auto emitWaitFront = [&](Value cb, int64_t tiles) {
    if (tiles <= 0)
      return;
    int64_t outstanding = 0;
    auto it = waitState.find(cb);
    if (it != waitState.end())
      outstanding = it->second;
    if (outstanding >= tiles)
      return;
    CBWaitFrontOp::create(rewriter, loc, cb, i32(tiles));
    waitState[cb] = tiles;
  };

  emitWaitFront(inCb, inputTiles);

  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  rewriter.create<CopyTileInitOp>(loc, inCb);
  if (analysis.combineOp == TileReductionCombineOp::Sum)
    rewriter.create<AddBinaryTilesInitOp>(loc);
  else
    rewriter.create<BinaryMaxTileInitOp>(loc);

  scf::ForOp outTileLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, outTilesV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(outTileLoop.getBody());
    Value outTileIdx = outTileLoop.getInductionVar();

    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, inCb, outTileIdx, zeroI32);

    scf::ForOp reduceLoop =
        scf::ForOp::create(rewriter, loc, oneI32, reduceSlicesV, oneI32);
    {
      OpBuilder::InsertionGuard reduceGuard(rewriter);
      rewriter.setInsertionPointToStart(reduceLoop.getBody());
      Value reduceIdx = reduceLoop.getInductionVar();
      Value sliceOffset =
          arith::MulIOp::create(rewriter, loc, reduceIdx, outTilesV);
      Value inTileIdx =
          arith::AddIOp::create(rewriter, loc, sliceOffset, outTileIdx);
      rewriter.create<CopyTileOp>(loc, inCb, inTileIdx, oneI32);
      if (analysis.combineOp == TileReductionCombineOp::Sum) {
        rewriter.create<AddBinaryTilesOp>(loc, zeroI32, oneI32, zeroI32);
      } else {
        rewriter.create<BinaryMaxTileOp>(loc, zeroI32, oneI32, zeroI32);
      }
    }
    rewriter.setInsertionPointAfter(reduceLoop);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, outTileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  waitState[outCb] = 0;
  //CBPopFrontOp::create(rewriter, loc, inCb, i32(inputTiles));
  waitState[inCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static bool isIdentityMapForRank(AffineMap map, unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank)
    return false;
  for (unsigned i = 0; i < rank; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (!dimExpr || dimExpr.getPosition() != i)
      return false;
  }
  return true;
}

static bool hasAllIdentityMapsForRank(linalg::GenericOp op, unsigned rank) {
  if (op.getNumLoops() != rank)
    return false;
  for (AffineMap map : op.getIndexingMapsArray())
    if (!isIdentityMapForRank(map, rank))
      return false;
  return true;
}

static bool isScalarInputMapForRank(AffineMap map, unsigned rank) {
  return map && map.getNumDims() == rank && map.getNumSymbols() == 0 &&
         map.getNumResults() == 0;
}

/**
 * @brief Return the dropped output dimension for a pure broadcast map.
 *
 * @details A supported broadcast map is rank-1 and consists only of strictly
 *          increasing affine dim expressions, i.e. identity with exactly one
 *          missing dimension.
 */
static std::optional<unsigned> getDroppedDimFromBroadcastMap(AffineMap map,
                                                             unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank - 1 || rank == 0)
    return std::nullopt;

  llvm::SmallBitVector seenDims(rank);
  unsigned prevPos = 0;
  bool hasPrev = false;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return std::nullopt;
    unsigned pos = dimExpr.getPosition();
    if (pos >= rank || seenDims.test(pos))
      return std::nullopt;
    if (hasPrev && pos <= prevPos)
      return std::nullopt;
    seenDims.set(pos);
    prevPos = pos;
    hasPrev = true;
  }

  for (unsigned dim = 0; dim < rank; ++dim) {
    if (!seenDims.test(dim))
      return dim;
  }
  return std::nullopt;
}

/// Classify broadcast kind by dropped output dimension.
static ElementwiseBroadcastKind
classifyElementwiseBroadcastKind(unsigned droppedDim, unsigned rank) {
  if (rank >= 1 && droppedDim == rank - 1)
    return ElementwiseBroadcastKind::RowBroadcast;
  if (rank >= 2 && droppedDim == rank - 2)
    return ElementwiseBroadcastKind::ColBroadcast;
  return ElementwiseBroadcastKind::BatchBroadcast;
}

/**
 * @brief Map analysis broadcast kind to TTKernel unary-broadcast enum.
 *
 * @details TTKernel naming follows hardware lanes (`col`/`row`), so the
 *          semantic mapping from dropped dimensions is:
 *          - RowBroadcast (drop last dim)      -> `BcastType::Col`
 *          - ColBroadcast (drop second-last)   -> `BcastType::Row`
 *          - BatchBroadcast                     -> no unary broadcast op
 */
static std::optional<BcastType>
getUnaryBcastType(ElementwiseBroadcastKind kind) {
  switch (kind) {
  case ElementwiseBroadcastKind::RowBroadcast:
    return BcastType::Col;
  case ElementwiseBroadcastKind::ColBroadcast:
    return BcastType::Row;
  case ElementwiseBroadcastKind::BatchBroadcast:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Product of tile extents strictly after @p droppedDim.
static std::optional<int64_t>
getSuffixTilesAfterDim(ArrayRef<int64_t> outTileShape, unsigned droppedDim) {
  if (droppedDim >= outTileShape.size())
    return std::nullopt;
  int64_t suffix = 1;
  for (unsigned dim = droppedDim + 1; dim < outTileShape.size(); ++dim) {
    if (outTileShape[dim] <= 0)
      return std::nullopt;
    suffix *= outTileShape[dim];
  }
  return suffix;
}

static std::optional<unsigned> getBodyInputIndex(linalg::GenericOp op, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &op.getRegion().front())
    return std::nullopt;
  if (blockArg.getArgNumber() >= op.getNumDpsInputs())
    return std::nullopt;
  return blockArg.getArgNumber();
}

struct GenericBodyTileOperand {
  bool isOutput = false;
  unsigned inputIdx = 0;
};

static std::optional<GenericBodyTileOperand>
getGenericBodyTileOperand(linalg::GenericOp op, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &op.getRegion().front())
    return std::nullopt;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber < op.getNumDpsInputs())
    return GenericBodyTileOperand{/*isOutput=*/false, argNumber};

  if (op.getNumDpsInits() == 1 && argNumber == op.getNumDpsInputs())
    return GenericBodyTileOperand{/*isOutput=*/true, 0};

  return std::nullopt;
}

static bool bodyHasOnlyOps(linalg::GenericOp op, ArrayRef<Operation *> expected) {
  for (Operation &inner : op.getRegion().front().without_terminator())
    if (!llvm::is_contained(expected, &inner))
      return false;
  return true;
}

static bool isAllParallelGeneric(linalg::GenericOp op) {
  return llvm::none_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
    return type == utils::IteratorType::reduction;
  });
}

static bool isSameRankLastDimZeroBroadcastMap(AffineMap map, unsigned rank) {
  if (!map || rank == 0 || map.getNumDims() != rank ||
      map.getNumSymbols() != 0 || map.getNumResults() != rank)
    return false;

  for (unsigned dim = 0; dim + 1 < rank; ++dim) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(dim));
    if (!dimExpr || dimExpr.getPosition() != dim)
      return false;
  }

  auto constExpr = dyn_cast<AffineConstantExpr>(map.getResult(rank - 1));
  return constExpr && constExpr.getValue() == 0;
}

enum class SimpleBinaryTileKind {
  Add,
  Mul,
  Sub
};

enum class SimpleBinaryTileUnaryKind {
  Exp,
  Log
};

struct SimpleBinaryTileAnalysis {
  SimpleBinaryTileKind kind = SimpleBinaryTileKind::Add;
  GenericBodyTileOperand lhs;
  GenericBodyTileOperand rhs;
  SmallVector<SimpleBinaryTileUnaryKind, 2> unaryTail;
  int64_t outTiles = 0;
};

static LogicalResult analyzeSimpleBinaryTileGeneric(
    linalg::GenericOp op, SimpleBinaryTileAnalysis &analysis) {
  if (op.getNumDpsInits() != 1 || !isAllParallelGeneric(op))
    return failure();

  unsigned rank = op.getNumLoops();
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != static_cast<size_t>(op.getNumDpsInputs() + 1))
    return failure();
  if (!isIdentityMapForRank(maps.back(), rank))
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || !outType.hasStaticShape() || outType.getRank() != rank)
    return failure();

  auto outTiles = getNumTilesFromShapedType(outType);
  if (!outTiles || *outTiles <= 0)
    return failure();

  for (unsigned i = 0; i < op.getNumDpsInputs(); ++i) {
    if (!isIdentityMapForRank(maps[i], rank))
      return failure();
    auto inputType = dyn_cast<ShapedType>(op.getDpsInputs()[i].getType());
    if (!inputType || !inputType.hasStaticShape() ||
        inputType.getShape() != outType.getShape())
      return failure();
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  Value binaryValue = yieldOp.getValues().front();
  SmallVector<SimpleBinaryTileUnaryKind, 2> unaryTail;
  SmallVector<Operation *, 4> expectedOps;
  while (Operation *defOp = binaryValue.getDefiningOp()) {
    if (defOp->getBlock() != &op.getRegion().front())
      return failure();

    if (auto expOp = dyn_cast<math::ExpOp>(defOp)) {
      unaryTail.push_back(SimpleBinaryTileUnaryKind::Exp);
      expectedOps.push_back(defOp);
      binaryValue = expOp.getOperand();
      continue;
    }

    if (auto logOp = dyn_cast<math::LogOp>(defOp)) {
      unaryTail.push_back(SimpleBinaryTileUnaryKind::Log);
      expectedOps.push_back(defOp);
      binaryValue = logOp.getOperand();
      continue;
    }

    break;
  }

  Operation *defOp = binaryValue.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return failure();
  expectedOps.push_back(defOp);
  if (!bodyHasOnlyOps(op, expectedOps))
    return failure();

  Value lhs;
  Value rhs;
  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    analysis.kind = SimpleBinaryTileKind::Add;
    lhs = addOp.getLhs();
    rhs = addOp.getRhs();
  } else if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
    analysis.kind = SimpleBinaryTileKind::Mul;
    lhs = mulOp.getLhs();
    rhs = mulOp.getRhs();
  } else if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
    analysis.kind = SimpleBinaryTileKind::Sub;
    lhs = subOp.getLhs();
    rhs = subOp.getRhs();
  } else {
    return failure();
  }

  auto lhsOperand = getGenericBodyTileOperand(op, lhs);
  auto rhsOperand = getGenericBodyTileOperand(op, rhs);
  if (!lhsOperand || !rhsOperand)
    return failure();

  analysis.lhs = *lhsOperand;
  analysis.rhs = *rhsOperand;
  std::reverse(unaryTail.begin(), unaryTail.end());
  analysis.unaryTail = std::move(unaryTail);
  analysis.outTiles = *outTiles;
  return success();
}

static bool isSupportedSimpleBinaryTileGeneric(linalg::GenericOp op) {
  SimpleBinaryTileAnalysis analysis;
  return succeeded(analyzeSimpleBinaryTileGeneric(op, analysis));
}

struct AddInplaceAnalysis {
  unsigned accInputIdx = 0;
  unsigned addendInputIdx = 1;
  int64_t outTiles = 0;
};

static LogicalResult analyzeAddInplaceGeneric(linalg::GenericOp op,
                                              AddInplaceAnalysis &analysis) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !isAllParallelGeneric(op))
    return failure();

  unsigned rank = op.getNumLoops();
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return failure();
  if (!isIdentityMapForRank(maps[0], rank) ||
      !isIdentityMapForRank(maps[1], rank) ||
      !isIdentityMapForRank(maps[2], rank))
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || !outType.hasStaticShape() || outType.getRank() != rank)
    return failure();

  auto outTiles = getNumTilesFromShapedType(outType);
  if (!outTiles || *outTiles <= 0)
    return failure();

  for (Value input : op.getDpsInputs()) {
    auto inputType = dyn_cast<ShapedType>(input.getType());
    if (!inputType || !inputType.hasStaticShape() ||
        inputType.getShape() != outType.getShape())
      return failure();
  }

  unsigned aliasCount = 0;
  unsigned accInputIdx = 0;
  for (unsigned i = 0; i < op.getNumDpsInputs(); ++i) {
    if (op.getDpsInputs()[i] != op.getDpsInits()[0])
      continue;
    ++aliasCount;
    accInputIdx = i;
  }
  if (aliasCount != 1)
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  auto addOp = yieldOp.getValues().front().getDefiningOp<arith::AddFOp>();
  if (!addOp || addOp->getBlock() != &op.getRegion().front())
    return failure();

  SmallVector<Operation *, 1> expectedOps{addOp.getOperation()};
  if (!bodyHasOnlyOps(op, expectedOps))
    return failure();

  auto lhs = getGenericBodyTileOperand(op, addOp.getLhs());
  auto rhs = getGenericBodyTileOperand(op, addOp.getRhs());
  if (!lhs || !rhs || lhs->isOutput || rhs->isOutput ||
      lhs->inputIdx == rhs->inputIdx)
    return failure();

  unsigned addendInputIdx = accInputIdx == 0 ? 1 : 0;
  bool usesAcc = lhs->inputIdx == accInputIdx || rhs->inputIdx == accInputIdx;
  bool usesAddend =
      lhs->inputIdx == addendInputIdx || rhs->inputIdx == addendInputIdx;
  if (!usesAcc || !usesAddend)
    return failure();

  analysis.accInputIdx = accInputIdx;
  analysis.addendInputIdx = addendInputIdx;
  analysis.outTiles = *outTiles;
  return success();
}

static bool isSupportedAddInplaceGeneric(linalg::GenericOp op) {
  AddInplaceAnalysis analysis;
  return succeeded(analyzeAddInplaceGeneric(op, analysis));
}

static std::optional<unsigned> getProducerBroadcastDim(Value value) {
  Value current = value;
  llvm::SmallPtrSet<Value, 8> visited;
  while (current && visited.insert(current).second) {
    if (auto broadcastOp = current.getDefiningOp<::loom::BroadcastOp>()) {
      auto inputTiles = getNumTilesFromShapedType(broadcastOp.getIns().getType());
      auto outputTiles =
          getNumTilesFromShapedType(broadcastOp.getInit().getType());
      // Broadcasts into a distinct destination CB are materialized by
      // ConvertLoomBroadcastOp. Their consumers should use the broadcasted
      // output CB directly, not recover the producer dim and broadcast again.
      bool distinctDestination =
          stripElementwiseInputWrappers(broadcastOp.getIns()) !=
          stripElementwiseInputWrappers(broadcastOp.getInit());
      if ((inputTiles && outputTiles && *inputTiles != *outputTiles) ||
          distinctDestination)
        return std::nullopt;
      return static_cast<unsigned>(broadcastOp.getDim());
    }

    if (auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto dimAttr =
              castOp->getAttrOfType<IntegerAttr>(kBroadcastDimAttrName)) {
        int64_t dim = dimAttr.getInt();
        if (dim < 0)
          return std::nullopt;
        return static_cast<unsigned>(dim);
      }
      if (castOp.getNumOperands() == 1) {
        current = castOp.getOperand(0);
        continue;
      }
      break;
    }

    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return std::nullopt;
}

/// Unwrap temporary CB-type bridges created for loom.broadcast logical views.
static Value stripBroadcastBridgeCast(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1)
    return value;
  if (!cast->hasAttr(kBroadcastDimAttrName))
    return value;
  Value source = cast.getOperand(0);
  if (!isa<CBType>(source.getType()))
    return value;
  return source;
}

static std::optional<float> getConstFloatValue(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantFloatOp>()) {
    auto attr = dyn_cast<FloatAttr>(cst.getValue());
    if (!attr)
      return std::nullopt;
    return static_cast<float>(attr.getValueAsDouble());
  }

  // Accept casted float constants (e.g. f32 const -> truncf -> f16 scalar).
  if (auto trunc = value.getDefiningOp<arith::TruncFOp>())
    return getConstFloatValue(trunc.getIn());
  if (auto ext = value.getDefiningOp<arith::ExtFOp>())
    return getConstFloatValue(ext.getIn());

  return std::nullopt;
}

/// Tile-shape metadata for elementwise generics where only innermost two dims
/// are tiled (32x32), and outer dims are already tile-count dimensions.
struct ElementwiseTileShapeInfo {
  SmallVector<int64_t, 4> dimExtents;
  int64_t totalTiles = 0;
};

static std::optional<ElementwiseTileShapeInfo>
getElementwiseTileShapeInfo(ShapedType type) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  unsigned rank = type.getRank();
  unsigned tiledStartDim = rank > 2 ? rank - 2 : 0;

  ElementwiseTileShapeInfo info;
  info.totalTiles = 1;
  info.dimExtents.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    int64_t dimSize = type.getShape()[dim];
    if (dimSize <= 0)
      return std::nullopt;

    int64_t dimExtent = dimSize;
    if (dim >= tiledStartDim) {
      auto dimTiles = ceilDiv32(dimSize);
      if (!dimTiles || *dimTiles <= 0)
        return std::nullopt;
      dimExtent = *dimTiles;
    }

    info.dimExtents.push_back(dimExtent);
    info.totalTiles *= dimExtent;
  }

  return info;
}

static bool hasTileShapeDroppingDim(ArrayRef<int64_t> sourceTileShape,
                                    ArrayRef<int64_t> targetTileShape,
                                    unsigned droppedDim) {
  if (droppedDim >= targetTileShape.size() ||
      sourceTileShape.size() + 1 != targetTileShape.size())
    return false;

  unsigned sourceDim = 0;
  for (unsigned targetDim = 0; targetDim < targetTileShape.size();
       ++targetDim) {
    if (targetDim == droppedDim)
      continue;
    if (sourceDim >= sourceTileShape.size() ||
        sourceTileShape[sourceDim] != targetTileShape[targetDim])
      return false;
    ++sourceDim;
  }

  return sourceDim == sourceTileShape.size();
}

static bool hasSameRankTileShapeBroadcastingDim(
    ArrayRef<int64_t> sourceTileShape, ArrayRef<int64_t> targetTileShape,
    unsigned broadcastDim) {
  if (sourceTileShape.size() != targetTileShape.size() ||
      broadcastDim >= targetTileShape.size())
    return false;

  for (unsigned dim = 0; dim < targetTileShape.size(); ++dim) {
    if (dim == broadcastDim) {
      if (sourceTileShape[dim] != 1 || targetTileShape[dim] <= 0)
        return false;
      continue;
    }
    if (sourceTileShape[dim] != targetTileShape[dim])
      return false;
  }
  return true;
}

struct FusedMulBcastAddAnalysis {
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t outTiles = 0;
  int64_t scaleTiles = 0;
};

static bool matchInputBlockArg(linalg::GenericOp op, Value value,
                               unsigned inputIdx) {
  auto actual = getBodyInputIndex(op, value);
  return actual && *actual == inputIdx;
}

static bool matchMulOperands(linalg::GenericOp op, arith::MulFOp mulOp,
                             unsigned lhsIdx, unsigned rhsIdx) {
  return (matchInputBlockArg(op, mulOp.getLhs(), lhsIdx) &&
          matchInputBlockArg(op, mulOp.getRhs(), rhsIdx)) ||
         (matchInputBlockArg(op, mulOp.getLhs(), rhsIdx) &&
          matchInputBlockArg(op, mulOp.getRhs(), lhsIdx));
}

static LogicalResult analyzeMulBcastColsShape(
    linalg::GenericOp op, unsigned accInputIdx, unsigned scaleInputIdx,
    FusedMulBcastAddAnalysis &analysis) {
  if (accInputIdx >= op.getNumDpsInputs() ||
      scaleInputIdx >= op.getNumDpsInputs())
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  auto accType = dyn_cast<ShapedType>(op.getDpsInputs()[accInputIdx].getType());
  auto scaleType =
      dyn_cast<ShapedType>(op.getDpsInputs()[scaleInputIdx].getType());
  if (!outType || !accType || !scaleType || !outType.hasStaticShape() ||
      !accType.hasStaticShape() || !scaleType.hasStaticShape() ||
      accType.getShape() != outType.getShape() ||
      scaleType.getRank() != outType.getRank())
    return failure();

  auto outTileInfo = getElementwiseTileShapeInfo(outType);
  auto scaleTileInfo = getElementwiseTileShapeInfo(scaleType);
  if (!outTileInfo || !scaleTileInfo || outTileInfo->dimExtents.empty())
    return failure();
  unsigned bcastDim = outTileInfo->dimExtents.size() - 1;
  if (!hasSameRankTileShapeBroadcastingDim(
          scaleTileInfo->dimExtents, outTileInfo->dimExtents, bcastDim))
    return failure();

  int64_t cols = outTileInfo->dimExtents.back();
  if (cols <= 0 || outTileInfo->totalTiles % cols != 0)
    return failure();
  int64_t rows = outTileInfo->totalTiles / cols;
  if (rows <= 0 || scaleTileInfo->totalTiles != rows)
    return failure();

  analysis.rows = rows;
  analysis.cols = cols;
  analysis.outTiles = outTileInfo->totalTiles;
  analysis.scaleTiles = scaleTileInfo->totalTiles;
  return success();
}

static LogicalResult analyzeFusedMulBcastAddGeneric(
    linalg::GenericOp op, FusedMulBcastAddAnalysis &analysis) {
  if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1 ||
      !isAllParallelGeneric(op))
    return failure();

  unsigned rank = op.getNumLoops();
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 4)
    return failure();
  if (!isIdentityMapForRank(maps[0], rank) ||
      !isIdentityMapForRank(maps[1], rank) ||
      !isSameRankLastDimZeroBroadcastMap(maps[2], rank) ||
      !isIdentityMapForRank(maps[3], rank))
    return failure();

  if (op.getDpsInputs()[1] != op.getDpsInits()[0])
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  auto addType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  if (!outType || !addType || !outType.hasStaticShape() ||
      !addType.hasStaticShape() || addType.getShape() != outType.getShape())
    return failure();

  if (failed(analyzeMulBcastColsShape(op, /*accInputIdx=*/1,
                                      /*scaleInputIdx=*/2, analysis)))
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  auto addOp = yieldOp.getValues().front().getDefiningOp<arith::AddFOp>();
  if (!addOp || addOp->getBlock() != &op.getRegion().front())
    return failure();

  Value maybeMul = addOp.getLhs();
  Value maybeAddInput = addOp.getRhs();
  if (matchInputBlockArg(op, maybeMul, 0)) {
    maybeAddInput = maybeMul;
    maybeMul = addOp.getRhs();
  }
  if (!matchInputBlockArg(op, maybeAddInput, 0))
    return failure();

  auto mulOp = maybeMul.getDefiningOp<arith::MulFOp>();
  if (!mulOp || mulOp->getBlock() != &op.getRegion().front() ||
      !matchMulOperands(op, mulOp, /*lhsIdx=*/1, /*rhsIdx=*/2))
    return failure();

  SmallVector<Operation *, 2> expectedOps{addOp.getOperation(),
                                          mulOp.getOperation()};
  if (!bodyHasOnlyOps(op, expectedOps))
    return failure();

  return success();
}

static bool isSupportedFusedMulBcastAddGeneric(linalg::GenericOp op) {
  FusedMulBcastAddAnalysis analysis;
  return succeeded(analyzeFusedMulBcastAddGeneric(op, analysis));
}

static LogicalResult analyzeInplaceMulBcastColsGeneric(
    linalg::GenericOp op, FusedMulBcastAddAnalysis &analysis) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !isAllParallelGeneric(op))
    return failure();

  unsigned rank = op.getNumLoops();
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return failure();
  if (!isIdentityMapForRank(maps[0], rank) ||
      !isSameRankLastDimZeroBroadcastMap(maps[1], rank) ||
      !isIdentityMapForRank(maps[2], rank))
    return failure();

  if (op.getDpsInputs()[0] != op.getDpsInits()[0])
    return failure();

  if (failed(analyzeMulBcastColsShape(op, /*accInputIdx=*/0,
                                      /*scaleInputIdx=*/1, analysis)))
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  auto mulOp = yieldOp.getValues().front().getDefiningOp<arith::MulFOp>();
  if (!mulOp || mulOp->getBlock() != &op.getRegion().front() ||
      !matchMulOperands(op, mulOp, /*lhsIdx=*/0, /*rhsIdx=*/1))
    return failure();

  SmallVector<Operation *, 1> expectedOps{mulOp.getOperation()};
  if (!bodyHasOnlyOps(op, expectedOps))
    return failure();

  return success();
}

static bool isSupportedInplaceMulBcastColsGeneric(linalg::GenericOp op) {
  FusedMulBcastAddAnalysis analysis;
  return succeeded(analyzeInplaceMulBcastColsGeneric(op, analysis));
}

static std::optional<int64_t> getTileDim(ShapedType type, unsigned dim) {
  if (!type.hasStaticShape() || dim >= type.getRank())
    return std::nullopt;
  return ceilDiv32(type.getShape()[dim]);
}

struct MatmulTileInfo {
  int64_t batchSize = 1;
  int64_t rt = 0;
  int64_t kt = 0;
  int64_t ct = 0;
  int64_t nt = 0;
  int64_t in0TilesPerBatch = 0;
  int64_t in1TilesPerBatch = 0;
  int64_t outTilesPerBatch = 0;
  int64_t in0TilesTotal = 0;
  int64_t in1TilesTotal = 0;
  int64_t outTilesTotal = 0;
};

static int64_t getDstCapacityTiles(ShapedType /*outType*/) {
  //Don't clean this comment, the setting of DST capacity is related to setting of dst_full_sync_en in host program. There are 16 DSTs in total, using half of them supports double-buffered execution.
  return 8;
}

/**
 * @brief Analyze rank-3 batch_matmul shapes into tile metadata.
 *
 * @details `linalg.matmul` is normalized to batch-1 `linalg.batch_matmul`
 *          earlier in the pipeline, so this helper only accepts rank-3 shapes.
 */
static std::optional<MatmulTileInfo>
getMatmulTileInfo(ShapedType lhsType, ShapedType rhsType, ShapedType outType) {
  if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outType.hasStaticShape())
    return std::nullopt;

  MatmulTileInfo info;

  auto toI64 = [](std::optional<int64_t> value) -> std::optional<int64_t> {
    if (!value || *value <= 0)
      return std::nullopt;
    return value;
  };

  if (lhsType.getRank() == 3 && rhsType.getRank() == 3 &&
      outType.getRank() == 3) {
    ArrayRef<int64_t> lhs = lhsType.getShape();
    ArrayRef<int64_t> rhs = rhsType.getShape();
    ArrayRef<int64_t> out = outType.getShape();
    if (lhs[0] != rhs[0] || lhs[0] != out[0] || lhs[1] != out[1] ||
        lhs[2] != rhs[1] || rhs[2] != out[2] || lhs[0] <= 0)
      return std::nullopt;

    auto rt = toI64(getTileDim(lhsType, 1));
    auto kt = toI64(getTileDim(lhsType, 2));
    auto ct = toI64(getTileDim(rhsType, 2));
    auto nt = toI64(getTileDim(outType, 2));
    if (!rt || !kt || !ct || !nt)
      return std::nullopt;

    info.batchSize = lhs[0];
    info.rt = *rt;
    info.kt = *kt;
    info.ct = *ct;
    info.nt = *nt;
  } else {
    return std::nullopt;
  }

  info.in0TilesPerBatch = info.rt * info.kt;
  info.in1TilesPerBatch = info.kt * info.ct;
  info.outTilesPerBatch = info.rt * info.nt;
  info.in0TilesTotal = info.in0TilesPerBatch * info.batchSize;
  info.in1TilesTotal = info.in1TilesPerBatch * info.batchSize;
  info.outTilesTotal = info.outTilesPerBatch * info.batchSize;
  if (info.in0TilesPerBatch <= 0 || info.in1TilesPerBatch <= 0 ||
      info.outTilesPerBatch <= 0 || info.in0TilesTotal <= 0 ||
      info.in1TilesTotal <= 0 || info.outTilesTotal <= 0)
    return std::nullopt;

  return info;
}

static Value getScalarBitsFromFloat(ConversionPatternRewriter &rewriter,
                                    Location loc, float value) {
  Value f32Const = rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), llvm::APFloat(value));
  return rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(), f32Const);
}

static bool hasOutputAlias(Value outCb, ValueRange inCbs) {
  return llvm::is_contained(inCbs, outCb);
}

static int64_t getOutstandingWaitTiles(const llvm::DenseMap<Value, int64_t> &state,
                                       Value cb) {
  auto it = state.find(cb);
  return it == state.end() ? 0 : it->second;
}

static void emitWaitFrontIfNeeded(ConversionPatternRewriter &rewriter, Location loc,
                                  Value cb, int64_t tiles,
                                  llvm::DenseMap<Value, int64_t> &state) {
  if (tiles <= 0)
    return;
  int64_t outstanding = getOutstandingWaitTiles(state, cb);
  if (outstanding >= tiles)
    return;
  CBWaitFrontOp::create(rewriter, loc, cb, i32Const(rewriter, loc, tiles));
  state[cb] = tiles;
}

static std::optional<std::string> getStaticCBExpr(Value cb) {
  Value cbRoot = cb;
  while (auto cast = cbRoot.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1 || !isa<CBType>(cast.getOperand(0).getType()))
      break;
    cbRoot = cast.getOperand(0);
  }

  auto cbConst = cbRoot.getDefiningOp<GetCompileArgValOp>();
  if (!cbConst)
    return std::nullopt;

  if (auto nameAttr = cbConst->getAttrOfType<StringAttr>(kCBConstNameAttrName))
    return nameAttr.getValue().str();
  if (auto valueAttr =
          cbConst->getAttrOfType<IntegerAttr>(kCBConstValueAttrName))
    return std::to_string(valueAttr.getInt());
  return std::to_string(cbConst.getArgIndex());
}

static FailureOr<Value>
createEmitCCBLiteral(ConversionPatternRewriter &rewriter, Location loc,
                     Value cb) {
  std::optional<std::string> cbExpr = getStaticCBExpr(cb);
  if (!cbExpr)
    return failure();
  return rewriter
      .create<emitc::LiteralOp>(
          loc, rewriter.getType<emitc::OpaqueType>("::tt::CB"), *cbExpr)
      .getResult();
}

static void emitPackReconfigDataFormat(ConversionPatternRewriter &rewriter,
                                       Location loc, Value outCb) {
  std::optional<std::string> cbExpr = getStaticCBExpr(outCb);
  if (!cbExpr) {
    emitError(loc) << "failed to emit pack_reconfig_data_format for "
                   << "non-static CB";
    return;
  }
  rewriter.create<emitc::VerbatimOp>(
      loc, "pack_reconfig_data_format(" + *cbExpr + ");");
}

static void emitPackReconfigL1Acc(ConversionPatternRewriter &rewriter,
                                  Location loc, int64_t l1AccEn) {
  rewriter.create<emitc::VerbatimOp>(
      loc, "pack_reconfig_l1_acc(" + std::to_string(l1AccEn) + ");");
}

static void emitPackOverwriteMode(ConversionPatternRewriter &rewriter,
                                  Location loc, Value outCb) {
  emitPackReconfigDataFormat(rewriter, loc, outCb);
  emitPackReconfigL1Acc(rewriter, loc, 0);
}

static std::string stringifyAttribute(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  os.flush();
  return text;
}

static bool attrContains(Attribute attr, StringRef needle) {
  return attr && StringRef(stringifyAttribute(attr)).contains(needle);
}

static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static bool isZeroIntegerValue(Value value) {
  std::optional<int64_t> intValue = getConstantIntValue(value);
  return intValue && *intValue == 0;
}

static bool isZeroFloatValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constant.getValue()))
      return floatAttr.getValue().isZero();
    if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return intAttr.getValue().isZero();
  }
  if (auto ext = value.getDefiningOp<arith::ExtFOp>())
    return isZeroFloatValue(ext.getIn());
  if (auto cast = value.getDefiningOp<arith::SIToFPOp>())
    return isZeroIntegerValue(cast.getIn());
  return false;
}

static bool containsZeroFillTile(Operation *op) {
  bool found = false;
  op->walk([&](FillTileOp fillTile) {
    if (isZeroFloatValue(fillTile.getValue()))
      found = true;
  });
  return found;
}

static bool isZeroFillForValue(linalg::FillOp fillOp, Value output) {
  if (fillOp.getOutputs().empty() ||
      stripViewLikeWrappers(fillOp.getOutputs()[0]) !=
          stripViewLikeWrappers(output))
    return false;
  if (fillOp.getInputs().empty())
    return false;
  return isZeroFloatValue(fillOp.getInputs()[0]);
}

static bool hasKnownZeroFillInitializerBefore(Operation *anchor,
                                              Value originalOut, Value outCb) {
  if (!anchor || !anchor->getBlock())
    return false;

  bool sawConvertedPush = false;
  for (Operation *prev = anchor->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (auto fillOp = dyn_cast<linalg::FillOp>(prev)) {
      if (isZeroFillForValue(fillOp, originalOut))
        return true;
    }

    if (auto push = dyn_cast<CBPushBackOp>(prev)) {
      if (push.getCb() == outCb)
        sawConvertedPush = true;
      continue;
    }

    if (sawConvertedPush && containsZeroFillTile(prev))
      return true;

    if (sawConvertedPush) {
      if (auto reserve = dyn_cast<CBReserveBackOp>(prev)) {
        if (reserve.getCb() == outCb)
          return false;
      }
    }
  }
  return false;
}

static bool isNextUseDpsInitOfSameOutput(linalg::FillOp fillOp) {
  if (fillOp->getNumResults() != 0 || fillOp.getOutputs().size() != 1)
    return false;

  Value output = stripViewLikeWrappers(fillOp.getOutputs()[0]);
  if (!output)
    return false;

  for (Operation *next = fillOp->getNextNode(); next;
       next = next->getNextNode()) {
    bool usesOutput = false;
    bool usesAsInit = false;
    bool usesAsInput = false;

    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(next)) {
      for (Value input : dpsOp.getDpsInputs()) {
        if (stripViewLikeWrappers(input) != output)
          continue;
        usesOutput = true;
        usesAsInput = true;
      }
      for (Value init : dpsOp.getDpsInits()) {
        if (stripViewLikeWrappers(init) != output)
          continue;
        usesOutput = true;
        usesAsInit = true;
      }
    } else {
      for (OpOperand &operand : next->getOpOperands()) {
        if (stripViewLikeWrappers(operand.get()) != output)
          continue;
        usesOutput = true;
      }
    }

    if (!usesOutput)
      continue;

    // Safe to erase only when the next consumer overwrites the buffer without
    // also reading it (e.g. matmul outs(%C), not in-place generic ins+outs(%C)).
    if (usesAsInit && !usesAsInput)
      return true;
    return false;
  }

  return false;
}

struct KSplitLoopInfo {
  scf::ForOp loop;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
};

static std::optional<KSplitLoopInfo>
getEnclosingStaticKSplitLoop(Operation *op) {
  for (scf::ForOp loop = op->getParentOfType<scf::ForOp>(); loop;
       loop = loop->getParentOfType<scf::ForOp>()) {
    Attribute iterType = loop->getAttr("loom.iter_type");
    if (!attrContains(iterType, "sequential"))
      continue;

    std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper - *lower <= 1)
      continue;

    return KSplitLoopInfo{loop, *lower, *upper, *step};
  }
  return std::nullopt;
}

static Value intConstLike(OpBuilder &builder, Location loc, Type type,
                          int64_t value) {
  if (type.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, value);
  if (auto intType = dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantIntOp>(loc, intType, value);
  return {};
}

static Value cmpBlockIv(ConversionPatternRewriter &rewriter, Location loc,
                        KSplitLoopInfo loopInfo,
                        arith::CmpIPredicate pred, int64_t value) {
  Value iv = loopInfo.loop.getInductionVar();
  Value rhs = intConstLike(rewriter, loc, iv.getType(), value);
  return rewriter.create<arith::CmpIOp>(loc, pred, iv, rhs);
}

static void copyTile(ConversionPatternRewriter &rewriter, Location loc,
                     Value inputCb, Value outputCb, Value tiles,
                     bool popInputCb = true) {
  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  CBWaitFrontOp::create(rewriter, loc, inputCb, tiles);
  CBReserveBackOp::create(rewriter, loc, outputCb, tiles);
  emitPackOverwriteMode(rewriter, loc, outputCb);
  rewriter.create<CopyTileInitOp>(loc, inputCb);
  scf::ForOp tileLoop =
      scf::ForOp::create(rewriter, loc, zero, tiles, one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, inputCb, tileIdx, zero);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zero, outputCb, tileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  if (popInputCb) {
    // Default behavior consumes input tiles once the copy is complete.
    CBPopFrontOp::create(rewriter, loc, inputCb, tiles);
  }
}

template <typename BuilderFn>
static LogicalResult emitElementwiseTiles(
    ConversionPatternRewriter &rewriter, Location loc, Value outCb, int64_t outTiles,
    BuilderFn &&builder) {
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);
  Value outTilesV = i32Const(rewriter, loc, outTiles);
  scf::ForOp tileLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, outTilesV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();

    TileRegsAcquireOp::create(rewriter, loc);
    if (failed(builder(tileIdx)))
      return failure();
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, tileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  return success();
}

static bool isGreaterCmp(arith::CmpFPredicate pred) {
  return pred == arith::CmpFPredicate::OGT ||
         pred == arith::CmpFPredicate::OGE ||
         pred == arith::CmpFPredicate::UGT ||
         pred == arith::CmpFPredicate::UGE;
}

static bool isLessCmp(arith::CmpFPredicate pred) {
  return pred == arith::CmpFPredicate::OLT ||
         pred == arith::CmpFPredicate::OLE ||
         pred == arith::CmpFPredicate::ULT ||
         pred == arith::CmpFPredicate::ULE;
}

static bool matchMaxSelect(arith::SelectOp selectOp, Value &lhs, Value &rhs) {
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp)
    return false;

  arith::CmpFPredicate pred = cmp.getPredicate();
  Value cmpLhs = cmp.getLhs();
  Value cmpRhs = cmp.getRhs();
  Value trueV = selectOp.getTrueValue();
  Value falseV = selectOp.getFalseValue();

  if (isGreaterCmp(pred) && trueV == cmpLhs && falseV == cmpRhs) {
    lhs = cmpLhs;
    rhs = cmpRhs;
    return true;
  }

  if (isLessCmp(pred) && trueV == cmpRhs && falseV == cmpLhs) {
    lhs = cmpLhs;
    rhs = cmpRhs;
    return true;
  }

  return false;
}

enum class GenericExprFeature {
  BinopWithScalar,
  SubBinary,
  AddBinary,
  MulBinary,
  BinaryMax,
  PowBinary,
  Recip,
  Log,
  Exp,
  Fill
};

/**
 * @brief Recursively analyze a generic expression tree.
 *
 * @details This helper is shared by FlashAttention elementwise analysis and
 *          tile-generic analysis. Callers provide leaf handling and feature
 *          flag plumbing while this function walks the op tree.
 */
template <typename LeafHandlerT, typename ConstHandlerT,
          typename PowBaseConstHandlerT, typename FlagHandlerT>
static LogicalResult analyzeGenericExprTree(
    Value exprValue, linalg::GenericOp op, LeafHandlerT &&handleLeaf,
    ConstHandlerT &&handleConst, PowBaseConstHandlerT &&handlePowBaseConst,
    FlagHandlerT &&setFlag, bool allowMaximumFOp) {
  auto recurse = [&](auto &&self, Value value) -> LogicalResult {
    if (handleLeaf(value))
      return success();

    if (getConstFloatValue(value).has_value()) {
      handleConst();
      return success();
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != &op.getRegion().front())
      return failure();

    if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(mulOp.getLhs());
      auto rhsConst = getConstFloatValue(mulOp.getRhs());
      if (lhsConst && !rhsConst) {
        setFlag(GenericExprFeature::BinopWithScalar);
        return self(self, mulOp.getRhs());
      }
      if (rhsConst && !lhsConst) {
        setFlag(GenericExprFeature::BinopWithScalar);
        return self(self, mulOp.getLhs());
      }
      setFlag(GenericExprFeature::MulBinary);
      if (failed(self(self, mulOp.getLhs())))
        return failure();
      return self(self, mulOp.getRhs());
    }

    if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
      setFlag(GenericExprFeature::AddBinary);
      if (failed(self(self, addOp.getLhs())))
        return failure();
      return self(self, addOp.getRhs());
    }

    if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
      setFlag(GenericExprFeature::SubBinary);
      if (failed(self(self, subOp.getLhs())))
        return failure();
      return self(self, subOp.getRhs());
    }

    if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
      setFlag(GenericExprFeature::Recip);
      setFlag(GenericExprFeature::MulBinary);
      if (failed(self(self, divOp.getLhs())))
        return failure();
      return self(self, divOp.getRhs());
    }

    if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
      if (!getConstFloatValue(powOp.getLhs()).has_value())
        return failure();
      handlePowBaseConst();
      setFlag(GenericExprFeature::PowBinary);
      return self(self, powOp.getRhs());
    }

    if (auto logOp = dyn_cast<math::LogOp>(defOp)) {
      setFlag(GenericExprFeature::Log);
      return self(self, logOp.getOperand());
    }

    if (auto expOp = dyn_cast<math::ExpOp>(defOp)) {
      setFlag(GenericExprFeature::Exp);
      return self(self, expOp.getOperand());
    }

    if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
      if (!allowMaximumFOp)
        return failure();
      setFlag(GenericExprFeature::BinaryMax);
      if (failed(self(self, maxOp.getLhs())))
        return failure();
      return self(self, maxOp.getRhs());
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      Value lhs;
      Value rhs;
      if (!matchMaxSelect(selectOp, lhs, rhs))
        return failure();
      setFlag(GenericExprFeature::BinaryMax);
      if (failed(self(self, lhs)))
        return failure();
      return self(self, rhs);
    }

    return failure();
  };

  return recurse(recurse, exprValue);
}

/**
 * @brief Resolve tile-generic body values to input or accumulator operands.
 */
static std::optional<TileGenericBodyOperand>
getTileGenericBodyOperand(linalg::GenericOp op, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &op.getRegion().front())
    return std::nullopt;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber == 0)
    return TileGenericBodyOperand::Input;

  if (argNumber == op.getNumDpsInputs())
    return TileGenericBodyOperand::Accumulator;

  return std::nullopt;
}

static bool hasOnlyBodyOps(linalg::GenericOp op, Operation *first,
                           Operation *second = nullptr) {
  for (Operation &inner : op.getRegion().front().without_terminator())
    if (&inner != first && &inner != second)
      return false;
  return true;
}

static bool matchTileReductionOperands(linalg::GenericOp op, Value lhs,
                                       Value rhs) {
  auto lhsOperand = getTileGenericBodyOperand(op, lhs);
  auto rhsOperand = getTileGenericBodyOperand(op, rhs);
  if (!lhsOperand || !rhsOperand)
    return false;

  return (*lhsOperand == TileGenericBodyOperand::Input &&
          *rhsOperand == TileGenericBodyOperand::Accumulator) ||
         (*lhsOperand == TileGenericBodyOperand::Accumulator &&
          *rhsOperand == TileGenericBodyOperand::Input);
}

static LogicalResult analyzeTileReductionGeneric(
    linalg::GenericOp op, TileReductionAnalysis &analysis) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  Value yielded = yieldOp.getValues().front();
  Operation *defOp = yielded.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return failure();

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    if (!hasOnlyBodyOps(op, addOp.getOperation()) ||
        !matchTileReductionOperands(op, addOp.getLhs(), addOp.getRhs()))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Sum;
    return success();
  }

  if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
    if (!hasOnlyBodyOps(op, maxOp.getOperation()) ||
        !matchTileReductionOperands(op, maxOp.getLhs(), maxOp.getRhs()))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Max;
    return success();
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value lhs;
    Value rhs;
    if (!matchMaxSelect(selectOp, lhs, rhs) ||
        !matchTileReductionOperands(op, lhs, rhs))
      return failure();
    Operation *cmpOp = selectOp.getCondition().getDefiningOp();
    if (!cmpOp || cmpOp->getBlock() != &op.getRegion().front() ||
        !hasOnlyBodyOps(op, selectOp.getOperation(), cmpOp))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Max;
    return success();
  }

  return failure();
}

/**
 * @brief Shared recursive emission for generic expression trees.
 *
 * @details This helper emits all non-leaf expression operators shared by
 *          FlashAttention elementwise and tile-generic lowering. Callers
 *          provide leaf materialization and operand-order policy.
 */
template <typename LeafEmitterT, typename RhsFirstPolicyT,
          typename RuntimeScalarLookupT>
static LogicalResult emitGenericExprToRegImpl(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, bool emitInlineInitOps, bool allowMaximumFOp,
    LeafEmitterT &&emitLeafToReg, RhsFirstPolicyT &&emitRhsFirstForOperands,
    RuntimeScalarLookupT &&lookupRuntimeScalarBits) {
  auto recurse =
      [&](auto &&self, Value value, int curDstReg, int curTmpRegA,
          int curTmpRegB) -> LogicalResult {
    Value curDstRegVal = i32Const(rewriter, loc, curDstReg);
    Value curTmpRegAVal = i32Const(rewriter, loc, curTmpRegA);

    bool handledLeaf = false;
    if (failed(emitLeafToReg(value, curDstReg, handledLeaf)))
      return failure();
    if (handledLeaf)
      return success();

    if (auto constFloat = getConstFloatValue(value)) {
      if (emitInlineInitOps)
        rewriter.create<FillTileInitOp>(loc);
      Value constVal = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), llvm::APFloat(*constFloat));
      rewriter.create<FillTileOp>(loc, curDstRegVal, constVal);
      return success();
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != &op.getRegion().front())
      return failure();

    if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(mulOp.getLhs());
      auto rhsConst = getConstFloatValue(mulOp.getRhs());
      if (lhsConst && !rhsConst) {
        if (failed(self(self, mulOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(
            loc, curDstRegVal, getScalarBitsFromFloat(rewriter, loc, *lhsConst));
        return success();
      }
      if (rhsConst && !lhsConst) {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(
            loc, curDstRegVal, getScalarBitsFromFloat(rewriter, loc, *rhsConst));
        return success();
      }
      Value lhsRuntimeScalar = lookupRuntimeScalarBits(mulOp.getLhs());
      Value rhsRuntimeScalar = lookupRuntimeScalarBits(mulOp.getRhs());
      if (lhsRuntimeScalar && !rhsRuntimeScalar) {
        if (failed(self(self, mulOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(loc, curDstRegVal, lhsRuntimeScalar);
        return success();
      }
      if (rhsRuntimeScalar && !lhsRuntimeScalar) {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(loc, curDstRegVal, rhsRuntimeScalar);
        return success();
      }
      if (lhsRuntimeScalar && rhsRuntimeScalar)
        return failure();

      bool emitRhsFirst = emitRhsFirstForOperands(mulOp.getLhs(), mulOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, mulOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, mulOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<MulBinaryTilesInitOp>(loc);
      rewriter.create<MulBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(addOp.getLhs(), addOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, addOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, addOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, addOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, addOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<AddBinaryTilesInitOp>(loc);
      rewriter.create<AddBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(subOp.getLhs(), subOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, subOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, subOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, subOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, subOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<SubBinaryTilesInitOp>(loc);
      rewriter.create<SubBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(divOp.getLhs(), divOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, divOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, divOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, divOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, divOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<RecipTileInitOp>(loc);
      rewriter.create<RecipTileOp>(loc, curTmpRegAVal);
      if (emitInlineInitOps)
        rewriter.create<MulBinaryTilesInitOp>(loc);
      rewriter.create<MulBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(powOp.getLhs());
      if (!lhsConst)
        return failure();
      if (failed(self(self, powOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
        return failure();
      Value lhsConstVal = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), llvm::APFloat(*lhsConst));
      if (emitInlineInitOps)
        rewriter.create<FillTileInitOp>(loc);
      rewriter.create<FillTileOp>(loc, curTmpRegAVal, lhsConstVal);
      if (emitInlineInitOps)
        rewriter.create<PowBinaryTilesInitOp>(loc);
      rewriter.create<PowBinaryTilesOp>(loc, curTmpRegAVal, curDstRegVal,
                                        curDstRegVal);
      return success();
    }

    if (auto logOp = dyn_cast<math::LogOp>(defOp)) {
      if (failed(self(self, logOp.getOperand(), curDstReg, curTmpRegA,
                      curTmpRegB)))
        return failure();
      if (emitInlineInitOps)
        rewriter.create<LogTileInitOp>(loc);
      rewriter.create<LogTileOp>(loc, curDstRegVal);
      return success();
    }

    if (auto expOp = dyn_cast<math::ExpOp>(defOp)) {
      if (failed(self(self, expOp.getOperand(), curDstReg, curTmpRegA,
                      curTmpRegB)))
        return failure();
      if (emitInlineInitOps)
        rewriter.create<ExpTileInitOp>(loc);
      rewriter.create<ExpTileOp>(loc, curDstRegVal);
      return success();
    }

    auto emitBinaryMax = [&](Value lhs, Value rhs) -> LogicalResult {
      bool emitRhsFirst = emitRhsFirstForOperands(lhs, rhs);
      if (emitRhsFirst) {
        if (failed(self(self, rhs, curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, lhs, curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, lhs, curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, rhs, curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<BinaryMaxTileInitOp>(loc);
      rewriter.create<BinaryMaxTileOp>(loc, curDstRegVal, curTmpRegAVal,
                                       curDstRegVal);
      return success();
    };

    if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
      if (!allowMaximumFOp)
        return failure();
      return emitBinaryMax(maxOp.getLhs(), maxOp.getRhs());
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      Value lhs;
      Value rhs;
      if (!matchMaxSelect(selectOp, lhs, rhs))
        return failure();
      return emitBinaryMax(lhs, rhs);
    }

    return failure();
  };

  return recurse(recurse, exprValue, dstReg, tmpRegA, tmpRegB);
}

static LogicalResult analyzeElementwiseExpr(Value exprValue, linalg::GenericOp op,
                                            ElementwiseAnalysis &analysis) {
  auto handleLeaf = [&](Value value) -> bool {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx)
      return false;
    analysis.usedInputs.set(*inputIdx);
    return true;
  };
  auto handleConst = [&]() {};
  auto handlePowBaseConst = [&]() {};
  auto setFlag = [&](GenericExprFeature feature) {
    switch (feature) {
    case GenericExprFeature::BinopWithScalar:
      analysis.needsBinopWithScalar = true;
      return;
    case GenericExprFeature::SubBinary:
      analysis.needsSubBinary = true;
      return;
    case GenericExprFeature::AddBinary:
      analysis.needsAddBinary = true;
      return;
    case GenericExprFeature::MulBinary:
      analysis.needsMulBinary = true;
      return;
    case GenericExprFeature::BinaryMax:
      analysis.needsBinaryMax = true;
      return;
    case GenericExprFeature::PowBinary:
      analysis.needsPowBinary = true;
      return;
    case GenericExprFeature::Recip:
      analysis.needsRecip = true;
      return;
    case GenericExprFeature::Log:
      analysis.needsLog = true;
      return;
    case GenericExprFeature::Exp:
      analysis.needsExp = true;
      return;
    case GenericExprFeature::Fill:
      return;
    }
  };
  return analyzeGenericExprTree(
      exprValue, op, handleLeaf, handleConst, handlePowBaseConst, setFlag,
      /*allowMaximumFOp=*/false);
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis) {
  if (op.getNumDpsInits() != 1)
    return failure();

  for (utils::IteratorType iterType : op.getIteratorTypesArray())
    if (iterType == utils::IteratorType::reduction)
      return failure();

  unsigned rank = op.getNumLoops();
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || !outType.hasStaticShape() || outType.getRank() != rank)
    return failure();

  auto outTileInfo = getElementwiseTileShapeInfo(outType);
  if (!outTileInfo)
    return failure();
  ArrayRef<int64_t> outTileShape = outTileInfo->dimExtents;

  unsigned numInputs = op.getNumDpsInputs();
  analysis.broadcastsByInput.assign(numInputs, std::nullopt);

  bool allIdentity = hasAllIdentityMapsForRank(op, rank);
  if (!allIdentity) {
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != numInputs + 1)
      return failure();
    if (!isIdentityMapForRank(maps[numInputs], rank))
      return failure();

    for (unsigned i = 0; i < numInputs; ++i) {
      if (isIdentityMapForRank(maps[i], rank))
        continue;

      ElementwiseBroadcastInfo info;
      info.inputIdx = i;
      if (isScalarInputMapForRank(maps[i], rank)) {
        // Scalar map: one scalar tile reused for every output tile.
        info.isScalar = true;
        info.kind = ElementwiseBroadcastKind::BatchBroadcast;
        info.suffixTiles = 1;
        info.droppedDimTiles = outTileInfo->totalTiles;
      } else {
        std::optional<unsigned> droppedDim =
            getDroppedDimFromBroadcastMap(maps[i], rank);
        if (!droppedDim)
          return failure();

        auto suffixTiles = getSuffixTilesAfterDim(outTileShape, *droppedDim);
        if (!suffixTiles || *suffixTiles <= 0)
          return failure();

        int64_t droppedDimTiles = outTileShape[*droppedDim];
        if (droppedDimTiles <= 0)
          return failure();

        info.droppedDim = *droppedDim;
        info.kind = classifyElementwiseBroadcastKind(*droppedDim, rank);
        info.droppedDimTiles = droppedDimTiles;
        info.suffixTiles = *suffixTiles;
        // Elementwise row/col broadcasts must be explicit loom.broadcast ops.
        if (info.kind != ElementwiseBroadcastKind::BatchBroadcast)
          return failure();
      }
      analysis.broadcastsByInput[i] = info;
    }

    // Non-identity elementwise maps must all be supported pure broadcasts.
    bool foundBroadcast = llvm::any_of(analysis.broadcastsByInput, [](const auto &info) {
      return info.has_value();
    });
    if (!foundBroadcast)
      return failure();
  }

  // Identity-mapped inputs can still be logical broadcasts when produced by
  // loom.broadcast. Use producer metadata to recover dropped dimension info.
  for (unsigned i = 0; i < numInputs; ++i) {
    if (analysis.broadcastsByInput[i])
      continue;

    auto producerDim = getProducerBroadcastDim(op.getDpsInputs()[i]);
    if (!producerDim)
      continue;
    if (*producerDim >= rank)
      return failure();

    auto suffixTiles = getSuffixTilesAfterDim(outTileShape, *producerDim);
    if (!suffixTiles || *suffixTiles <= 0)
      return failure();

    int64_t droppedDimTiles = outTileShape[*producerDim];
    if (droppedDimTiles <= 0)
      return failure();

    ElementwiseBroadcastInfo info;
    info.inputIdx = i;
    info.isScalar = false;
    info.droppedDim = *producerDim;
    info.kind = classifyElementwiseBroadcastKind(*producerDim, rank);
    if (info.kind != ElementwiseBroadcastKind::BatchBroadcast)
      return failure();
    info.droppedDimTiles = droppedDimTiles;
    info.suffixTiles = *suffixTiles;
    analysis.broadcastsByInput[i] = info;
  }

  if (outTileInfo->totalTiles <= 0)
    return failure();
  analysis.outTiles = outTileInfo->totalTiles;

  analysis.inputWaitTiles.assign(numInputs, analysis.outTiles);
  analysis.usedInputs.resize(numInputs);
  analysis.usedInputs.reset();

  for (unsigned i = 0; i < numInputs; ++i) {
    if (!analysis.broadcastsByInput[i])
      continue;
    if (analysis.broadcastsByInput[i]->isScalar) {
      analysis.inputWaitTiles[i] = 1;
      continue;
    }
    int64_t droppedDimTiles = analysis.broadcastsByInput[i]->droppedDimTiles;
    if (droppedDimTiles <= 0 || analysis.outTiles % droppedDimTiles != 0)
      return failure();
    analysis.inputWaitTiles[i] = analysis.outTiles / droppedDimTiles;
  }

  for (auto [idx, input] : llvm::enumerate(op.getDpsInputs())) {
    auto annotatedTiles = getAnnotatedVecTilesFromInput(input);
    if (!annotatedTiles)
      continue;
    if (*annotatedTiles <= 0)
      return failure();
    analysis.inputWaitTiles[idx] = *annotatedTiles;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();
  analysis.yieldValue = yieldOp.getValues().front();

  if (failed(analyzeElementwiseExpr(analysis.yieldValue, op, analysis)))
    return failure();

  return success();
}

static LogicalResult emitElementwiseExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Value> inputCbs, Value tileIdx,
    ArrayRef<std::optional<Value>> broadcastTileIdxByInput,
    ArrayRef<std::optional<Value>> runtimeScalarBitsByInput) {
  auto emitLeaf = [&](Value value, int reg, bool &handled) -> LogicalResult {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx) {
      handled = false;
      return success();
    }

    handled = true;
    Value regVal = i32Const(rewriter, loc, reg);
    if (*inputIdx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[*inputIdx].has_value())
      return failure();

    if (*inputIdx < broadcastTileIdxByInput.size() &&
        broadcastTileIdxByInput[*inputIdx].has_value()) {
      rewriter.create<CopyTileOp>(loc, inputCbs[*inputIdx],
                                  *broadcastTileIdxByInput[*inputIdx], regVal);
      return success();
    }

    rewriter.create<CopyTileOp>(loc, inputCbs[*inputIdx], tileIdx, regVal);
    return success();
  };

  auto emitRhsFirstForOperands = [&](Value lhs, Value rhs) -> bool {
    (void)lhs;
    (void)rhs;
    return false;
  };
  auto lookupRuntimeScalarBits = [&](Value value) -> Value {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx || *inputIdx >= runtimeScalarBitsByInput.size() ||
        !runtimeScalarBitsByInput[*inputIdx].has_value())
      return {};
    return *runtimeScalarBitsByInput[*inputIdx];
  };

  return emitGenericExprToRegImpl(
      exprValue, dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter, loc,
      /*emitInlineInitOps=*/false, /*allowMaximumFOp=*/false, emitLeaf,
      emitRhsFirstForOperands, lookupRuntimeScalarBits);
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  ReduceType reduceType;
  if (bodyHasOp<arith::MaximumFOp>(op)) {
    reduceType = ReduceType::Max;
  } else if (bodyHasOp<arith::AddFOp>(op)) {
    reduceType = ReduceType::Sum;
  } else {
    return failure();
  }

  Value inCb = adaptor.getInputs()[0];
  Value outCb = adaptor.getOutputs()[0];
  Value scaleCb = adaptor.getInputs()[1];
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(scaleCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!inType || !outType || !inType.hasStaticShape() ||
      !outType.hasStaticShape() || inType.getRank() != 3)
    return failure();
  if (outType.getRank() != 2 && outType.getRank() != 3)
    return failure();
  if (outType.getRank() == 3 && outType.getShape()[2] != 1)
    return failure();

  // Rank-3 reduction input uses [batch, m, n] where batch is already
  // tile-domain multiplicity (not an element-domain extent).
  int64_t bTiles = inType.getShape()[0];
  auto mTiles = getTileDim(inType, 1);
  auto nTiles = getTileDim(inType, 2);
  auto outTiles = getNumTilesFromShapedType(outType);
  if (bTiles <= 0 || !mTiles || !nTiles || !outTiles)
    return failure();

  int64_t rows = bTiles * (*mTiles);
  int64_t cols = *nTiles;
  int64_t numTiles = rows * cols;
  if (*outTiles != rows)
    return failure();

  Location loc = op.getLoc();
  Value outTilesV = i32Const(rewriter, loc, *outTiles);
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);

  SmallVector<std::tuple<Value, Value, int64_t>, 2> inputPlans;
  inputPlans.emplace_back(inCb, op.getDpsInputs()[0], numTiles);
  inputPlans.emplace_back(scaleCb, op.getDpsInputs()[1], 1);

  for (const auto &plan : inputPlans) {
    Value inputCb = std::get<0>(plan);
    int64_t waitTiles = std::get<2>(plan);
    emitWaitFrontIfNeeded(rewriter, loc, inputCb, waitTiles, waitState);
  }

  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  Value rowsV = i32Const(rewriter, loc, rows);
  Value colsV = i32Const(rewriter, loc, cols);
  scf::ForOp rowLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, rowsV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    Value rowIdx = rowLoop.getInductionVar();
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<ReduceInitOp>(loc, inCb, scaleCb, outCb, reduceType,
                                  ReduceDim::Row);

    Value rowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, colsV);
    scf::ForOp colLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, colsV, oneI32);
    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value colIdx = colLoop.getInductionVar();
      Value inTile = rewriter.create<arith::AddIOp>(loc, rowOffset, colIdx);
      rewriter.create<ReduceTileOp>(loc, inCb, scaleCb, inTile, zeroI32,
                                    zeroI32, reduceType, ReduceDim::Row);
    }

    rewriter.setInsertionPointAfter(colLoop);
    rewriter.create<ReduceUninitOp>(loc);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, rowIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  //TODO: add pop first, since current output also works as input
  //CBPopFrontOp::create(rewriter, loc, outCb, outTilesV);
  //CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  waitState[outCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static Value resolveSimpleBinaryOperand(const SimpleBinaryTileAnalysis &analysis,
                                        const GenericBodyTileOperand &operand,
                                        ArrayRef<Value> inputCbs, Value outCb) {
  (void)analysis;
  if (operand.isOutput)
    return outCb;
  if (operand.inputIdx >= inputCbs.size())
    return {};
  return inputCbs[operand.inputIdx];
}

static void emitSimpleBinaryTileUnaryInit(ConversionPatternRewriter &rewriter,
                                          Location loc,
                                          SimpleBinaryTileUnaryKind kind) {
  switch (kind) {
  case SimpleBinaryTileUnaryKind::Exp:
    rewriter.create<ExpTileInitOp>(loc);
    return;
  case SimpleBinaryTileUnaryKind::Log:
    rewriter.create<LogTileInitOp>(loc);
    return;
  }
}

static void emitSimpleBinaryTileUnaryOp(ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        SimpleBinaryTileUnaryKind kind,
                                        Value dstReg) {
  switch (kind) {
  case SimpleBinaryTileUnaryKind::Exp:
    rewriter.create<ExpTileOp>(loc, dstReg);
    return;
  case SimpleBinaryTileUnaryKind::Log:
    rewriter.create<LogTileOp>(loc, dstReg);
    return;
  }
}

static LogicalResult rewriteAddInplaceGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  AddInplaceAnalysis analysis;
  if (failed(analyzeAddInplaceGeneric(op, analysis)))
    return failure();

  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(outCb.getType()))
    return failure();

  SmallVector<Value, 2> inputCbs;
  inputCbs.reserve(adaptor.getInputs().size());
  for (Value inputCb : adaptor.getInputs()) {
    Value effectiveInputCb = stripBroadcastBridgeCast(inputCb);
    if (!isa<CBType>(effectiveInputCb.getType()))
      return failure();
    inputCbs.push_back(effectiveInputCb);
  }

  Value accCb = inputCbs[analysis.accInputIdx];
  Value addendCb = inputCbs[analysis.addendInputIdx];
  if (accCb != outCb)
    return failure();

  Location loc = op.getLoc();
  emitWaitFrontIfNeeded(rewriter, loc, accCb, analysis.outTiles, waitState);
  if (addendCb != accCb)
    emitWaitFrontIfNeeded(rewriter, loc, addendCb, analysis.outTiles, waitState);

  rewriter.create<CopyTileInitOp>(loc, addendCb);
  emitPackReconfigDataFormat(rewriter, loc, accCb);
  emitPackReconfigL1Acc(rewriter, loc, 1);

  LogicalResult result =
      emitElementwiseTiles(rewriter, loc, accCb, analysis.outTiles,
                           [&](Value tileIdx) -> LogicalResult {
                             Value zeroI32 = i32Const(rewriter, loc, 0);
                             rewriter.create<CopyTileOp>(loc, addendCb, tileIdx,
                                                         zeroI32);
                             return success();
                           });
  if (failed(result))
    return failure();

  emitPackOverwriteMode(rewriter, loc, accCb);
  Value outTilesV = i32Const(rewriter, loc, analysis.outTiles);
  CBPopFrontOp::create(rewriter, loc, accCb, outTilesV);
  CBReserveBackOp::create(rewriter, loc, accCb, outTilesV);
  CBPushBackOp::create(rewriter, loc, accCb, outTilesV);
  waitState[accCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteSimpleBinaryTileGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getOutputs().size() != 1)
    return failure();

  SimpleBinaryTileAnalysis analysis;
  if (failed(analyzeSimpleBinaryTileGeneric(op, analysis)))
    return failure();

  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(outCb.getType()))
    return failure();

  SmallVector<Value, 4> inputCbs;
  inputCbs.reserve(adaptor.getInputs().size());
  for (Value inputCb : adaptor.getInputs()) {
    Value effectiveInputCb = stripBroadcastBridgeCast(inputCb);
    if (!isa<CBType>(effectiveInputCb.getType()))
      return failure();
    inputCbs.push_back(effectiveInputCb);
  }

  Value lhsCb = resolveSimpleBinaryOperand(analysis, analysis.lhs, inputCbs, outCb);
  Value rhsCb = resolveSimpleBinaryOperand(analysis, analysis.rhs, inputCbs, outCb);
  if (!lhsCb || !rhsCb || !isa<CBType>(lhsCb.getType()) ||
      !isa<CBType>(rhsCb.getType()))
    return failure();

  Location loc = op.getLoc();
  emitWaitFrontIfNeeded(rewriter, loc, lhsCb, analysis.outTiles, waitState);
  if (rhsCb != lhsCb)
    emitWaitFrontIfNeeded(rewriter, loc, rhsCb, analysis.outTiles, waitState);

  bool outAliasesOperand = outCb == lhsCb || outCb == rhsCb;
  if (!outAliasesOperand)
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));

  if (!analysis.unaryTail.empty())
    rewriter.create<InitSFPUOp>(loc, lhsCb, outCb);

  switch (analysis.kind) {
  case SimpleBinaryTileKind::Add:
    rewriter.create<AddTilesInitOp>(loc, lhsCb, rhsCb);
    break;
  case SimpleBinaryTileKind::Mul:
    rewriter.create<MulTilesInitOp>(loc, lhsCb, rhsCb);
    break;
  case SimpleBinaryTileKind::Sub:
    rewriter.create<SubTilesInitOp>(loc, lhsCb, rhsCb);
    break;
  }
  for (SimpleBinaryTileUnaryKind kind : analysis.unaryTail)
    emitSimpleBinaryTileUnaryInit(rewriter, loc, kind);

  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, outCb, analysis.outTiles, [&](Value tileIdx) {
        Value zeroI32 = i32Const(rewriter, loc, 0);
        switch (analysis.kind) {
        case SimpleBinaryTileKind::Add:
          AddTilesOp::create(rewriter, loc, lhsCb, rhsCb, tileIdx, tileIdx,
                             zeroI32);
          break;
        case SimpleBinaryTileKind::Mul:
          MulTilesOp::create(rewriter, loc, lhsCb, rhsCb, tileIdx, tileIdx,
                             zeroI32);
          break;
        case SimpleBinaryTileKind::Sub:
          SubTilesOp::create(rewriter, loc, lhsCb, rhsCb, tileIdx, tileIdx,
                             zeroI32);
          break;
        }
        for (SimpleBinaryTileUnaryKind kind : analysis.unaryTail)
          emitSimpleBinaryTileUnaryOp(rewriter, loc, kind, zeroI32);
        return success();
      });
  if (failed(result))
    return failure();

  if (outAliasesOperand) {
    CBPopFrontOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
  } else {
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
  }
  waitState[outCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static void emitOpaqueCall(ConversionPatternRewriter &rewriter, Location loc,
                           StringRef callee, ValueRange args) {
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange(), callee, nullptr,
                                       nullptr, args);
}

static LogicalResult emitMulBlockBcastColsInplace(
    ConversionPatternRewriter &rewriter, Location loc, Value accCb,
    Value scaleCb, const FusedMulBcastAddAnalysis &analysis,
    llvm::DenseMap<Value, int64_t> &waitState) {
  FailureOr<Value> accLiteral = createEmitCCBLiteral(rewriter, loc, accCb);
  FailureOr<Value> scaleLiteral = createEmitCCBLiteral(rewriter, loc, scaleCb);
  if (failed(accLiteral) || failed(scaleLiteral))
    return failure();

  emitOpaqueCall(rewriter, loc, "mul_bcast_cols_init_short",
                 ValueRange{*accLiteral, *scaleLiteral});
  emitWaitFrontIfNeeded(rewriter, loc, accCb, analysis.outTiles, waitState);
  emitWaitFrontIfNeeded(rewriter, loc, scaleCb, analysis.scaleTiles, waitState);

  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);
  Value rowsV = i32Const(rewriter, loc, analysis.rows);
  Value colsV = i32Const(rewriter, loc, analysis.cols);
  Value dstTilesV =
      i32Const(rewriter, loc, std::min<int64_t>(analysis.cols, 8));

  scf::ForOp rowLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, rowsV, oneI32);
  {
    OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value rowIdx = rowLoop.getInductionVar();

    scf::ForOp colLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, colsV, dstTilesV);
    {
      OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value colBase = colLoop.getInductionVar();
      Value remaining = rewriter.create<arith::SubIOp>(loc, colsV, colBase);
      Value isPartial = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, remaining, dstTilesV);
      Value chunk =
          rewriter.create<arith::SelectOp>(loc, isPartial, remaining, dstTilesV);

      TileRegsAcquireOp::create(rewriter, loc);
      scf::ForOp mulLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, chunk, oneI32);
      {
        OpBuilder::InsertionGuard mulGuard(rewriter);
        rewriter.setInsertionPointToStart(mulLoop.getBody());
        Value dstIdx = mulLoop.getInductionVar();
        emitOpaqueCall(rewriter, loc, "mul_tiles_bcast_cols",
                       ValueRange{*accLiteral, *scaleLiteral, dstIdx, rowIdx,
                                  dstIdx});
      }

      rewriter.setInsertionPointAfter(mulLoop);
      TileRegsCommitOp::create(rewriter, loc);
      CBPopFrontOp::create(rewriter, loc, accCb, chunk);
      CBReserveBackOp::create(rewriter, loc, accCb, chunk);
      TileRegsWaitOp::create(rewriter, loc);

      scf::ForOp packLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, chunk, oneI32);
      {
        OpBuilder::InsertionGuard packGuard(rewriter);
        rewriter.setInsertionPointToStart(packLoop.getBody());
        Value dstIdx = packLoop.getInductionVar();
        PackTileOp::create(rewriter, loc, dstIdx, accCb, dstIdx);
      }

      rewriter.setInsertionPointAfter(packLoop);
      CBPushBackOp::create(rewriter, loc, accCb, chunk);
      TileRegsReleaseOp::create(rewriter, loc);
    }
  }

  waitState[accCb] = 0;
  waitState[scaleCb] = analysis.scaleTiles;
  return success();
}

static LogicalResult rewriteInplaceMulBcastColsGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  FusedMulBcastAddAnalysis analysis;
  if (failed(analyzeInplaceMulBcastColsGeneric(op, analysis)))
    return failure();

  Value accCb = stripBroadcastBridgeCast(adaptor.getInputs()[0]);
  Value scaleCb = stripBroadcastBridgeCast(adaptor.getInputs()[1]);
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(accCb.getType()) || !isa<CBType>(scaleCb.getType()) ||
      !isa<CBType>(outCb.getType()) || accCb != outCb)
    return failure();

  if (failed(emitMulBlockBcastColsInplace(rewriter, op.getLoc(), accCb, scaleCb,
                                          analysis, waitState)))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteFusedMulBcastAddGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 3 || adaptor.getOutputs().size() != 1)
    return failure();

  FusedMulBcastAddAnalysis analysis;
  if (failed(analyzeFusedMulBcastAddGeneric(op, analysis)))
    return failure();

  Value addCb = stripBroadcastBridgeCast(adaptor.getInputs()[0]);
  Value accCb = stripBroadcastBridgeCast(adaptor.getInputs()[1]);
  Value scaleCb = stripBroadcastBridgeCast(adaptor.getInputs()[2]);
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(addCb.getType()) || !isa<CBType>(accCb.getType()) ||
      !isa<CBType>(scaleCb.getType()) || !isa<CBType>(outCb.getType()) ||
      accCb != outCb)
    return failure();

  Location loc = op.getLoc();
  if (failed(emitMulBlockBcastColsInplace(rewriter, loc, accCb, scaleCb,
                                          analysis, waitState)))
    return failure();

  emitWaitFrontIfNeeded(rewriter, loc, addCb, analysis.outTiles, waitState);
  emitWaitFrontIfNeeded(rewriter, loc, accCb, analysis.outTiles, waitState);

  rewriter.create<AddTilesInitOp>(loc, addCb, accCb);
  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, accCb, analysis.outTiles, [&](Value tileIdx) {
        Value zeroI32 = i32Const(rewriter, loc, 0);
        AddTilesOp::create(rewriter, loc, addCb, accCb, tileIdx, tileIdx,
                           zeroI32);
        return success();
      });
  if (failed(result))
    return failure();

  CBPopFrontOp::create(rewriter, loc, accCb,
                       i32Const(rewriter, loc, analysis.outTiles));
  CBReserveBackOp::create(rewriter, loc, accCb,
                          i32Const(rewriter, loc, analysis.outTiles));
  CBPushBackOp::create(rewriter, loc, accCb,
                       i32Const(rewriter, loc, analysis.outTiles));
  waitState[accCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteElementwiseGeneric(linalg::GenericOp op,
                                               linalg::GenericOp::Adaptor adaptor,
                                               ConversionPatternRewriter &rewriter,
                                               llvm::DenseMap<Value, int64_t> &waitState,
                                               std::shared_ptr<CompileArgTracker> tracker) {
  if (adaptor.getOutputs().size() != 1 || adaptor.getInputs().empty())
    return failure();

  ElementwiseAnalysis analysis;
  if (failed(analyzeElementwiseGeneric(op, analysis)))
    return failure();

  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(outCb.getType()))
    return failure();
  SmallVector<Value, 4> inputCbs;
  inputCbs.reserve(adaptor.getInputs().size());
  for (Value inputCb : adaptor.getInputs()) {
    Value effectiveInputCb = stripBroadcastBridgeCast(inputCb);
    if (!isa<CBType>(effectiveInputCb.getType()))
      return failure();
    inputCbs.push_back(effectiveInputCb);
  }
  for (Value inputCb : inputCbs)
    if (!isa<CBType>(inputCb.getType()))
      return failure();

  Location loc = op.getLoc();
  bool outAliasesInput = hasOutputAlias(outCb, inputCbs);
  unsigned numInputs = adaptor.getInputs().size();
  SmallVector<std::optional<Value>, 4> runtimeScalarBitsByInput(numInputs,
                                                                 std::nullopt);
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (tracker && parentFunc) {
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i >= analysis.broadcastsByInput.size() || !analysis.broadcastsByInput[i] ||
          !analysis.broadcastsByInput[i]->isScalar)
        continue;
      if (!analysis.usedInputs.test(i))
        continue;

      auto siteId = getScalarSiteIdForGenericInput(op, i);
      if (!siteId)
        continue;
      ArrayRef<Value> scalarArgs =
          tracker->getScalarRuntimeArgs(parentFunc.getOperation(), *siteId);
      if (scalarArgs.empty())
        continue;

      Value scalarBits;
      if (scalarArgs.size() == 1) {
        scalarBits = scalarArgs.front();
      } else {
        FailureOr<Value> listIndex =
            buildScalarRuntimeListIndex(op, *siteId, rewriter, loc);
        if (failed(listIndex))
          return failure();
        scalarBits = selectScalarRuntimeListElement(rewriter, loc, *listIndex,
                                                    scalarArgs);
      }
      if (!scalarBits)
        continue;
      runtimeScalarBitsByInput[i] = scalarBits;
      analysis.needsBinopWithScalar = true;
    }
  }

  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (!analysis.usedInputs.test(idx))
      continue;
    if (idx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[idx].has_value())
      continue;

    int64_t waitTiles = analysis.inputWaitTiles.empty()
                            ? analysis.outTiles
                            : analysis.inputWaitTiles[idx];
    emitWaitFrontIfNeeded(rewriter, loc, inputCb, waitTiles, waitState);

    if (inputCb == outCb)
      continue;
  }

  if (!outAliasesInput)
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));

  Value inCbForInit = outCb;
  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (analysis.usedInputs.test(idx) &&
        !(idx < runtimeScalarBitsByInput.size() &&
          runtimeScalarBitsByInput[idx].has_value())) {
      inCbForInit = inputCb;
      break;
    }
  }
  rewriter.create<InitSFPUOp>(loc, inCbForInit, outCb);
  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (!analysis.usedInputs.test(idx))
      continue;
    if (idx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[idx].has_value())
      continue;
    rewriter.create<CopyTileInitOp>(loc, inputCb);
  }
  if (analysis.needsBinopWithScalar)
    rewriter.create<BinopWithScalarTileInitOp>(loc);
  if (analysis.needsSubBinary)
    rewriter.create<SubBinaryTilesInitOp>(loc);
  if (analysis.needsAddBinary)
    rewriter.create<AddBinaryTilesInitOp>(loc);
  if (analysis.needsMulBinary)
    rewriter.create<MulBinaryTilesInitOp>(loc);
  if (analysis.needsBinaryMax)
    rewriter.create<BinaryMaxTileInitOp>(loc);
  if (analysis.needsPowBinary) {
    rewriter.create<FillTileInitOp>(loc);
    rewriter.create<PowBinaryTilesInitOp>(loc);
  }
  if (analysis.needsRecip)
    rewriter.create<RecipTileInitOp>(loc);
  if (analysis.needsLog)
    rewriter.create<LogTileInitOp>(loc);
  if (analysis.needsExp)
    rewriter.create<ExpTileInitOp>(loc);

  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, outCb, analysis.outTiles, [&](Value tileIdx) -> LogicalResult {
        SmallVector<std::optional<Value>, 4> broadcastTileIdxByInput(
            numInputs, std::nullopt);
        for (unsigned i = 0; i < numInputs; ++i) {
          if (i >= analysis.broadcastsByInput.size() || !analysis.broadcastsByInput[i])
            continue;
          const ElementwiseBroadcastInfo &broadcastInfo =
              *analysis.broadcastsByInput[i];
          if (broadcastInfo.isScalar) {
            broadcastTileIdxByInput[i] = i32Const(rewriter, loc, 0);
            continue;
          }
          Value suffixTiles = i32Const(rewriter, loc, broadcastInfo.suffixTiles);
          Value droppedDimSpan = i32Const(
              rewriter, loc,
              broadcastInfo.droppedDimTiles * broadcastInfo.suffixTiles);
          Value prefix = rewriter.create<arith::DivSIOp>(loc, tileIdx, droppedDimSpan);
          if (broadcastInfo.suffixTiles == 1) {
            broadcastTileIdxByInput[i] = prefix;
          } else {
            Value suffixOffset =
                rewriter.create<arith::RemSIOp>(loc, tileIdx, suffixTiles);
            Value prefixBase =
                rewriter.create<arith::MulIOp>(loc, prefix, suffixTiles);
            broadcastTileIdxByInput[i] =
                rewriter.create<arith::AddIOp>(loc, prefixBase, suffixOffset);
          }
        }
        return emitElementwiseExprToReg(
            analysis.yieldValue, /*dstReg=*/0, /*tmpRegA=*/1, /*tmpRegB=*/2, op,
            adaptor, rewriter, loc, inputCbs, tileIdx, broadcastTileIdxByInput,
            runtimeScalarBitsByInput);
      });
  if (failed(result))
    return failure();

  if (outAliasesInput) {
    //release the input cb first for later usage of output cb
    CBPopFrontOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    waitState[outCb] = 0;
  } else {
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    waitState[outCb] = 0;
  }

  rewriter.eraseOp(op);
  return success();
}

enum class BatchMatmulPackMode {
  Overwrite,
  PackL1,
};

struct BatchMatmulLoweringConfig {
  Operation *op = nullptr;
  Value in0Cb;
  Value in1Cb;
  Value outCb;
  ShapedType lhsShapedType;
  ShapedType rhsShapedType;
  ShapedType outShapedType;
  Value originalOut;
  BatchMatmulPackMode mode = BatchMatmulPackMode::Overwrite;
};

static void emitPackL1Prologue(
    const BatchMatmulLoweringConfig &config,
    std::optional<KSplitLoopInfo> &kLoopInfo, Value totalOutTiles,
    ConversionPatternRewriter &rewriter) {
  Location loc = config.op->getLoc();
  Operation *initAnchor =
      kLoopInfo ? kLoopInfo->loop.getOperation() : config.op;
  bool hasInitialOutTiles = hasKnownZeroFillInitializerBefore(
      initAnchor, config.originalOut, config.outCb);

  if (kLoopInfo) {
    Value isFirstBlock = cmpBlockIv(rewriter, loc, *kLoopInfo,
                                    arith::CmpIPredicate::eq,
                                    kLoopInfo->lower);
    scf::IfOp setupIf =
        rewriter.create<scf::IfOp>(loc, isFirstBlock,
                                   /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&setupIf.getThenRegion().front());
      if (hasInitialOutTiles) {
        CBWaitFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
        CBPopFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
      }
      emitPackOverwriteMode(rewriter, loc, config.outCb);

      rewriter.setInsertionPointToStart(&setupIf.getElseRegion().front());
      Value isSecondBlock =
          cmpBlockIv(rewriter, loc, *kLoopInfo, arith::CmpIPredicate::eq,
                     kLoopInfo->lower + kLoopInfo->step);
      scf::IfOp secondIf =
          rewriter.create<scf::IfOp>(loc, isSecondBlock,
                                     /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard secondGuard(rewriter);
        rewriter.setInsertionPointToStart(&secondIf.getThenRegion().front());
        emitPackReconfigDataFormat(rewriter, loc, config.outCb);
        emitPackReconfigL1Acc(rewriter, loc, 1);
      }
    }
    rewriter.setInsertionPointAfter(setupIf);
    return;
  }

  if (hasInitialOutTiles) {
    CBWaitFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
    CBPopFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
  }
  emitPackOverwriteMode(rewriter, loc, config.outCb);
}

static void emitPackL1Epilogue(
    const BatchMatmulLoweringConfig &config,
    std::optional<KSplitLoopInfo> &kLoopInfo, Value totalOutTiles,
    ConversionPatternRewriter &rewriter) {
  Location loc = config.op->getLoc();
  if (kLoopInfo) {
    Value isNotLastBlock =
        cmpBlockIv(rewriter, loc, *kLoopInfo, arith::CmpIPredicate::ne,
                   kLoopInfo->upper - kLoopInfo->step);
    scf::IfOp finishIf =
        rewriter.create<scf::IfOp>(loc, isNotLastBlock,
                                   /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&finishIf.getThenRegion().front());
      CBWaitFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
      CBPopFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);

      rewriter.setInsertionPointToStart(&finishIf.getElseRegion().front());
      emitPackOverwriteMode(rewriter, loc, config.outCb);
    }
    rewriter.setInsertionPointAfter(finishIf);
    return;
  }

  emitPackOverwriteMode(rewriter, loc, config.outCb);
}

static FailureOr<::loom::CopyOp>
findAnnotatedMatmulStreamBridge(Operation *op, Value matmulOut) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return op->emitOpError()
           << "streamed matmul output requires parent func";

  Value outBase = stripViewLikeWrappers(matmulOut);
  SmallVector<::loom::CopyOp, 2> bridges;
  func.walk([&](::loom::CopyOp copyOp) {
    if (!copyOp->hasAttr(kMatmulStreamBridgeAttrName))
      return;
    if (stripViewLikeWrappers(copyOp.getSource()) == outBase)
      bridges.push_back(copyOp);
  });

  if (bridges.size() != 1) {
    return op->emitOpError()
           << "expected exactly one annotated matmul stream bridge, found "
           << bridges.size();
  }
  return bridges.front();
}

static FailureOr<Value> getMatmulStreamOutputCb(
    Operation *op, Value matmulOut, ConversionPatternRewriter &rewriter) {
  FailureOr<::loom::CopyOp> bridge =
      findAnnotatedMatmulStreamBridge(op, matmulOut);
  if (failed(bridge))
    return failure();

  Value streamCb = rewriter.getRemappedValue(bridge->getDestination());
  if (!streamCb)
    return op->emitOpError()
           << "failed to resolve converted matmul stream output CB";
  if (!isa<CBType>(streamCb.getType()))
    return op->emitOpError()
           << "matmul stream output destination did not convert to CB type";
  return streamCb;
}

static void emitPackL1FinalStreamEpilogue(
    const BatchMatmulLoweringConfig &config, KSplitLoopInfo kLoopInfo,
    Value totalOutTiles, ConversionPatternRewriter &rewriter) {
  Location loc = config.op->getLoc();
  Value isLastBlock = cmpBlockIv(rewriter, loc, kLoopInfo,
                                 arith::CmpIPredicate::eq,
                                 kLoopInfo.upper - kLoopInfo.step);
  scf::IfOp finishIf =
      rewriter.create<scf::IfOp>(loc, isLastBlock, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&finishIf.getThenRegion().front());
    emitPackOverwriteMode(rewriter, loc, config.outCb);

    rewriter.setInsertionPointToStart(&finishIf.getElseRegion().front());
    Value shouldPopAccumulator =
        cmpBlockIv(rewriter, loc, kLoopInfo, arith::CmpIPredicate::ne,
                   kLoopInfo.upper - 2 * kLoopInfo.step);
    scf::IfOp popIf =
        rewriter.create<scf::IfOp>(loc, shouldPopAccumulator,
                                   /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard popGuard(rewriter);
      rewriter.setInsertionPointToStart(&popIf.getThenRegion().front());
      CBWaitFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
      CBPopFrontOp::create(rewriter, loc, config.outCb, totalOutTiles);
    }
  }
  rewriter.setInsertionPointAfter(finishIf);
}

static LogicalResult lowerBatchMatmul(
    const BatchMatmulLoweringConfig &config,
    ConversionPatternRewriter &rewriter) {
  Operation *op = config.op;
  Value in0Cb = config.in0Cb;
  Value in1Cb = config.in1Cb;
  Value outCb = config.outCb;
  ShapedType lhsShapedType = config.lhsShapedType;
  ShapedType rhsShapedType = config.rhsShapedType;
  ShapedType outShapedType = config.outShapedType;

  if (!op || !isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  std::optional<MatmulTileInfo> tileInfo =
      getMatmulTileInfo(lhsShapedType, rhsShapedType, outShapedType);
  if (!tileInfo || tileInfo->batchSize <= 0)
    return failure();

  if (lhsShapedType.getRank() != 3 || rhsShapedType.getRank() != 3 ||
      outShapedType.getRank() != 3)
    return failure();

  Location loc = op->getLoc();
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);
  Value transpose = zeroI32;
  Value ntDim = i32Const(rewriter, loc, tileInfo->nt);
  Value ktDim = i32Const(rewriter, loc, tileInfo->kt);

  int64_t dstCapacity = getDstCapacityTiles(outShapedType);
  FailureOr<MatmulSubblockInfo> subblock = getMatmulSubblockAttrs(op);
  if (failed(subblock))
    return op->emitOpError()
           << "missing precomputed matmul subblock attributes";
  if (subblock->rt * subblock->ct > dstCapacity)
    return op->emitOpError()
           << "precomputed matmul subblock uses "
           << (subblock->rt * subblock->ct)
           << " DST tiles, exceeding capacity " << dstCapacity;

  Value totalOutTiles = i32Const(rewriter, loc, tileInfo->outTilesTotal);
  std::optional<KSplitLoopInfo> kLoopInfo;
  if (config.mode == BatchMatmulPackMode::PackL1) {
    kLoopInfo = getEnclosingStaticKSplitLoop(op);
    emitPackL1Prologue(config, kLoopInfo, totalOutTiles, rewriter);
  }

  bool hasStreamAttr = op->hasAttr(kMatmulStreamOutputSubblocksAttrName);
  bool streamOverwriteSubblocks =
      config.mode == BatchMatmulPackMode::Overwrite && hasStreamAttr;
  bool streamPackL1FinalPacks =
      config.mode == BatchMatmulPackMode::PackL1 && kLoopInfo &&
      op->hasAttr(kMatmulStreamFinalPackAttrName);
  Value streamOutCb;
  if (streamPackL1FinalPacks) {
    FailureOr<Value> resolvedStreamOutCb =
        getMatmulStreamOutputCb(op, config.originalOut, rewriter);
    if (failed(resolvedStreamOutCb))
      return failure();
    streamOutCb = *resolvedStreamOutCb;
  }
  if ((streamOverwriteSubblocks || streamPackL1FinalPacks) &&
      tileInfo->batchSize != 1)
    return op->emitOpError()
           << "streamed matmul output only supports batch size 1";

  Value in0TileCount = i32Const(rewriter, loc, tileInfo->in0TilesTotal);
  Value in1TileCount = i32Const(rewriter, loc, tileInfo->in1TilesTotal);
  Value in0BatchStride = i32Const(rewriter, loc, tileInfo->in0TilesPerBatch);
  Value in1BatchStride = i32Const(rewriter, loc, tileInfo->in1TilesPerBatch);
  Value outBatchStride = i32Const(rewriter, loc, tileInfo->outTilesPerBatch);
  Value batchCount = i32Const(rewriter, loc, tileInfo->batchSize);

  auto minI32 = [&](Value lhs, Value rhs) -> Value {
    Value takeLhs = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhs, rhs);
    return rewriter.create<arith::SelectOp>(loc, takeLhs, lhs, rhs);
  };

  Value rtTotal = i32Const(rewriter, loc, tileInfo->rt);
  Value ctTotal = i32Const(rewriter, loc, tileInfo->ct);

  auto emitMatmulBody = [&](Value targetCb, bool streamChunks,
                            bool seedFromAccumulator) -> LogicalResult {
    if (in0Cb == in1Cb) {
      int64_t sharedTiles =
          std::max(tileInfo->in0TilesTotal, tileInfo->in1TilesTotal);
      CBWaitFrontOp::create(rewriter, loc, in0Cb,
                            i32Const(rewriter, loc, sharedTiles));
    } else {
      CBWaitFrontOp::create(rewriter, loc, in0Cb, in0TileCount);
      CBWaitFrontOp::create(rewriter, loc, in1Cb, in1TileCount);
    }

    if (!streamChunks)
      CBReserveBackOp::create(rewriter, loc, targetCb, totalOutTiles);

    scf::ForOp batchLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, batchCount, oneI32);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    Value batchIdx = batchLoop.getInductionVar();
    Value in0BatchBase =
        rewriter.create<arith::MulIOp>(loc, batchIdx, in0BatchStride);
    Value in1BatchBase =
        rewriter.create<arith::MulIOp>(loc, batchIdx, in1BatchStride);
    Value outBatchBase =
        rewriter.create<arith::MulIOp>(loc, batchIdx, outBatchStride);

    auto emitAccumulatorSeed = [&](Value rowBase, Value colBase, Value actualRt,
                                   Value actualCt) {
      emitPackOverwriteMode(rewriter, loc, targetCb);
      rewriter.create<CopyTileInitOp>(loc, outCb);

      scf::ForOp seedRowLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, actualRt, oneI32);
      {
        OpBuilder::InsertionGuard seedRowGuard(rewriter);
        rewriter.setInsertionPointToStart(seedRowLoop.getBody());
        Value localRow = seedRowLoop.getInductionVar();
        Value globalRow = rewriter.create<arith::AddIOp>(loc, rowBase, localRow);
        Value srcRowBase = rewriter.create<arith::MulIOp>(loc, globalRow, ntDim);
        Value dstRowBase =
            rewriter.create<arith::MulIOp>(loc, localRow, actualCt);

        scf::ForOp seedColLoop =
            scf::ForOp::create(rewriter, loc, zeroI32, actualCt, oneI32);
        {
          OpBuilder::InsertionGuard seedColGuard(rewriter);
          rewriter.setInsertionPointToStart(seedColLoop.getBody());
          Value localCol = seedColLoop.getInductionVar();
          Value globalCol =
              rewriter.create<arith::AddIOp>(loc, colBase, localCol);
          Value srcTile = rewriter.create<arith::AddIOp>(
              loc, outBatchBase,
              rewriter.create<arith::AddIOp>(loc, srcRowBase, globalCol));
          Value dstTile =
              rewriter.create<arith::AddIOp>(loc, dstRowBase, localCol);

          TileRegsAcquireOp::create(rewriter, loc);
          rewriter.create<CopyTileOp>(loc, outCb, srcTile, zeroI32);
          TileRegsCommitOp::create(rewriter, loc);
          TileRegsWaitOp::create(rewriter, loc);
          PackTileOp::create(rewriter, loc, zeroI32, targetCb, dstTile);
          TileRegsReleaseOp::create(rewriter, loc);
        }
      }

      rewriter.setInsertionPointAfter(seedRowLoop);
      emitPackReconfigDataFormat(rewriter, loc, targetCb);
      emitPackReconfigL1Acc(rewriter, loc, 1);
    };

    auto emitSubblock = [&](Value rowBase, Value colBase, Value actualRt,
                            Value actualCt, int64_t maxRt,
                            int64_t maxCt) -> LogicalResult {
      if (maxRt * maxCt > dstCapacity)
        return op->emitOpError()
               << "matmul subblock uses " << (maxRt * maxCt)
               << " DST tiles, exceeding capacity " << dstCapacity;

      Value in0RowOffset = rewriter.create<arith::MulIOp>(loc, rowBase, ktDim);
      Value in0TileIdx =
          rewriter.create<arith::AddIOp>(loc, in0BatchBase, in0RowOffset);
      Value in1TileIdx =
          rewriter.create<arith::AddIOp>(loc, in1BatchBase, colBase);

      Value chunkTiles;
      if (streamChunks) {
        chunkTiles = rewriter.create<arith::MulIOp>(loc, actualRt, actualCt);
        CBReserveBackOp::create(rewriter, loc, targetCb, chunkTiles);
      }
      if (seedFromAccumulator)
        emitAccumulatorSeed(rowBase, colBase, actualRt, actualCt);

      rewriter.create<MatmulBlockInitShortOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, transpose, actualCt, actualRt, ktDim});
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<ExperimentalMatmulBlockOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, zeroI32, transpose,
                     actualCt, actualRt, ktDim, ntDim});
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);

      scf::ForOp packRowLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, actualRt, oneI32);
      {
        OpBuilder::InsertionGuard rowGuard(rewriter);
        rewriter.setInsertionPointToStart(packRowLoop.getBody());
        Value localRow = packRowLoop.getInductionVar();
        Value dstRowBase =
            rewriter.create<arith::MulIOp>(loc, localRow, actualCt);
        Value outRowBase;
        if (!streamChunks) {
          Value globalRow =
              rewriter.create<arith::AddIOp>(loc, rowBase, localRow);
          Value outRowOffset =
              rewriter.create<arith::MulIOp>(loc, globalRow, ntDim);
          outRowBase =
              rewriter.create<arith::AddIOp>(loc, outBatchBase, outRowOffset);
        }

        scf::ForOp packColLoop =
            scf::ForOp::create(rewriter, loc, zeroI32, actualCt, oneI32);
        OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(packColLoop.getBody());
        Value localCol = packColLoop.getInductionVar();
        Value dstIdx =
            rewriter.create<arith::AddIOp>(loc, dstRowBase, localCol);
        Value outTileIdx = dstIdx;
        if (!streamChunks) {
          Value outCol = rewriter.create<arith::AddIOp>(loc, colBase, localCol);
          outTileIdx = rewriter.create<arith::AddIOp>(loc, outRowBase, outCol);
        }
        PackTileOp::create(rewriter, loc, dstIdx, targetCb, outTileIdx);
      }

      rewriter.setInsertionPointAfter(packRowLoop);
      TileRegsReleaseOp::create(rewriter, loc);
      if (streamChunks)
        CBPushBackOp::create(rewriter, loc, targetCb, chunkTiles);
      return success();
    };

    if (tileInfo->rt * tileInfo->ct <= dstCapacity) {
      if (failed(emitSubblock(zeroI32, zeroI32, rtTotal, ctTotal, tileInfo->rt,
                              tileInfo->ct)))
        return failure();
    } else {
      Value rowStep = i32Const(rewriter, loc, subblock->rt);
      Value colStep = i32Const(rewriter, loc, subblock->ct);
      scf::ForOp rowLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, rtTotal, rowStep);
      {
        OpBuilder::InsertionGuard rowGuard(rewriter);
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        Value rowBase = rowLoop.getInductionVar();
        Value remainingRt =
            rewriter.create<arith::SubIOp>(loc, rtTotal, rowBase);
        Value actualRt = minI32(remainingRt, rowStep);

        scf::ForOp colLoop =
            scf::ForOp::create(rewriter, loc, zeroI32, ctTotal, colStep);
        OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value colBase = colLoop.getInductionVar();
        Value remainingCt =
            rewriter.create<arith::SubIOp>(loc, ctTotal, colBase);
        Value actualCt = minI32(remainingCt, colStep);
        if (failed(emitSubblock(rowBase, colBase, actualRt, actualCt,
                                subblock->rt, subblock->ct)))
          return failure();
      }
      rewriter.setInsertionPointAfter(rowLoop);
    }

    rewriter.setInsertionPointAfter(batchLoop);
    if (!streamChunks)
      CBPushBackOp::create(rewriter, loc, targetCb, totalOutTiles);
    return success();
  };

  if (streamPackL1FinalPacks) {
    Value isLastBlock = cmpBlockIv(rewriter, loc, *kLoopInfo,
                                   arith::CmpIPredicate::eq,
                                   kLoopInfo->upper - kLoopInfo->step);
    scf::IfOp finalPackIf =
        rewriter.create<scf::IfOp>(loc, isLastBlock,
                                   /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&finalPackIf.getThenRegion().front());
      CBWaitFrontOp::create(rewriter, loc, outCb, totalOutTiles);
      if (failed(emitMatmulBody(streamOutCb, /*streamChunks=*/true,
                                /*seedFromAccumulator=*/true)))
        return failure();

      rewriter.setInsertionPointToStart(&finalPackIf.getElseRegion().front());
      if (failed(emitMatmulBody(outCb, /*streamChunks=*/false,
                                /*seedFromAccumulator=*/false)))
        return failure();
    }
    rewriter.setInsertionPointAfter(finalPackIf);
  } else {
    if (failed(emitMatmulBody(outCb, streamOverwriteSubblocks,
                              /*seedFromAccumulator=*/false)))
      return failure();
  }

  if (config.mode == BatchMatmulPackMode::PackL1) {
    if (streamPackL1FinalPacks)
      emitPackL1FinalStreamEpilogue(config, *kLoopInfo, totalOutTiles,
                                    rewriter);
    else
      emitPackL1Epilogue(config, kLoopInfo, totalOutTiles, rewriter);
  }
  return success();
}

/**
 * @brief Lower `linalg.batch_matmul` through pack L1 accumulation.
 */
class ConvertLinalgBatchMatmulOp
    : public OpConversionPattern<linalg::BatchMatmulOp> {
public:
  using OpConversionPattern<linalg::BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
      return failure();

    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    BatchMatmulLoweringConfig config{
        op.getOperation(),
        adaptor.getInputs()[0],
        adaptor.getInputs()[1],
        adaptor.getOutputs()[0],
        lhsShapedType,
        rhsShapedType,
        outShapedType,
        op.getOutputs()[0],
        BatchMatmulPackMode::PackL1};
    if (failed(lowerBatchMatmul(config, rewriter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Lower `loom.batch_matmul` as overwrite `C = A * B`.
 */
class ConvertLoomBatchMatmulOp
    : public OpConversionPattern<::loom::BatchMatmulOp> {
public:
  using OpConversionPattern<::loom::BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsShapedType = dyn_cast<ShapedType>(op.getLhs().getType());
    auto rhsShapedType = dyn_cast<ShapedType>(op.getRhs().getType());
    auto outShapedType = dyn_cast<ShapedType>(op.getOuts().getType());
    BatchMatmulLoweringConfig config{
        op.getOperation(),
        adaptor.getLhs(),
        adaptor.getRhs(),
        adaptor.getOuts(),
        lhsShapedType,
        rhsShapedType,
        outShapedType,
        op.getOuts(),
        BatchMatmulPackMode::Overwrite};
    if (failed(lowerBatchMatmul(config, rewriter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgFillOp : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isNextUseDpsInitOfSameOutput(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
      return failure();

    Value outCb = adaptor.getOutputs()[0];
    if (!isa<CBType>(outCb.getType()))
      return failure();

    auto numTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
    if (!numTiles)
      return failure();

    Location loc = op.getLoc();
    Value fillValue = adaptor.getInputs()[0];
    if (!fillValue.getType().isF32()) {
      if (fillValue.getType().isIntOrIndex()) {
        fillValue =
            rewriter.create<arith::SIToFPOp>(loc, rewriter.getF32Type(), fillValue);
      } else if (llvm::isa<FloatType>(fillValue.getType())) {
        fillValue =
            rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), fillValue);
      } else {
        return failure();
      }
    }

    Value zeroI32 = i32Const(rewriter, loc, 0);
    Value oneI32 = i32Const(rewriter, loc, 1);
    Value numTilesV = i32Const(rewriter, loc, *numTiles);
    CBReserveBackOp::create(rewriter, loc, outCb, numTilesV);
    emitPackOverwriteMode(rewriter, loc, outCb);
    rewriter.create<InitSFPUOp>(loc, outCb, outCb);
    rewriter.create<FillTileInitOp>(loc);

    scf::ForOp fillLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, numTilesV, oneI32);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(fillLoop.getBody());
      Value tileIdx = fillLoop.getInductionVar();
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<FillTileOp>(loc, zeroI32, fillValue);
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      PackTileOp::create(rewriter, loc, zeroI32, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    CBPushBackOp::create(rewriter, loc, outCb, numTilesV);
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgCopyOp : public OpConversionPattern<linalg::CopyOp> {
public:
  using OpConversionPattern<linalg::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLinalgCopy(op))
      return failure();

    if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
      return failure();

    Value inCb = adaptor.getInputs()[0];
    Value outCb = adaptor.getOutputs()[0];
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    auto inTiles = getNumTilesFromShapedType(op.getInputs()[0].getType());
    auto outTiles = getNumTilesFromShapedType(op.getOutputs()[0].getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    if (inCb == outCb) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value tileCount = i32Const(rewriter, loc, *inTiles);

    // linalg.copy produces into the destination CB.  Source consumption is
    // represented by the following loom.semaphore_give in the original IR.
    // clean the output cb first. This works for flash attention but fails for matmul.
    CBPopFrontOp::create(rewriter, loc, outCb, tileCount);
    copyTile(rewriter, loc, inCb, outCb, tileCount, /*popInputCb=*/false);
    CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.eraseOp(op);
    return success();
  }
};

static bool shouldReclaimLoomCopyDestination(::loom::CopyOp op) {
  if (auto attr = op->getAttrOfType<BoolAttr>("reclaim"))
    return attr.getValue();
  return op->hasAttr("reclaim");
}

static std::optional<std::pair<int64_t, int64_t>>
getStaticTilePlane(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() || shaped.getRank() < 2)
    return std::nullopt;
  ArrayRef<int64_t> shape = shaped.getShape();
  auto rt = ceilDiv32(shape[shape.size() - 2]);
  auto ct = ceilDiv32(shape[shape.size() - 1]);
  if (!rt || !ct)
    return std::nullopt;
  return std::make_pair(*rt, *ct);
}

static LogicalResult lowerMatmulStreamBridgeCopy(
    ::loom::CopyOp op, Value inCb, Value outCb,
    ConversionPatternRewriter &rewriter) {
  if (op->hasAttr(kMatmulStreamFinalPackAttrName)) {
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<MatmulSubblockInfo> subblock =
      getMatmulSubblockAttrs(op.getOperation());
  if (failed(subblock))
    return op.emitOpError()
           << "missing precomputed matmul stream subblock attributes";

  auto sourcePlane = getStaticTilePlane(op.getSource().getType());
  auto destPlane = getStaticTilePlane(op.getDestination().getType());
  if (!sourcePlane || !destPlane || *sourcePlane != *destPlane)
    return op.emitOpError()
           << "streamed matmul bridge copy requires matching static tile planes";

  Location loc = op.getLoc();
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);
  Value rtTotal = i32Const(rewriter, loc, sourcePlane->first);
  Value ctTotal = i32Const(rewriter, loc, sourcePlane->second);
  Value totalTiles =
      i32Const(rewriter, loc, sourcePlane->first * sourcePlane->second);
  Value rowStep = i32Const(rewriter, loc, subblock->rt);
  Value colStep = i32Const(rewriter, loc, subblock->ct);

  auto minI32 = [&](Value lhs, Value rhs) -> Value {
    Value takeLhs = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhs, rhs);
    return rewriter.create<arith::SelectOp>(loc, takeLhs, lhs, rhs);
  };

  CBWaitFrontOp::create(rewriter, loc, inCb, totalTiles);
  emitPackOverwriteMode(rewriter, loc, outCb);
  rewriter.create<CopyTileInitOp>(loc, inCb);

  scf::ForOp rowLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, rtTotal, rowStep);
  {
    OpBuilder::InsertionGuard rowGuard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value rowBase = rowLoop.getInductionVar();
    Value remainingRt = rewriter.create<arith::SubIOp>(loc, rtTotal, rowBase);
    Value actualRt = minI32(remainingRt, rowStep);

    scf::ForOp colLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, ctTotal, colStep);
    {
      OpBuilder::InsertionGuard colGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value colBase = colLoop.getInductionVar();
      Value remainingCt = rewriter.create<arith::SubIOp>(loc, ctTotal, colBase);
      Value actualCt = minI32(remainingCt, colStep);
      Value chunkTiles = rewriter.create<arith::MulIOp>(loc, actualRt, actualCt);
      CBReserveBackOp::create(rewriter, loc, outCb, chunkTiles);

      scf::ForOp localRowLoop =
          scf::ForOp::create(rewriter, loc, zeroI32, actualRt, oneI32);
      {
        OpBuilder::InsertionGuard localRowGuard(rewriter);
        rewriter.setInsertionPointToStart(localRowLoop.getBody());
        Value localRow = localRowLoop.getInductionVar();
        Value globalRow = rewriter.create<arith::AddIOp>(loc, rowBase, localRow);
        Value srcRowBase = rewriter.create<arith::MulIOp>(loc, globalRow, ctTotal);
        Value dstRowBase =
            rewriter.create<arith::MulIOp>(loc, localRow, actualCt);

        scf::ForOp localColLoop =
            scf::ForOp::create(rewriter, loc, zeroI32, actualCt, oneI32);
        {
          OpBuilder::InsertionGuard localColGuard(rewriter);
          rewriter.setInsertionPointToStart(localColLoop.getBody());
          Value localCol = localColLoop.getInductionVar();
          Value globalCol =
              rewriter.create<arith::AddIOp>(loc, colBase, localCol);
          Value srcTile =
              rewriter.create<arith::AddIOp>(loc, srcRowBase, globalCol);
          Value dstTile =
              rewriter.create<arith::AddIOp>(loc, dstRowBase, localCol);

          TileRegsAcquireOp::create(rewriter, loc);
          rewriter.create<CopyTileOp>(loc, inCb, srcTile, zeroI32);
          TileRegsCommitOp::create(rewriter, loc);
          TileRegsWaitOp::create(rewriter, loc);
          PackTileOp::create(rewriter, loc, zeroI32, outCb, dstTile);
          TileRegsReleaseOp::create(rewriter, loc);
        }
      }

      rewriter.setInsertionPointAfter(localRowLoop);
      CBPushBackOp::create(rewriter, loc, outCb, chunkTiles);
    }
  }

  rewriter.setInsertionPointAfter(rowLoop);
  rewriter.eraseOp(op);
  return success();
}

class ConvertLoomCopyOp : public OpConversionPattern<::loom::CopyOp> {
public:
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isComputeKernel(op.getOperation()))
      return failure();
    if (op.getSource().getDefiningOp<memref::ReinterpretCastOp>() ||
        op.getDestination().getDefiningOp<memref::ReinterpretCastOp>())
      return failure();

    Location loc = op.getLoc();
    Value inCb = adaptor.getSource();
    Value outCb = adaptor.getDestination();
    if (!inCb || !isa<CBType>(inCb.getType()))
      return op.emitOpError("expected source to be converted to CB type");
    if (!outCb || !isa<CBType>(outCb.getType()))
      return op.emitOpError("expected destination to be converted to CB type");

    if (op->hasAttr(kMatmulStreamBridgeAttrName))
      return lowerMatmulStreamBridgeCopy(op, inCb, outCb, rewriter);

    auto inTiles = getNumTilesFromShapedType(op.getSource().getType());
    auto outTiles = getNumTilesFromShapedType(op.getDestination().getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    if (inCb == outCb) {
      rewriter.eraseOp(op);
      return success();
    }

    Value tileCount = i32Const(rewriter, loc, *inTiles);
    if (shouldReclaimLoomCopyDestination(op))
      CBPopFrontOp::create(rewriter, loc, outCb, tileCount);
    copyTile(rewriter, loc, inCb, outCb, tileCount, /*popInputCb=*/false);
    CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.eraseOp(op);
    return success();
  }
};

static bool has2DSwapPermutation(linalg::TransposeOp op) {
  Attribute permAttr = op->getAttr("permutation");
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(permAttr)) {
    if (dense.size() != 2)
      return false;
    return dense[0] == 1 && dense[1] == 0;
  }
  if (auto arr = dyn_cast_or_null<ArrayAttr>(permAttr)) {
    if (arr.size() != 2)
      return false;
    auto p0 = dyn_cast<IntegerAttr>(arr[0]);
    auto p1 = dyn_cast<IntegerAttr>(arr[1]);
    if (!p0 || !p1)
      return false;
    return p0.getInt() == 1 && p1.getInt() == 0;
  }
  return false;
}

struct WHTransposeTileInfo {
  int64_t heightTiles = 0;
  int64_t widthTiles = 0;
  int64_t totalTiles = 0;
};

static LogicalResult getWHTransposeTileInfo(linalg::TransposeOp op,
                                            WHTransposeTileInfo &info) {
  auto inType = dyn_cast<ShapedType>(op.getInput().getType());
  auto outType = dyn_cast<ShapedType>(op.getInit().getType());
  if (!inType || !outType || !inType.hasStaticShape() ||
      !outType.hasStaticShape())
    return failure();

  if (inType.getRank() != 2 || outType.getRank() != 2)
    return failure();

  int64_t inputH = inType.getDimSize(0);
  int64_t inputW = inType.getDimSize(1);
  int64_t outputH = outType.getDimSize(0);
  int64_t outputW = outType.getDimSize(1);

  if (outputH != inputW || outputW != inputH) {
    return op.emitOpError()
           << "unsupported compute WH transpose: output shape must be [W, H]";
  }

  constexpr int64_t kTileDim = 32;
  if (inputH <= 0 || inputW <= 0 || inputH % kTileDim != 0 ||
      inputW % kTileDim != 0) {
    return op.emitOpError()
           << "unsupported compute WH transpose: input H and W must be "
              "positive multiples of 32";
  }

  info.heightTiles = inputH / kTileDim;
  info.widthTiles = inputW / kTileDim;
  info.totalTiles = info.heightTiles * info.widthTiles;
  return success();
}

static Value getOrCreateTileTypedCB(ConversionPatternRewriter &rewriter,
                                    Location loc, Value cb,
                                    int64_t tileCount,
                                    StringRef nameSuffix) {
  auto cbType = dyn_cast<CBType>(cb.getType());
  if (!cbType)
    return {};
  if (isa<mlir::tt::ttcore::TileType>(cbType.getElementType()))
    return cb;

  Type tileType = mlir::tt::ttcore::TileType::get(cbType.getElementType());
  Type tileCbType = CBType::get(rewriter.getContext(), tileCount, tileType);

  Value cbRoot = cb;
  while (auto cast = cbRoot.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1 || !isa<CBType>(cast.getOperand(0).getType()))
      break;
    cbRoot = cast.getOperand(0);
  }

  auto cbConst = cbRoot.getDefiningOp<GetCompileArgValOp>();
  if (!cbConst) {
    auto cast =
        rewriter.create<UnrealizedConversionCastOp>(loc, tileCbType, cb);
    return cast.getResult(0);
  }

  std::optional<int64_t> cbIndex;
  if (auto valueAttr =
          cbConst->getAttrOfType<IntegerAttr>(kCBConstValueAttrName))
    cbIndex = valueAttr.getInt();
  else
    cbIndex = cbConst.getArgIndex();

  auto parentFunc = cbConst->getParentOfType<func::FuncOp>();
  if (!parentFunc || !cbIndex)
    return {};

  std::string name;
  if (auto nameAttr =
          cbConst->getAttrOfType<StringAttr>(kCBConstNameAttrName)) {
    name = nameAttr.getValue().str();
  } else {
    name = "cb_id_transpose_" + std::to_string(*cbIndex);
  }
  name += "_tile_";
  name += nameSuffix.str();

  FailureOr<Value> tileCb =
      getOrCreateCBConst(loc, rewriter, parentFunc, *cbIndex, name, tileCbType);
  if (failed(tileCb))
    return {};
  return *tileCb;
}

class ConvertLinalgTransposeOp : public OpConversionPattern<linalg::TransposeOp> {
public:
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLinalgTranspose(op))
      return failure();

    Value inCb = adaptor.getInput();
    Value outCb = adaptor.getInit();
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    WHTransposeTileInfo tileInfo;
    if (failed(getWHTransposeTileInfo(op, tileInfo)))
      return failure();

    if (inCb == outCb)
      return op.emitOpError()
             << "unsupported compute WH transpose: in-place transpose is not "
                "supported";

    Location loc = op.getLoc();
    Value zero = i32Const(rewriter, loc, 0);
    Value one = i32Const(rewriter, loc, 1);
    Value heightTiles = i32Const(rewriter, loc, tileInfo.heightTiles);
    Value widthTiles = i32Const(rewriter, loc, tileInfo.widthTiles);
    Value totalTiles = i32Const(rewriter, loc, tileInfo.totalTiles);

    Value tileInCb = getOrCreateTileTypedCB(rewriter, loc, inCb,
                                            tileInfo.totalTiles, "in");
    Value tileOutCb = getOrCreateTileTypedCB(rewriter, loc, outCb,
                                             tileInfo.totalTiles, "out");
    if (!tileInCb || !tileOutCb)
      return failure();

    CBWaitFrontOp::create(rewriter, loc, inCb, totalTiles);
    CBReserveBackOp::create(rewriter, loc, outCb, totalTiles);
    rewriter.create<TransposeInitOp>(loc, tileInCb, tileOutCb);

    scf::ForOp wtLoop =
        scf::ForOp::create(rewriter, loc, zero, widthTiles, one);
    {
      OpBuilder::InsertionGuard wtGuard(rewriter);
      rewriter.setInsertionPointToStart(wtLoop.getBody());
      Value wt = wtLoop.getInductionVar();

      scf::ForOp htLoop =
          scf::ForOp::create(rewriter, loc, zero, heightTiles, one);
      {
        OpBuilder::InsertionGuard htGuard(rewriter);
        rewriter.setInsertionPointToStart(htLoop.getBody());
        Value ht = htLoop.getInductionVar();

        Value inputRowBase =
            arith::MulIOp::create(rewriter, loc, ht, widthTiles);
        Value inputTileIdx =
            arith::AddIOp::create(rewriter, loc, inputRowBase, wt);
        Value outputRowBase =
            arith::MulIOp::create(rewriter, loc, wt, heightTiles);
        Value outputTileIdx =
            arith::AddIOp::create(rewriter, loc, outputRowBase, ht);

        TileRegsAcquireOp::create(rewriter, loc);
        rewriter.create<TransposeTileOp>(loc, tileInCb, inputTileIdx, zero);
        TileRegsCommitOp::create(rewriter, loc);
        TileRegsWaitOp::create(rewriter, loc);
        PackTileOp::create(rewriter, loc, zero, tileOutCb, outputTileIdx);
        TileRegsReleaseOp::create(rewriter, loc);
      }
    }

    CBPushBackOp::create(rewriter, loc, outCb, totalTiles);

    rewriter.eraseOp(op);
    return success();
  }
};


struct ComputeBroadcastPlan {
  int64_t inputTiles = 0;
  int64_t outputTiles = 0;
  unsigned dim = 0;
  std::optional<BcastType> bcastType;
  bool materializesFullOutput = false;
  int64_t sourceSuffixTiles = 1;
  int64_t sourceDimSpan = 1;
};

static FailureOr<ComputeBroadcastPlan>
analyzeComputeBroadcast(::loom::BroadcastOp op) {
  auto inputType = dyn_cast<ShapedType>(op.getIns().getType());
  auto outputType = dyn_cast<ShapedType>(op.getInit().getType());
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape())
    return op.emitOpError()
           << "unsupported compute broadcast: requires static shaped input "
              "and output";

  if (op.getNumResults() > 1)
    return failure();
  if (op.getNumResults() == 1) {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != outputType.getRank())
      return failure();
  }

  auto inputTiles = getNumTilesFromShapedType(op.getIns().getType());
  auto outputTiles = getNumTilesFromShapedType(op.getInit().getType());
  if (!inputTiles || !outputTiles)
    return op.emitOpError()
           << "unsupported compute broadcast: failed to compute input/output "
              "tile counts";

  int64_t dim = op.getDim();
  unsigned rank = outputType.getRank();
  if (dim < 0 || static_cast<unsigned>(dim) >= rank)
    return failure();

  ComputeBroadcastPlan plan;
  plan.inputTiles = *inputTiles;
  plan.outputTiles = *outputTiles;
  plan.dim = static_cast<unsigned>(dim);
  plan.bcastType =
      getUnaryBcastType(classifyElementwiseBroadcastKind(plan.dim, rank));

  if (plan.inputTiles == plan.outputTiles)
    return plan;

  auto inputTileInfo = getElementwiseTileShapeInfo(inputType);
  auto outputTileInfo = getElementwiseTileShapeInfo(outputType);
  if (!inputTileInfo || !outputTileInfo)
    return op.emitOpError()
           << "unsupported compute broadcast: failed to compute tile shapes";

  if (inputType.getRank() == outputType.getRank()) {
    if (inputType.getShape()[plan.dim] != 1)
      return op.emitOpError()
             << "unsupported compute broadcast: input broadcast dimension "
                "must have logical size 1";
    if (!hasSameRankTileShapeBroadcastingDim(
            inputTileInfo->dimExtents, outputTileInfo->dimExtents, plan.dim))
      return op.emitOpError()
             << "unsupported compute broadcast: input tile shape must match "
                "output tile shape except for a singleton broadcast dimension";
  } else {
    if (plan.dim != 0)
      return op.emitOpError()
             << "unsupported rank-dropping compute broadcast: only dim(0) is "
                "currently supported";
    if (!plan.bcastType)
      return op.emitOpError()
             << "unsupported rank-dropping compute broadcast: dim(0) must be "
                "a row/col broadcast";
    if (inputType.getRank() + 1 != outputType.getRank())
      return op.emitOpError()
             << "unsupported rank-dropping compute broadcast: expected input "
                "rank + 1 to equal output rank";
    if (!hasTileShapeDroppingDim(inputTileInfo->dimExtents,
                                 outputTileInfo->dimExtents, plan.dim))
      return op.emitOpError()
             << "unsupported rank-dropping compute broadcast: input tile "
                "shape must match output tile shape with the broadcast "
                "dimension removed";
  }

  auto suffixTiles = getSuffixTilesAfterDim(outputTileInfo->dimExtents,
                                            plan.dim);
  if (!suffixTiles || *suffixTiles <= 0)
    return op.emitOpError()
           << "unsupported compute broadcast: failed to compute suffix tile "
              "count";

  int64_t broadcastDimTiles = outputTileInfo->dimExtents[plan.dim];
  if (broadcastDimTiles <= 0)
    return op.emitOpError()
           << "unsupported compute broadcast: invalid broadcast tile extent";

  plan.materializesFullOutput = true;
  plan.sourceSuffixTiles = *suffixTiles;
  plan.sourceDimSpan = broadcastDimTiles * plan.sourceSuffixTiles;
  return plan;
}

static Value getBroadcastSourceTileIndex(ConversionPatternRewriter &rewriter,
                                         Location loc, Value outputTileIdx,
                                         const ComputeBroadcastPlan &plan) {
  if (!plan.materializesFullOutput)
    return outputTileIdx;

  Value dimSpan = i32Const(rewriter, loc, plan.sourceDimSpan);
  Value prefix = rewriter.create<arith::DivSIOp>(loc, outputTileIdx, dimSpan);
  if (plan.sourceSuffixTiles == 1)
    return prefix;

  Value suffixTiles = i32Const(rewriter, loc, plan.sourceSuffixTiles);
  Value suffixOffset =
      rewriter.create<arith::RemSIOp>(loc, outputTileIdx, suffixTiles);
  Value prefixBase = rewriter.create<arith::MulIOp>(loc, prefix, suffixTiles);
  return rewriter.create<arith::AddIOp>(loc, prefixBase, suffixOffset);
}

static LogicalResult
replaceLoomBroadcastOp(::loom::BroadcastOp op, Value outCb,
                       const ComputeBroadcastPlan &plan,
                       ConversionPatternRewriter &rewriter,
                       const TypeConverter &typeConverter) {
  if (op.getNumResults() == 0) {
    rewriter.eraseOp(op);
    return success();
  }

  Type convertedResultType = typeConverter.convertType(op->getResult(0).getType());
  if (!convertedResultType)
    return failure();
  if (convertedResultType == outCb.getType()) {
    rewriter.replaceOp(op, ValueRange{outCb});
    return success();
  }

  if (plan.materializesFullOutput)
    return op.emitOpError()
           << "unsupported compute broadcast: result type must match "
              "materialized output CB type";

  auto cast = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), TypeRange{convertedResultType}, outCb);
  cast->setAttr(kBroadcastDimAttrName,
                rewriter.getI64IntegerAttr(plan.dim));
  rewriter.replaceOp(op, cast.getResults());
  return success();
}

class ConvertLoomBroadcastOp : public OpConversionPattern<::loom::BroadcastOp> {
public:
  using OpConversionPattern<::loom::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isComputeKernel(op.getOperation()) || adaptor.getOperands().size() != 2)
      return failure();

    Value inCb = adaptor.getOperands()[0];
    Value outCb = adaptor.getOperands()[1];
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    FailureOr<ComputeBroadcastPlan> planOr = analyzeComputeBroadcast(op);
    if (failed(planOr))
      return failure();
    ComputeBroadcastPlan plan = *planOr;

    if (inCb == outCb) {
      if (plan.materializesFullOutput)
        return op.emitOpError()
               << "unsupported compute broadcast: cannot materialize a "
                  "tile-count-changing broadcast in-place";
      if (plan.bcastType)
        return failure();
      return replaceLoomBroadcastOp(op, outCb, plan, rewriter,
                                    *getTypeConverter());
    }

    Location loc = op.getLoc();
    Value inputTileCount = i32Const(rewriter, loc, plan.inputTiles);
    Value outputTileCount = i32Const(rewriter, loc, plan.outputTiles);
    Value zero = i32Const(rewriter, loc, 0);
    Value one = i32Const(rewriter, loc, 1);

    CBWaitFrontOp::create(rewriter, loc, inCb, inputTileCount);
    CBReserveBackOp::create(rewriter, loc, outCb, outputTileCount);

    if (plan.bcastType)
      rewriter.create<UnaryBcastInitOp>(loc, inCb, outCb, *plan.bcastType);
    else
      rewriter.create<CopyTileInitOp>(loc, inCb);

    scf::ForOp tileLoop =
        scf::ForOp::create(rewriter, loc, zero, outputTileCount, one);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      Value tileIdx = tileLoop.getInductionVar();
      Value sourceTileIdx =
          getBroadcastSourceTileIndex(rewriter, loc, tileIdx, plan);

      TileRegsAcquireOp::create(rewriter, loc);
      if (plan.bcastType) {
        rewriter.create<UnaryBcastTileOp>(loc, inCb, sourceTileIdx, zero,
                                          *plan.bcastType);
      } else {
        rewriter.create<CopyTileOp>(loc, inCb, sourceTileIdx, zero);
      }
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      PackTileOp::create(rewriter, loc, zero, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    CBPushBackOp::create(rewriter, loc, outCb, outputTileCount);
    return replaceLoomBroadcastOp(op, outCb, plan, rewriter,
                                  *getTypeConverter());
  }
};

class ConvertMemrefCollapseShapeOp
    : public OpConversionPattern<memref::CollapseShapeOp> {
public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSrcType());
    auto dstTy = dyn_cast<MemRefType>(op.getResultType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return failure();
    if (srcTy.getNumElements() != dstTy.getNumElements())
      return failure();

    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class ConvertMemrefExpandShapeOp
    : public OpConversionPattern<memref::ExpandShapeOp> {
public:
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSrcType());
    auto dstTy = dyn_cast<MemRefType>(op.getResultType());
    if (!srcTy || !dstTy)
      return failure();

    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class ConvertFlashAttentionGenericOp
    : public OpConversionPattern<linalg::GenericOp> {
public:
  ConvertFlashAttentionGenericOp(TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<linalg::GenericOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    if (parentFunc != activeFunc) {
      waitState.clear();
      activeFunc = parentFunc;
    }

    // Route [reduction, parallel, parallel] through tileGenericOp first.
    if (isTileGenericOp(op))
      return tileGenericOp(op, adaptor, rewriter, waitState);

    if (!mlir::loom::isSupportedFlashAttentionGeneric(op))
      return failure();

    auto kind = classifyFlashAttentionGeneric(op);
    if (!kind)
      return failure();

    switch (*kind) {
    case FlashAttentionGenericKind::Reduction:
      return rewriteReduceGeneric(op, adaptor, rewriter, waitState);
    case FlashAttentionGenericKind::FusedMulBcastAdd:
      return rewriteFusedMulBcastAddGeneric(op, adaptor, rewriter, waitState);
    case FlashAttentionGenericKind::InplaceMulBcastCols:
      return rewriteInplaceMulBcastColsGeneric(op, adaptor, rewriter,
                                               waitState);
    case FlashAttentionGenericKind::AddInplace:
      return rewriteAddInplaceGeneric(op, adaptor, rewriter, waitState);
    case FlashAttentionGenericKind::SimpleBinaryTile:
      return rewriteSimpleBinaryTileGeneric(op, adaptor, rewriter, waitState);
    case FlashAttentionGenericKind::Elementwise:
      return rewriteElementwiseGeneric(op, adaptor, rewriter, waitState,
                                       tracker);
    }

    return failure();
  }

private:
  mutable llvm::DenseMap<Value, int64_t> waitState;
  mutable func::FuncOp activeFunc;
  std::shared_ptr<CompileArgTracker> tracker;
};

} // namespace

bool mlir::loom::isSupportedFlashAttentionGeneric(linalg::GenericOp op) {
  return isComputeKernel(op.getOperation()) &&
         classifyFlashAttentionGeneric(op).has_value();
}

bool mlir::loom::shouldConvertComputeLinalgCopy(linalg::CopyOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return false;

  auto inTiles = getNumTilesFromShapedType(op.getInputs()[0].getType());
  auto outTiles = getNumTilesFromShapedType(op.getOutputs()[0].getType());
  return inTiles && outTiles && *inTiles == *outTiles;
}

bool mlir::loom::shouldConvertComputeLinalgTranspose(linalg::TransposeOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  auto inType = dyn_cast<ShapedType>(op.getInput().getType());
  auto outType = dyn_cast<ShapedType>(op.getInit().getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return false;

  if (inType.getRank() != 2 || outType.getRank() != 2)
    return false;

  if (!has2DSwapPermutation(op))
    return false;

  return true;
}

void mlir::loom::populateComputeOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
  patterns.add<ConvertLinalgCopyOp>(typeConverter, context);
  patterns.add<ConvertLoomCopyOp>(typeConverter, context);
  patterns.add<ConvertLinalgTransposeOp>(typeConverter, context);
  patterns.add<ConvertLoomBroadcastOp>(typeConverter, context);
  patterns.add<ConvertLinalgBatchMatmulOp>(typeConverter, context);
  patterns.add<ConvertLoomBatchMatmulOp>(typeConverter, context);
  patterns.add<ConvertMemrefCollapseShapeOp>(typeConverter, context);
  patterns.add<ConvertMemrefExpandShapeOp>(typeConverter, context);
  patterns.add<ConvertFlashAttentionGenericOp>(typeConverter, context,
                                               std::move(tracker));
}
