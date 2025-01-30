#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"
#include "Dialect/Schedule/IR/ScheduleDialect.h"

#define DEBUG_TYPE "schedule-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::schedule;

/// Use `operand`'s shape information to create an `tensor.empty` op
/// with the exact same shape.
tensor::EmptyOp
schedule::createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                    SmallPtrSet<Operation *, 4> &newOps,
                                    Location loc, Attribute memorySpace) {
  auto tensorType = cast<TensorType>(operand.getType());
  ArrayRef<int64_t> staticShapes = tensorType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      auto dynDim = rewriter.create<tensor::DimOp>(loc, operand, i);
      newOps.insert(dynDim.getOperation());
      dynamicSizes.push_back(dynDim);
    }
  }
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, staticShapes, tensorType.getElementType(), dynamicSizes);
  if (memorySpace)
    emptyOp->setAttr("memorySpace", memorySpace);
  return emptyOp;
}

linalg::CopyOp schedule::createCacheRead(OpBuilder &rewriter, Value operand,
                                        Location loc, Attribute memorySpace) {
  SmallPtrSet<Operation *, 4> newOps;
  auto emptyOp =
      schedule::createEmptyOpWithSameShape(rewriter, operand, newOps, loc, memorySpace);
  auto cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{operand},
                                                  ValueRange{emptyOp});
  newOps.insert(emptyOp);
  newOps.insert(cachedOp);
  operand.replaceAllUsesExcept(cachedOp.getResult(0), newOps);
  return cachedOp;
}

FailureOr<linalg::CopyOp>
schedule::createCacheWrite(OpBuilder &rewriter, OpResult result, 
                          Value cacheWriteTo, Attribute memorySpace) {
  auto definingOp = dyn_cast<linalg::LinalgOp>(result.getOwner());
  if (!definingOp)
    return {};

  Location loc = definingOp->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  auto initOperand =
      definingOp.getDpsInitOperand(result.getResultNumber())->get();

  linalg::CopyOp cachedOp;
  SmallPtrSet<Operation *, 4> exceptions;

  if (cacheWriteTo) {
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %ret = linalg.op ins(...) outs(%init)
    //   linalg.copy ins(%ret) outs(%res)
    rewriter.setInsertionPointAfter(definingOp);
    // TODO:验证cacheWriteTo的形状是否与result相同
    cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{result},
                                               ValueRange{cacheWriteTo});
  } else {
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %dim = tensor.dim %init
    //   %empty = tensor.empty(%dim)
    //   %ret = linalg.op ins(...) outs(%empty)
    //   linalg.copy ins(%ret) outs(%init)
    rewriter.setInsertionPoint(definingOp);
    // for dynamic shape scenario, need to use `initOperand` to create
    // tensor.dim ops
    tensor::EmptyOp emptyOp =
        createEmptyOpWithSameShape(rewriter, initOperand, exceptions, loc, memorySpace);
    exceptions.insert(emptyOp);
    definingOp->replaceUsesOfWith(initOperand, emptyOp);
    rewriter.setInsertionPointAfter(definingOp);
    cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{result},
                                               ValueRange{initOperand});
  }
  exceptions.insert(cachedOp);
  result.replaceAllUsesExcept(cachedOp.getResult(0), exceptions);
  return cachedOp;
}

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheReadOp::apply(TransformRewriter &rewriter,
                   TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    linalg::CopyOp cachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      auto definingOp = opResult.getOwner();
      rewriter.setInsertionPointAfter(definingOp);
      cachedOp = createCacheRead(rewriter, opResult, definingOp->getLoc(), getMemorySpaceAttr());
    } else if (auto blockArgument = dyn_cast_or_null<BlockArgument>(target)) {
      auto insertPoint = &(blockArgument.getParentBlock()->front());
      rewriter.setInsertionPoint(insertPoint);
      cachedOp = createCacheRead(rewriter, blockArgument, insertPoint->getLoc(), getMemorySpaceAttr());
    } else {
      llvm_unreachable("unsupported type");
    }
    cachedOps.push_back(cachedOp.getOperation());
  }
  transformResults.set(cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheReadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheWriteOp::apply(TransformRewriter &rewriter,
                    TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  auto targets = state.getPayloadValues(getTargets());
  auto cacheWriteTo = state.getPayloadValues(getCacheWriteTo());
  for (auto [target, cacheWriteTo] : llvm::zip(targets, cacheWriteTo)){
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    FailureOr<linalg::CopyOp> maybeCachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      maybeCachedOp = createCacheWrite(rewriter, opResult,
                                       dyn_cast_or_null<Value>(cacheWriteTo),
                                       getMemorySpaceAttr());
    } else {
      llvm_unreachable("unsupported type");
    }
    if (failed(maybeCachedOp))
      return DiagnosedSilenceableFailure::definiteFailure();
    cachedOps.push_back((*maybeCachedOp).getOperation());
  }
  transformResults.set(cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheWriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// MarkParallelOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MarkParallelOp::apply(TransformRewriter &rewriter,
                      TransformResults &transformResults,
                      TransformState &state) {
  SmallVector<Operation *> transformedOps;
  
  for (Operation *target : state.getPayloadOps(getTargets())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure(
          "only scf.for operations can be marked as parallel");
    }

    forOp->setAttr("parallel", 
        UnitAttr::get(rewriter.getContext()));
    
    transformedOps.push_back(forOp);
  }

  transformResults.set(cast<OpResult>(getTransformed()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

void MarkParallelOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// MarkVectorizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MarkVectorizeOp::apply(TransformRewriter &rewriter,
                       TransformResults &transformResults,
                       TransformState &state) {
  SmallVector<Operation *> transformedOps;
  
  for (Operation *target : state.getPayloadOps(getTargets())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure(
          "only scf.for operations can be marked for vectorization");
    }

    forOp->setAttr("vectorize", 
        UnitAttr::get(rewriter.getContext()));
    
    transformedOps.push_back(forOp);
  }

  transformResults.set(cast<OpResult>(getTransformed()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

void MarkVectorizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

#define GET_OP_CLASSES
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.cpp.inc"