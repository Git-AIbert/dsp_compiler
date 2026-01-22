#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"
#include "Dialect/Schedule/Transforms/CustomCanonicalizationPatterns.h"
#include "ConsumerFusion.h"
#include "FusionValidator.h"
#include "Dialect/Schedule/IR/ScheduleDialect.h"

#define DEBUG_TYPE "schedule-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::schedule;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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
                                        Location loc, Attribute memorySpace,
                                        bool multiBuffer) {
  SmallPtrSet<Operation *, 4> newOps;
  auto emptyOp =
      schedule::createEmptyOpWithSameShape(rewriter, operand, newOps, loc, memorySpace);
  auto cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{operand},
                                                  ValueRange{emptyOp});
  // 如果启用多缓冲，添加multi_buffer属性
  if (multiBuffer) {
    cachedOp->setAttr("multi_buffer", UnitAttr::get(rewriter.getContext()));
  }
  newOps.insert(emptyOp);
  newOps.insert(cachedOp);
  operand.replaceAllUsesExcept(cachedOp.getResult(0), newOps);
  return cachedOp;
}

FailureOr<linalg::CopyOp>
schedule::createCacheWrite(OpBuilder &rewriter, OpResult result, 
                          Value cacheWriteTo, Attribute memorySpace,
                          bool multiBuffer) {
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
  // 如果启用多缓冲，添加multi_buffer属性
  if (multiBuffer) {
    cachedOp->setAttr("multi_buffer", UnitAttr::get(rewriter.getContext()));
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
      cachedOp = createCacheRead(rewriter, opResult, definingOp->getLoc(), 
                                getMemorySpaceAttr(), getMultiBuffer());
    } else if (auto blockArgument = dyn_cast_or_null<BlockArgument>(target)) {
      auto insertPoint = &(blockArgument.getParentBlock()->front());
      rewriter.setInsertionPoint(insertPoint);
      cachedOp = createCacheRead(rewriter, blockArgument, insertPoint->getLoc(), 
                                getMemorySpaceAttr(), getMultiBuffer());
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
  // auto targets = state.getPayloadValues(getTargets());

  // 获取目标值并转换为SmallVector
  SmallVector<Value> targets(state.getPayloadValues(getTargets()).begin(), 
                            state.getPayloadValues(getTargets()).end());
  
  // 检查getCacheWriteTo()是否为空
  SmallVector<Value> cacheWriteToValues;
  if (getCacheWriteTo()) {
    // 获取cacheWriteTo值并转换为SmallVector
    cacheWriteToValues.assign(state.getPayloadValues(getCacheWriteTo()).begin(),
                             state.getPayloadValues(getCacheWriteTo()).end());
    
    // 确保targets和cacheWriteToValues的数量匹配
    if (targets.size() != cacheWriteToValues.size()) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  } else {
    // 如果为空，为每个target创建一个空值
    cacheWriteToValues.resize(targets.size(), Value());
  }

  for (auto [target, cacheWriteTo] : llvm::zip(targets, cacheWriteToValues)){
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    FailureOr<linalg::CopyOp> maybeCachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      maybeCachedOp = createCacheWrite(rewriter, opResult,
                                       dyn_cast_or_null<Value>(cacheWriteTo),
                                       getMemorySpaceAttr(), getMultiBuffer());
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
  if (getCacheWriteTo())
    onlyReadsHandle(getCacheWriteToMutable(), effects);
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

    // forOp->setAttr("parallel", 
    //     UnitAttr::get(rewriter.getContext()));
    forOp->setAttr("num_threads", 
        IntegerAttr::get(rewriter.getI32Type(), getNumThreads()));
    
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
// MarkUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure MarkUnrollOp::apply(TransformRewriter &rewriter,
                      TransformResults &transformResults,
                      TransformState &state) {
  SmallVector<Operation *> transformedOps;
  
  for (Operation *target : state.getPayloadOps(getTargets())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure(
          "only scf.for operations can be marked for unrolling");
    }
    
    forOp->setAttr("unroll_factor", 
        IntegerAttr::get(rewriter.getI32Type(), getUnrollFactor()));
        
    transformedOps.push_back(forOp);
  }

  transformResults.set(cast<OpResult>(getTransformed()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

void MarkUnrollOp::getEffects(
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

//===----------------------------------------------------------------------===//
// FuseConsumerIntoContainingOp
//===----------------------------------------------------------------------===//

// NOTE: findInsertSliceRecursive has been moved to FusionValidator.h/cpp

DiagnosedSilenceableFailure FuseConsumerIntoContainingOp::apply(
    TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  SmallVector<Operation *> fusedConsumerOps;
  SmallVector<Operation *> newContainingOps;

  // Get the consumer operations to fuse
  auto consumerOps = state.getPayloadOps(getConsumerOp());

  // Get the containing operations (loops)
  auto containingOps = state.getPayloadOps(getContainingOp());

  // Validate that we have exactly one containing op
  if (!llvm::hasSingleElement(containingOps)) {
    return emitDefiniteFailure(
        "requires exactly one containing_op handle (got ")
        << llvm::range_size(containingOps) << ")";
  }

  Operation *containingOp = *containingOps.begin();

  // The containing op should be an scf.for loop
  auto targetForOp = dyn_cast<scf::ForOp>(containingOp);
  if (!targetForOp) {
    return emitDefiniteFailure("containing_op must be an scf.for loop");
  }

  for (Operation *consumerOp : consumerOps) {
    // Step 1: Find which loop result the consumer actually uses
    // The consumer might use an outer loop's result, not necessarily targetForOp
    scf::ForOp actualSourceLoop = nullptr;
    Value consumedLoopResult;

    for (OpOperand &operand : consumerOp->getOpOperands()) {
      Value value = operand.get();
      if (auto opResult = dyn_cast<OpResult>(value)) {
        if (auto sourceLoop = dyn_cast<scf::ForOp>(opResult.getOwner())) {
          // Check if this loop contains or is the target loop
          // The source loop should be an ancestor of or equal to targetForOp
          Operation *checkOp = targetForOp.getOperation();
          bool isAncestorOrSelf = (sourceLoop == targetForOp);

          while (!isAncestorOrSelf && checkOp) {
            checkOp = checkOp->getParentOp();
            if (checkOp == sourceLoop.getOperation()) {
              isAncestorOrSelf = true;
              break;
            }
          }

          if (isAncestorOrSelf) {
            actualSourceLoop = sourceLoop;
            consumedLoopResult = value;
            break;
          }
        }
      }
    }

    if (!actualSourceLoop) {
      return emitDefiniteFailure(
          "consumer does not use any result from the specified containing_op "
          "or its ancestor loops");
    }

    // Step 2: Recursively find the actual insert_slice operation
    // This handles multi-layer nested loops by tracing through yield chains
    // IMPORTANT: The insert_slice must be in targetForOp's direct loop body
    FailureOr<tensor::InsertSliceOp> candidateSliceOp =
        findInsertSliceRecursive(consumedLoopResult, actualSourceLoop, targetForOp);

    if (failed(candidateSliceOp)) {
      return emitDefiniteFailure(
          "could not find insert_slice in the target loop's body - "
          "the insert_slice must be directly in the containing_op's loop body, "
          "not in a deeper nested loop");
    }
    
    LLVM_DEBUG({
      // 输出完整module
      if (auto moduleOp = candidateSliceOp->getOperation()->getParentOfType<ModuleOp>()) {
        llvm::dbgs() << "=== Complete Module ===\n" << *moduleOp << "\n";
      }
      llvm::dbgs() << "candidateSliceOp:\n" << candidateSliceOp << "\n";
    });

    // Step 3: Use the found insert_slice for fusion
    rewriter.setInsertionPoint(*candidateSliceOp);
    // Use local implementation of consumer fusion with explicit consumer
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
        localTileAndFuseConsumerOfSlice(rewriter, *candidateSliceOp, consumerOp);

    if (failed(fuseResult)) {
      return emitDefiniteFailure("failed to fuse consumer into containing op");
    }

    // Step 4: Get the fused consumer operation
    Operation *fusedConsumer =
        fuseResult->tiledAndFusedConsumerOperand->getOwner();

    // Step 5: Find the new containing loop
    // tileAndFuseConsumerOfSlice creates a new scf.for loop
    // The fused consumer must be inside the new loop
    Operation *newLoop = fusedConsumer->getParentOfType<scf::ForOp>();
    if (!newLoop) {
      return emitDefiniteFailure("could not find new containing loop after fusion");
    }

    // Collect results
    fusedConsumerOps.push_back(fusedConsumer);
    newContainingOps.push_back(newLoop);
  }

  // Set return results: (fused_consumer, new_containing_op)
  transformResults.set(cast<OpResult>(getFusedConsumer()), fusedConsumerOps);
  transformResults.set(cast<OpResult>(getNewContainingOp()), newContainingOps);
  return DiagnosedSilenceableFailure::success();
}

void FuseConsumerIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Helper functions for element-wise fusion
// NOTE: Validation functions have been moved to FusionValidator.h/cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FuseEltwiseConsumerOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure FuseEltwiseConsumerOp::apply(
    TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  SmallVector<Operation *> fusedConsumerOps;
  SmallVector<Operation *> newContainingOps;

  // Get the consumer operations to fuse
  auto consumerOps = state.getPayloadOps(getConsumerOp());

  // Get the containing operations (loops)
  auto containingOps = state.getPayloadOps(getContainingOp());

  // Validate that we have exactly one containing op
  if (!llvm::hasSingleElement(containingOps)) {
    return emitDefiniteFailure(
        "requires exactly one containing_op handle (got ")
        << llvm::range_size(containingOps) << ")";
  }

  Operation *containingOp = *containingOps.begin();

  // The containing op should be an scf.for loop
  auto targetForOp = dyn_cast<scf::ForOp>(containingOp);
  if (!targetForOp) {
    return emitDefiniteFailure("containing_op must be an scf.for loop");
  }

  for (Operation *consumerOp : consumerOps) {
    // Layer 2: Validate all preconditions using the unified validation entry point
    FailureOr<EltwiseFusionPreconditions> preconditions =
        validateAndPrepareEltwiseFusion(consumerOp, targetForOp);

    if (failed(preconditions)) {
      return emitDefiniteFailure(
          "failed to validate preconditions for element-wise fusion - "
          "consumer must be element-wise, use loop result as init, "
          "and be the only user of the loop result");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "=== FuseEltwiseConsumerOp: Before Fusion ===\n";
      if (auto moduleOp = preconditions->candidateSliceOp.getOperation()
                              ->getParentOfType<ModuleOp>()) {
        llvm::dbgs() << "Complete Module:\n" << *moduleOp << "\n";
      }
      llvm::dbgs() << "candidateSliceOp: " << preconditions->candidateSliceOp << "\n";
      llvm::dbgs() << "consumerOp: " << *consumerOp << "\n";
    });

    // Determine if this is a reduction axis or spatial axis fusion
    // Key insight: Check if the insert_slice is doing in-place updates
    // (inserting into [0, 0] with full sizes = reduction axis)
    // versus extracting/inserting different spatial regions (spatial axis)
    bool isReductionAxis = true;  // Assume reduction unless proven otherwise

    auto insertSliceOp = preconditions->candidateSliceOp;
    SmallVector<OpFoldResult> offsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insertSliceOp.getMixedSizes();
    Value dest = insertSliceOp.getDest();
    auto destType = cast<TensorType>(dest.getType());

    // Check if all offsets are 0 and sizes match destination shape
    // If so, this is an in-place update pattern (reduction axis)
    // Use MLIR's isConstantIntValue and getConstantIntValue helpers
    for (size_t i = 0; i < offsets.size(); ++i) {
      // Check if offset is 0 (using MLIR's isConstantIntValue helper)
      bool offsetIsZero = isConstantIntValue(offsets[i], 0);

      // Check if size matches destination dimension
      bool sizeMatchesDest = false;
      if (!destType.isDynamicDim(i)) {
        int64_t destDimSize = destType.getDimSize(i);
        sizeMatchesDest = isConstantIntValue(sizes[i], destDimSize);
      }

      // If any dimension has non-zero offset or size != dest size,
      // this is NOT a pure in-place update (likely spatial axis)
      if (!offsetIsZero || !sizeMatchesDest) {
        isReductionAxis = false;
        break;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Fusion strategy selection:\n";
      llvm::dbgs() << "  insert_slice offsets: [";
      for (auto offset : offsets) {
        if (auto constVal = getConstantIntValue(offset)) {
          llvm::dbgs() << *constVal << " ";
        } else {
          llvm::dbgs() << "SSA ";
        }
      }
      llvm::dbgs() << "]\n  insert_slice sizes: [";
      for (auto size : sizes) {
        if (auto constVal = getConstantIntValue(size)) {
          llvm::dbgs() << *constVal << " ";
        } else {
          llvm::dbgs() << "SSA ";
        }
      }
      llvm::dbgs() << "]\n  Destination type: " << destType << "\n";
      llvm::dbgs() << "  => " << (isReductionAxis ? "REDUCTION AXIS" : "SPATIAL AXIS")
                   << " fusion\n";
    });

    // Layer 3: Perform fusion using the appropriate strategy
    rewriter.setInsertionPoint(preconditions->candidateSliceOp);
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult;

    if (isReductionAxis) {
      // Use split-reduction strategy for reduction axes
      fuseResult = fuseEltwiseConsumerInPlaceWithSplitReduction(
          rewriter, preconditions->candidateSliceOp, preconditions->consumerOp);
    } else {
      // Use regular in-place fusion for spatial axes
      fuseResult = fuseEltwiseConsumerInPlace(
          rewriter, preconditions->candidateSliceOp, preconditions->consumerOp);
    }

    if (failed(fuseResult)) {
      return emitDefiniteFailure("failed to fuse element-wise consumer into containing op");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "=== FuseEltwiseConsumerOp: After Fusion ===\n";
      if (auto moduleOp = preconditions->candidateSliceOp.getOperation()
                              ->getParentOfType<ModuleOp>()) {
        llvm::dbgs() << "Complete Module:\n" << *moduleOp << "\n";
      }
    });

    // Get the fused consumer operation
    Operation *fusedConsumer =
        fuseResult->tiledAndFusedConsumerOperand->getOwner();

    // The original loop is modified in-place, so it's still the containing loop
    Operation *containingLoop = fusedConsumer->getParentOfType<scf::ForOp>();
    if (!containingLoop) {
      return emitDefiniteFailure("could not find containing loop after fusion");
    }

    // Collect results
    fusedConsumerOps.push_back(fusedConsumer);
    newContainingOps.push_back(containingLoop);
  }

  // Set return results: (fused_consumer, new_containing_op)
  transformResults.set(cast<OpResult>(getFusedConsumer()), fusedConsumerOps);
  transformResults.set(cast<OpResult>(getNewContainingOp()), newContainingOps);
  return DiagnosedSilenceableFailure::success();
}

void FuseEltwiseConsumerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyCustomCanonicalizationPatternsOp
//===----------------------------------------------------------------------===//

void ApplyCustomCanonicalizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  populateCustomCanonicalizationPatterns(patterns, ctx);
}

#define GET_OP_CLASSES
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.cpp.inc"