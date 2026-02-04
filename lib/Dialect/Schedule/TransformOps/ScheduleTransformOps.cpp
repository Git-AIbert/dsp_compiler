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
// ExtendedTileUsingForOp
//===----------------------------------------------------------------------===//

void transform::ExtendedTileUsingForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange loopTypes,
    Value target, ArrayRef<int64_t> staticTileSizes,
    ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  return build(builder, result, loopTypes,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               interchange, scalableSizes);
}

void transform::ExtendedTileUsingForOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes, ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  build(builder, result, target,
        getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
        interchange, scalableSizes);
}

void transform::ExtendedTileUsingForOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  // Loop types are automaticaly splat by the callee, setting up one is
  // enough.
  SmallVector<Type> loopTypes(1, builder.getType<transform::AnyOpType>());
  build(builder, result, loopTypes, target, mixedTileSizes, interchange,
        scalableSizes);
}

void transform::ExtendedTileUsingForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange loopTypes,
    Value target, ArrayRef<OpFoldResult> mixedTileSizes,
    ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  unsigned numExpectedLoops =
      staticTileSizes.size() - llvm::count(staticTileSizes, 0);
  SmallVector<Type> resultTypes;
  resultTypes.reserve(numExpectedLoops);
  assert((loopTypes.size() == 1 || loopTypes.size() == numExpectedLoops) &&
         "expected one loop type or as many as loops");
  if (loopTypes.size() == 1)
    resultTypes.append(numExpectedLoops, loopTypes[0]);
  else
    llvm::append_range(resultTypes, loopTypes);
  SmallVector<bool> expandedScalableSizes(mixedTileSizes.size(), false);
  if (scalableSizes.has_value())
    expandedScalableSizes.assign(scalableSizes->begin(), scalableSizes->end());
  build(builder, result, /*tiled_linalg_op=*/target.getType(),
        /*loops=*/resultTypes,
        /*target=*/target,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizesAttr,
        /*interchange=*/builder.getDenseI64ArrayAttr(interchange),
        /*scalable_sizes=*/expandedScalableSizes);
}

LogicalResult transform::ExtendedTileUsingForOp::verify() {
  if (getMixedSizes().size() != getScalableSizes().size())
    return emitOpError("expected same number of sizes (")
           << getMixedSizes().size() << ") and scalable sizes ("
           << getScalableSizes().size() << ")";
  ArrayRef<int64_t> staticSizes = getStaticSizes();
  unsigned numExpectedLoops = staticSizes.size() - llvm::count(staticSizes, 0);
  if (getLoops().size() != numExpectedLoops)
    return emitOpError("expected number of loops to tile (")
           << numExpectedLoops << ") to match number of `loops` results ("
           << getLoops().size() << ")";
  return success();
}

DiagnosedSilenceableFailure
transform::ExtendedTileUsingForOp::apply(transform::TransformRewriter &rewriter,
                                 TransformResults &transformResults,
                                 TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<SmallVector<Operation *>> dynamicSizeProducers;
  SmallVector<SmallVector<int64_t>> paramSizes;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  paramSizes.reserve(getDynamicSizes().size());
  for (Value transformValue : getDynamicSizes()) {
    if (isa<ParamType>(transformValue.getType())) {
      dynamicSizeProducers.push_back({});
      ArrayRef<Attribute> params = state.getParams(transformValue);
      paramSizes.push_back(
          llvm::to_vector(llvm::map_range(params, [](Attribute attr) {
            return cast<IntegerAttr>(attr).getValue().getSExtValue();
          })));

      if (paramSizes.back().size() != targets.size()) {
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "expected as many parameter values ("
            << dynamicSizeProducers.back().size() << ") as target ops ("
            << targets.size() << ")";
        diag.attachNote(transformValue.getLoc()) << "for this parameter";
        return diag;
      }

      continue;
    }
    paramSizes.push_back({});
    dynamicSizeProducers.push_back(
        llvm::to_vector(state.getPayloadOps(transformValue)));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(transformValue.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          isa<IndexType>(op->getResult(0).getType())) {
        continue;
      }

      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(transformValue.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  auto scalableSizes = getScalableSizes();
  for (auto [i, op] : llvm::enumerate(targets)) {
    auto tilingInterface = dyn_cast<TilingInterface>(op);
    if (!tilingInterface) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "only ops implementing TilingInterface are supported";
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }
    if (tileSizes.size() > tilingInterface.getLoopIteratorTypes().size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "too many tiles provided, expected at most "
          << tilingInterface.getLoopIteratorTypes().size() << " found "
          << tileSizes.size();
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }

    scf::SCFTilingOptions tilingOptions;
    if (tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [](OpBuilder &, Operation *) -> SmallVector<OpFoldResult> {
            return {};
          });
    } else {
      tilingOptions.setTileSizeComputationFunction([&, index = i](OpBuilder &b,
                                                                  Operation *) {
        SmallVector<OpFoldResult> sizes;
        sizes.reserve(tileSizes.size());
        unsigned dynamicIdx = 0;

        for (auto [ofrIdx, ofr] : llvm::enumerate(getMixedSizes())) {
          if (auto attr = llvm::dyn_cast_if_present<Attribute>(ofr)) {
            if (scalableSizes[ofrIdx]) {
              auto val = b.create<arith::ConstantIndexOp>(
                  getLoc(), cast<IntegerAttr>(attr).getInt());
              Value vscale =
                  b.create<vector::VectorScaleOp>(getLoc(), b.getIndexType());
              sizes.push_back(
                  b.create<arith::MulIOp>(getLoc(), val, vscale).getResult());
            } else {
              sizes.push_back(attr);
            }
            continue;
          }
          ArrayRef<Operation *> dynamicSizes = dynamicSizeProducers[dynamicIdx];
          ArrayRef<int64_t> params = paramSizes[dynamicIdx];
          ++dynamicIdx;
          assert((dynamicSizes.empty() ^ params.empty()) &&
                 "expected either dynamic sizes or parameters");
          if (!params.empty()) {
            sizes.push_back(b.getIndexAttr(params[index]));
          } else {
            sizes.push_back(dynamicSizes[index]->getResult(0));
          }
        }
        return sizes;
      });
    }

    tilingOptions.setInterchange(getInterchange());
    FailureOr<scf::SCFTilingResult> maybeTilingResult =
        tileUsingSCF(rewriter, tilingInterface, tilingOptions);
    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    // === 后处理优化：修复 in-place element-wise 操作 ===
    auto originalLinalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (originalLinalgOp && !maybeTilingResult->tiledOps.empty()) {
      
      // 直接遍历 tiled 操作
      for (Operation *tiledOp : maybeTilingResult->tiledOps) {
        auto tiledLinalgOp = dyn_cast<linalg::LinalgOp>(tiledOp);
        if (!tiledLinalgOp)
          continue;
        
        // 检查是否是 element-wise
        // bool isElementwise = originalLinalgOp.getNumLoops() == 
        //                      originalLinalgOp.getNumParallelLoops();
        // if(!isElementwise)
        //   continue;
        if (!linalg::isElementwise(originalLinalgOp))
          continue;
        
        // 查找 in-place 模式：输入 alias 输出
        for (unsigned outIdx = 0; outIdx < originalLinalgOp.getNumDpsInits(); ++outIdx) {
          Value origOutput = originalLinalgOp.getDpsInitOperand(outIdx)->get();
          
          for (unsigned inIdx = 0; inIdx < originalLinalgOp.getNumDpsInputs(); ++inIdx) {
            Value origInput = originalLinalgOp.getDpsInputOperand(inIdx)->get();
            
            // 检查 in-place：输入 == 输出
            if (origInput != origOutput)
              continue;
            
            // 检查索引映射相同
            AffineMap inputMap = originalLinalgOp.getMatchingIndexingMap(
                originalLinalgOp.getDpsInputOperand(inIdx));
            AffineMap outputMap = originalLinalgOp.getMatchingIndexingMap(
                originalLinalgOp.getDpsInitOperand(outIdx));
            
            if (inputMap != outputMap)
              continue;
            
            // 执行替换：直接用输出 slice 替换输入操作数
            Value tiledInput = tiledLinalgOp.getDpsInputOperand(inIdx)->get();
            Value tiledOutput = tiledLinalgOp.getDpsInitOperand(outIdx)->get();

            rewriter.modifyOpInPlace(tiledLinalgOp, [&]() {
              tiledLinalgOp->setOperand(
                  tiledLinalgOp.getDpsInputOperand(inIdx)->getOperandNumber(),
                  tiledOutput);
            });
          }
        }
      }
    }
    // === 优化结束 ===

    rewriter.replaceOp(op, maybeTilingResult->replacements);

    tiled.append(maybeTilingResult->tiledOps);
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(cast<OpResult>(getTiledLinalgOp()), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(cast<OpResult>(getLoops()[en.index()]), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::ExtendedTileUsingForOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

void transform::ExtendedTileUsingForOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicSizesMutable(), effects);
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

//===----------------------------------------------------------------------===//
// FuseElementwiseGenericOps - Helper Functions
//===----------------------------------------------------------------------===//

/// Extract op_label from an operation.
/// For generic ops, use the existing op_label attribute.
/// For named ops (like linalg.add), extract the operation name (e.g., "add").
static std::string extractOpLabel(Operation *op) {
  // Check if op_label attribute already exists
  if (auto labelAttr = op->getAttrOfType<StringAttr>("op_label")) {
    return labelAttr.getValue().str();
  }

  // For named linalg ops, extract the operation name
  StringRef opName = op->getName().getStringRef();

  // Remove "linalg." prefix if present
  if (opName.starts_with("linalg.")) {
    return opName.substr(7).str(); // Extract "add" from "linalg.add"
  }

  return "";
}

/// Ensure an operation is a linalg.generic. If it's a named op,
/// generalize it and set op_label to the operation name.
static FailureOr<linalg::GenericOp>
ensureGenericOp(RewriterBase &rewriter, Operation *op) {
  // Already a generic op
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    return genericOp;
  }

  // Try to generalize if it's a linalg op
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return failure();
  }

  // Get the op_label (either from attribute or operation name)
  std::string opLabel = extractOpLabel(op);

  // Generalize the named op
  rewriter.setInsertionPoint(op);
  FailureOr<linalg::GenericOp> genericOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);

  if (failed(genericOp)) {
    return failure();
  }

  // Set op_label on the generalized operation
  if (!opLabel.empty()) {
    genericOp->getOperation()->setAttr("op_label", rewriter.getStringAttr(opLabel));
  }

  return *genericOp;
}

/// Merge op_label attributes from producer and consumer.
/// Returns "producer_label" + "_" + "consumer_label"
/// If either label is missing, returns the available one.
/// If both are missing, returns empty string.
static std::string mergeOpLabels(Operation *producer, Operation *consumer) {
  std::string producerLabel = extractOpLabel(producer);
  std::string consumerLabel = extractOpLabel(consumer);

  if (!producerLabel.empty() && !consumerLabel.empty()) {
    return producerLabel + "_" + consumerLabel;
  } else if (!producerLabel.empty()) {
    return producerLabel;
  } else {
    return consumerLabel;
  }
}

//===----------------------------------------------------------------------===//
// FuseElementwiseGenericOps - Main Implementation
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure FuseElementwiseGenericOps::apply(
    TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {

  SmallVector<Operation *> fusedOps;

  auto producerOps = state.getPayloadOps(getProducer());
  auto consumerOps = state.getPayloadOps(getConsumer());

  // Validate that we have matching numbers of operations
  if (llvm::range_size(producerOps) != llvm::range_size(consumerOps)) {
    return emitDefiniteFailure(
        "expected same number of producer and consumer operations");
  }

  for (auto [producerOp, consumerOp] : llvm::zip(producerOps, consumerOps)) {

    // Step 1: Ensure both operations are linalg.generic
    // If they are named ops (like linalg.add), generalize them first
    // and automatically set op_label to the operation name
    auto producerGenericOr = ensureGenericOp(rewriter, producerOp);
    if (failed(producerGenericOr)) {
      return emitDefiniteFailure(
          "producer must be a linalg operation that can be generalized");
    }
    linalg::GenericOp producerGeneric = *producerGenericOr;

    auto consumerGenericOr = ensureGenericOp(rewriter, consumerOp);
    if (failed(consumerGenericOr)) {
      return emitDefiniteFailure(
          "consumer must be a linalg operation that can be generalized");
    }
    linalg::GenericOp consumerGeneric = *consumerGenericOr;

    // Step 2: Validate both operations are element-wise
    if (!linalg::isElementwise(producerGeneric)) {
      return emitDefiniteFailure(
          "producer must be element-wise (all parallel iterators, permutation maps)");
    }

    if (!linalg::isElementwise(consumerGeneric)) {
      return emitDefiniteFailure(
          "consumer must be element-wise (all parallel iterators, permutation maps)");
    }

    // Step 3: Find the OpOperand in consumer that uses producer's result
    Value producerResult = producerGeneric->getResult(0);
    OpOperand *fusedOperand = nullptr;

    for (OpOperand &operand : consumerGeneric->getOpOperands()) {
      if (operand.get() == producerResult) {
        fusedOperand = &operand;
        break;
      }
    }

    if (!fusedOperand) {
      return emitDefiniteFailure(
          "consumer must directly use producer's result as one of its inputs");
    }

    // Step 4: Check if fusion is possible using MLIR's validation
    if (!linalg::areElementwiseOpsFusable(fusedOperand)) {
      return emitDefiniteFailure(
          "operations are not fusable according to elementwise fusion preconditions");
    }

    // Step 5: Perform fusion using MLIR's standard implementation
    rewriter.setInsertionPoint(consumerGeneric);

    // Save op_labels before fusion (use the generalized operations)
    std::string mergedLabel = mergeOpLabels(
        producerGeneric.getOperation(), consumerGeneric.getOperation());

    FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
        linalg::fuseElementwiseOps(rewriter, fusedOperand);

    if (failed(fusionResult)) {
      return emitDefiniteFailure("failed to fuse elementwise operations");
    }

    // Step 6: Set the merged op_label attribute on the fused operation
    Operation *fusedOp = fusionResult->fusedOp;
    if (!mergedLabel.empty()) {
      fusedOp->setAttr("op_label", rewriter.getStringAttr(mergedLabel));
    }

    // Step 7: Handle in-place optimization for outs operands
    // If consumer's outs uses producer's result (in-place),
    // replace it with producer's outs for better memory efficiency
    auto fusedGeneric = cast<linalg::GenericOp>(fusedOp);
    Value producerOuts = producerGeneric.getDpsInits()[0];

    // Check and replace output operands that use producer's result
    for (OpOperand &outOperand : fusedGeneric.getDpsInitsMutable()) {
      if (outOperand.get() == producerResult) {
        // Consumer was using producer's result as outs (in-place)
        // Replace with producer's original outs for better in-place behavior
        rewriter.modifyOpInPlace(fusedOp, [&]() {
          outOperand.set(producerOuts);
        });
      }
    }

    // Step 8: Replace uses and erase original operations
    // Replace all external uses of the old values with the new fused results
    for (auto [origValue, replacement] : fusionResult->replacements) {
      rewriter.replaceUsesWithIf(origValue, replacement, [&](OpOperand &use) {
        // Don't replace uses within the consumer op itself (it will be erased)
        return use.getOwner() != consumerGeneric.getOperation();
      });
    }

    // Erase the consumer operation
    // Note: Producer will be removed by DCE if it has no remaining uses
    rewriter.eraseOp(consumerGeneric);

    fusedOps.push_back(fusedOp);
  }

  transformResults.set(cast<OpResult>(getFusedOp()), fusedOps);
  return DiagnosedSilenceableFailure::success();
}

void FuseElementwiseGenericOps::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerMutable(), effects);
  consumesHandle(getConsumerMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

#define GET_OP_CLASSES
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.cpp.inc"