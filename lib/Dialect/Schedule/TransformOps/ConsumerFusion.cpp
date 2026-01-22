//===- ConsumerFusion.cpp - Local implementation of consumer fusion -------===//
//
// This file contains a local copy of MLIR's tileAndFuseConsumerOfSlice
// implementation, which can be modified to support custom fusion strategies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "consumer-fusion"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper Functions (copied from MLIR's TileUsingInterface.cpp)
//===----------------------------------------------------------------------===//

/// Fetches the OpOperand of the only user (and use) of the value `val` which
/// implements `TilingInterface` and `DestinationStyleOpInterface`. Returns
/// failure otherwise.
static FailureOr<OpOperand *> getConsumerFromUses(Value val,
                                                  Block *containingOpBlock) {
  // Step 1. Check that the value has exactly one use.
  if (!llvm::hasSingleElement(val.getUses()))
    return failure();
  // Step 2. Get uses.
  OpOperand &operand = (*val.getUses().begin());
  Operation *consumerOp = operand.getOwner();
  // TODO: We have to init result of consumer before scf.for, use
  //       DestinationStyleOpInterface to get result shape from init for now.
  //       Add support for other op such as op has InferTypeOpInterface.
  if (!isa<TilingInterface>(consumerOp) ||
      !isa<DestinationStyleOpInterface>(consumerOp))
    return failure();
  if (containingOpBlock != consumerOp->getBlock())
    return failure();
  return &operand;
}

/// A utility function that checks whether the only use of the result of a
/// tensor.insert_slice op is in a scf.yield op.
static LogicalResult
checkAssumptionForFusingConsumer(tensor::InsertSliceOp candidateSliceOp) {
  Value result = candidateSliceOp.getResult();
  Value::use_range uses = result.getUses();
  if (!llvm::hasSingleElement(uses)) {
    LLVM_DEBUG(llvm::dbgs() << "Too many uses of the candidate slice op\n");
    return failure();
  }
  OpOperand &operandUse = (*uses.begin());
  Operation *userOp = operandUse.getOwner();
  if (!isa<scf::YieldOp>(userOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected scf.yield to be the only user, but got -> "
               << (*userOp));
    return failure();
  }
  if (result.getDefiningOp()->getBlock() != userOp->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Expected tensor.insert_slice and scf.yield to "
                               "be in the same block\n");
    return failure();
  }
  return success();
}

/// Fetch the untiled consumer of a scf.for's result which is yielded by a
/// tensor.insert_slice. This function makes the following assumptions :
/// 1.  tensor.insert_slice has scf.yield as its only user.
/// 2.  scf.for's corresponding result has only one use.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(tensor::InsertSliceOp candidateSliceOp) {
  if (failed(checkAssumptionForFusingConsumer(candidateSliceOp)))
    return failure();
  Value sliceResult = candidateSliceOp.getResult();
  // Step 1. Fetch the corresponding output.
  OpOperand &yieldOpOperand = (*sliceResult.getUses().begin());
  unsigned resultNumber = yieldOpOperand.getOperandNumber();
  // Step 2. Check containing op is scf.for.
  Operation *containingOp = candidateSliceOp->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(containingOp);
  if (!forOp)
    return failure();
  Value resultingValue = forOp->getResult(resultNumber);

  return getConsumerFromUses(resultingValue, containingOp->getBlock());
}

/// This utility currently checks whether the loop either :-
/// 1. Yields exactly one result.
/// 2. Has consumer op as its first user and other users to be in the same
/// containing block as that of consumer op's. Currently we clone the loop op
/// right before the consumer op in order to maintain a valid def-use chain.
/// This utility thus helps ensuring that no invalid IR is formed due to the
/// same.
static LogicalResult checkAssumptionForLoop(Operation *loopOp,
                                            Operation *consumerOp) {
  // Check if the loop op yields one result.
  if (loopOp->getNumResults() == 1)
    return success();
  // Check if the consumerOp is the first user of the loopOp and if other users
  // are in the same containing block as that of consumer op's.
  Block *parentBlock = consumerOp->getBlock();
  for (Operation *userOp : loopOp->getUsers()) {
    if (userOp == consumerOp)
      continue;
    if (parentBlock != userOp->getBlock() ||
        !consumerOp->isBeforeInBlock(userOp))
      return failure();
  }
  return success();
}

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty())
    return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// After fusing consumer into scf.for we want to modify the scf.yield operation
/// to reflect the same by returning the values yielded by the tiled consumer.
static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      TilingResult &tilingResult,
                      ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
                      ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
                      ArrayRef<BlockArgument> bbArgs) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  unsigned totalOldResults = oldTerminatorOp->getNumResults();
  unsigned totalTiledResults = tilingResult.tiledOps[0]->getNumResults();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(totalOldResults + totalTiledResults);
  for (auto oldResult : oldTerminatorOp.getResults()) {
    newYieldOperands.push_back(oldResult);
  }
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  Location loc = newForOp.getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult.tiledOps[0]->getResults(), bbArgs,
                       resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    Value newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledResult, bbArg, resultOffset, resultSize, strides);
    newYieldOperands.push_back(newInsertSliceOp);
  }
  rewriter.create<scf::YieldOp>(loc, newYieldOperands);
  rewriter.eraseOp(oldTerminatorOp);
}

/// After fusing consumer into scf.forall we want to yield each of the
/// resultant values by the tiled consumer within scf.forall.in_parallel region.
static void fixTerminatorSCFInParallel(
    RewriterBase &rewriter, scf::ForallOp newForallOp,
    ValueRange tiledConsumerResults,
    ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
    ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
    ArrayRef<BlockArgument> bbArgs) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tiledConsumerResults, bbArgs, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, tiledResult, bbArg, resultOffset, resultSize, strides);
  }
}

//===----------------------------------------------------------------------===//
// Main Fusion Function (local copy that can be modified)
//===----------------------------------------------------------------------===//

/// Local implementation of consumer fusion.
/// Modified version that accepts explicit consumer operation.
///
/// Key modifications:
/// 1. Accepts explicit consumer operation (bypasses single-use requirement)
/// 2. Does not use getUntiledConsumerFromSlice to infer consumer
/// 3. Allows fusion in multi-layer nested loop scenarios
FailureOr<scf::SCFFuseConsumerOfSliceResult>
localTileAndFuseConsumerOfSlice(RewriterBase &rewriter,
                                 Operation *candidateSliceOp,
                                 Operation *consumerOp) {
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();

  bool isInsertSliceOp = isa<tensor::InsertSliceOp>(candidateSliceOp);

  // 1. Find which operand of the consumer uses the loop result
  // Instead of using getUntiledConsumerFromSlice, we directly search
  // the consumer's operands for the one that uses the loop result
  Operation *oldLoopOp = candidateSliceOp->getParentOp();
  if (!isa<scf::ForOp, scf::ForallOp>(oldLoopOp)) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "insert_slice parent is not a loop op");
  }

  // Find the result number that the insert_slice yields
  unsigned resultNumber = 0;
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldLoopOp);
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    // Find which yield operand is the insert_slice result
    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      if (yieldOp.getOperand(i).getDefiningOp() == candidateSliceOp) {
        resultNumber = i;
        break;
      }
    }
  }

  // Find which operand of consumerOp uses the loop result
  OpOperand *consumerOpOperand = nullptr;
  Value loopResult = oldLoopOp->getResult(resultNumber);
  for (OpOperand &operand : consumerOp->getOpOperands()) {
    if (operand.get() == loopResult) {
      consumerOpOperand = &operand;
      break;
    }
  }

  if (!consumerOpOperand) {
    return rewriter.notifyMatchFailure(
        consumerOp, "consumer does not use the loop result from insert_slice");
  }

  unsigned operandNumber = consumerOpOperand->getOperandNumber();

  // Re-acquire loop information for the fusion process
  SmallVector<Value> newOuts;
  Block *oldLoopBody = nullptr;
  unsigned initSize = 0;
  unsigned rank = 1;
  if (isInsertSliceOp) {
    auto forOp = candidateSliceOp->getParentOfType<scf::ForOp>();
    oldLoopOp = forOp;
    llvm::append_range(newOuts, forOp.getInits());
    oldLoopBody = forOp.getBody();
    initSize = forOp.getInits().size();
  } else {
    auto forallOp = candidateSliceOp->getParentOfType<scf::ForallOp>();
    oldLoopOp = forallOp;
    llvm::append_range(newOuts, forallOp.getOutputs());
    oldLoopBody = forallOp.getBody();
    initSize = forallOp.getOutputs().size();
    rank = forallOp.getRank();
  }

  // Skip the checkAssumptionForLoop check since we explicitly provide
  // the consumer operation and can handle multi-use scenarios.
  // The original check is too restrictive for our multi-layer fusion case.
  // if (failed(checkAssumptionForLoop(oldLoopOp, consumerOp))) {
  //   return rewriter.notifyMatchFailure(
  //       oldLoopOp, "containing loop op should either yield just one value or "
  //                  "have the consumer op as its first user");
  // }

  OpBuilder::InsertionGuard g(rewriter);

  // 2. Check consumer is not using scf loop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, oldLoopOp->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  newOuts.append(dpsInits);

  Location loc = oldLoopOp->getLoc();

  // 3. Create new scf loop op.
  rewriter.setInsertionPoint(consumerOp);
  Operation *newLoopOp = nullptr;
  Block *newLoopBody = nullptr;
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldLoopOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newLoopOp = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldLoopOp);
    auto newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    newLoopOp = newForallOp;
    rewriter.eraseOp(newForallOp.getTerminator());
    newLoopBody = newForallOp.getBody();
  }

  // 4. Move the loop body to the new op.
  unsigned oldNumArguments = oldLoopBody->getNumArguments();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));

  // 5. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    rewriter.setInsertionPoint(newForallOp.getTerminator());
    clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, sliceOp.getSource(), sliceOp.getDest(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
    clonedInsertSliceOp =
        cast<tensor::InsertSliceOp>(rewriter.clone(*candidateSliceOp));
  }

  // 6.a. Clone consumer op.
  auto newForOpBlockArgsForConsumerDest =
      newLoopBody->getArguments().drop_front(oldNumArguments);
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newForOpBlockArgsForConsumerDest));

  // 6.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 7 - Perform tiling of the cloned consumer and replace the operand at
  // `operandNumber` with the source of the cloned tensor.insert_slice op.
  auto ossSliceOp =
      cast<OffsetSizeAndStrideOpInterface>(clonedInsertSliceOp.getOperation());

  LLVM_DEBUG({
    llvm::dbgs() << "=== Before replaceInsertSliceWithTiledConsumer ===\n";
    llvm::dbgs() << "Module state:\n";
    newLoopOp->getParentOfType<ModuleOp>()->dump();
    llvm::dbgs() << "clonedConsumerOp:\n";
    clonedConsumerOp->dump();
    llvm::dbgs() << "clonedInsertSliceOp:\n";
    clonedInsertSliceOp->dump();
  });

  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp, clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    LLVM_DEBUG(llvm::dbgs() << "replaceInsertSliceWithTiledConsumer failed\n");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "=== After replaceInsertSliceWithTiledConsumer ===\n";
    llvm::dbgs() << "Module state:\n";
    newLoopOp->getParentOfType<ModuleOp>()->dump();
    llvm::dbgs() << "Tiled consumer op:\n";
    tileAndFuseResult->tiledOps[0]->dump();
  });

  rewriter.replaceAllUsesWith(
      tileAndFuseResult->tiledOps[0]->getOperand(operandNumber),
      clonedInsertSliceOp.getSource());

  // 8 - Extract offset/sizes/strides required to create the
  // tensor.insert_slice/parallel_insert_slice for each result of the consumer.
  SmallVector<OpFoldResult> offsets = ossSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = ossSliceOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = ossSliceOp.getMixedStrides();

  // 9. Check all insert stride is 1.
  if (llvm::any_of(strides, [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "containingOp's result yield with stride");
  }

  // 10. Try to get iter domain position from input position.
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(clonedConsumerOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
          iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        clonedConsumerOp, "can't get iter domain position from input position");
  }

  // 11. Try to fetch the offset and size for all results of the cloned
  // consumer. This would then be used to form the corresponding
  // tensor.insert_slice/parallel_insert_slice later.
  unsigned totalNumResultsOfConsumer = clonedConsumerOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(
      totalNumResultsOfConsumer);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(totalNumResultsOfConsumer);
  for (auto [idx, v] : llvm::enumerate(clonedConsumerOp->getResults())) {
    if (failed(clonedConsumerOp.getResultTilePosition(
            rewriter, idx, iterDomainOffsets, iterDomainSizes,
            resultOffsets[idx], resultSizes[idx]))) {
      return rewriter.notifyMatchFailure(
          clonedConsumerOp,
          "can't get result domain position from iter domain position");
    }
  }

  auto arrayRefOffsets = ArrayRef<SmallVector<OpFoldResult>>(resultOffsets);
  auto arrayRefSizes = ArrayRef<SmallVector<OpFoldResult>>(resultSizes);

  LLVM_DEBUG({
    llvm::dbgs() << "=== Before fixTerminator ===\n";
    llvm::dbgs() << "Module state:\n";
    newLoopOp->getParentOfType<ModuleOp>()->dump();
  });

  if (isInsertSliceOp) {
    auto newForOp = cast<scf::ForOp>(newLoopOp);
    fixTerminatorSCFYield(
        rewriter, newForOp, *tileAndFuseResult, arrayRefOffsets, arrayRefSizes,
        newForOp.getBody()->getArguments().drop_front(1 + initSize));
  } else {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    fixTerminatorSCFInParallel(
        rewriter, newForallOp, tileAndFuseResult->tiledOps[0]->getResults(),
        arrayRefOffsets, arrayRefSizes,
        newForallOp.getBody()->getArguments().drop_front(rank + initSize));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "=== After fixTerminator ===\n";
    llvm::dbgs() << "Module state:\n";
    newLoopOp->getParentOfType<ModuleOp>()->dump();
  });

  // 12. Replace the result of scf loop and consumer op with new loop's results.
  for (auto &&[oldResult, newResult] :
       llvm::zip_first(oldLoopOp->getResults(), newLoopOp->getResults())) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 newLoopOp->getResults().drop_front(initSize))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 13. Need to erase the old scf loop and the cloned consumer op.
  rewriter.eraseOp(oldLoopOp);
  rewriter.eraseOp(clonedConsumerOp);

  LLVM_DEBUG({
    llvm::dbgs() << "=== Final result after fusion ===\n";
    llvm::dbgs() << "Module state:\n";
    newLoopOp->getParentOfType<ModuleOp>()->dump();
  });

  return scf::SCFFuseConsumerOfSliceResult{
      consumerOpOperand,
      &(tileAndFuseResult->tiledOps[0]->getOpOperand(operandNumber)),
      tileAndFuseResult->tiledOps};
}

//===----------------------------------------------------------------------===//
// Helper Functions for Element-wise Fusion
//===----------------------------------------------------------------------===//

namespace {

/// Helper struct to hold loop and consumer analysis results
struct FusionAnalysis {
  scf::ForOp forOp;
  unsigned resultNumber;
  unsigned consumerOperandNumber;
  Value loopResult;
};

/// Analyzes the loop and consumer to extract fusion parameters.
/// This is common to both in-place and split-reduction fusion strategies.
static FailureOr<FusionAnalysis>
analyzeFusionContext(tensor::InsertSliceOp candidateSliceOp,
                     Operation *consumerOp) {
  FusionAnalysis analysis;

  // Get the containing scf.for loop
  analysis.forOp = candidateSliceOp->getParentOfType<scf::ForOp>();
  if (!analysis.forOp) {
    return failure();
  }

  // Find the result number that the insert_slice yields
  auto yieldOp = cast<scf::YieldOp>(analysis.forOp.getBody()->getTerminator());
  bool found = false;
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (yieldOp.getOperand(i) == candidateSliceOp.getResult()) {
      analysis.resultNumber = i;
      found = true;
      break;
    }
  }

  if (!found) {
    return failure();
  }

  analysis.loopResult = analysis.forOp->getResult(analysis.resultNumber);

  // Find which operand of consumer uses the loop result
  found = false;
  for (OpOperand &operand : consumerOp->getOpOperands()) {
    if (operand.get() == analysis.loopResult) {
      analysis.consumerOperandNumber = operand.getOperandNumber();
      found = true;
      break;
    }
  }

  if (!found) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Fusion analysis results:\n";
    llvm::dbgs() << "  Loop result number: " << analysis.resultNumber << "\n";
    llvm::dbgs() << "  Consumer operand number: " << analysis.consumerOperandNumber << "\n";
  });

  return analysis;
}

/// Performs the consumer fusion within a loop body.
/// This handles cloning the consumer, tiling it, and updating the insert_slice.
/// Returns the tiled consumer operation and the new insert_slice.
struct FusionResult {
  Operation *tiledConsumer;
  tensor::InsertSliceOp newInsertSlice;
  SmallVector<Operation *> tiledOps;
};

static FailureOr<FusionResult>
fuseConsumerIntoLoop(RewriterBase &rewriter,
                     tensor::InsertSliceOp insertSliceOp,
                     Operation *consumerOp,
                     Value loopResultToReplace,
                     unsigned consumerOperandNumber,
                     Location loc) {

  LLVM_DEBUG({
    llvm::dbgs() << "=== Fusing consumer into loop ===\n";
    llvm::dbgs() << "insertSliceOp:\n";
    insertSliceOp->dump();
    llvm::dbgs() << "consumerOp:\n";
    consumerOp->dump();
  });

  // Step 1: Clone the consumer into the loop body
  // Map the loop result to the insert_slice result
  rewriter.setInsertionPointAfter(insertSliceOp);

  IRMapping consumerMapper;
  consumerMapper.map(loopResultToReplace, insertSliceOp.getResult());
  Operation *clonedConsumer = rewriter.clone(*consumerOp, consumerMapper);

  LLVM_DEBUG({
    llvm::dbgs() << "Cloned consumer:\n";
    clonedConsumer->dump();
  });

  // Step 2: Use replaceInsertSliceWithTiledConsumer to tile the consumer
  auto ossSliceOp = cast<OffsetSizeAndStrideOpInterface>(
      insertSliceOp.getOperation());

  FailureOr<TilingResult> tileResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp,
          clonedConsumer->getOpOperand(consumerOperandNumber));

  if (failed(tileResult)) {
    LLVM_DEBUG(llvm::dbgs() << "replaceInsertSliceWithTiledConsumer failed\n");
    return failure();
  }

  Operation *tiledConsumer = tileResult->tiledOps[0];

  LLVM_DEBUG({
    llvm::dbgs() << "Tiled consumer:\n";
    tiledConsumer->dump();
  });

  // Step 3: Replace extract_slice(insert_slice_result) with producer's tiled result
  // This eliminates redundant extract operations
  Value producerTiledResult = insertSliceOp.getSource();
  rewriter.modifyOpInPlace(tiledConsumer, [&]() {
    for (unsigned i = 0; i < tiledConsumer->getNumOperands(); ++i) {
      Value operand = tiledConsumer->getOperand(i);
      if (auto extractOp = operand.getDefiningOp<tensor::ExtractSliceOp>()) {
        if (extractOp.getSource() == insertSliceOp.getResult()) {
          tiledConsumer->setOperand(i, producerTiledResult);
          LLVM_DEBUG(llvm::dbgs() << "Replaced operand " << i
                                  << " with producer tiled result\n");
        }
      }
    }
  });

  // Step 4: Create new insert_slice with consumer's result as source
  IRMapping insertMapper;
  insertMapper.map(insertSliceOp.getSource(), tiledConsumer->getResult(0));
  auto newInsertSliceOp = cast<tensor::InsertSliceOp>(
      rewriter.clone(*insertSliceOp, insertMapper));

  LLVM_DEBUG({
    llvm::dbgs() << "Created new insert_slice:\n";
    newInsertSliceOp->dump();
  });

  // Erase the cloned consumer (we only need the tiled version)
  rewriter.eraseOp(clonedConsumer);

  return FusionResult{tiledConsumer, newInsertSliceOp, tileResult->tiledOps};
}

/// Result of cloning a loop body
struct ClonedLoopBody {
  tensor::InsertSliceOp clonedInsertSlice;
  scf::YieldOp newYieldOp;
};

/// Clones a loop body into a new loop.
/// Returns the cloned insert_slice operation and the new yield operation.
static ClonedLoopBody
cloneLoopBodyIntoNewLoop(RewriterBase &rewriter,
                         scf::ForOp oldForOp,
                         scf::ForOp newForOp,
                         tensor::InsertSliceOp candidateSliceOp) {
  Block *oldLoopBody = oldForOp.getBody();
  Block *newLoopBody = newForOp.getBody();

  // Build block argument mapping
  IRMapping mapper;
  for (auto [oldArg, newArg] : llvm::zip(oldLoopBody->getArguments(),
                                          newLoopBody->getArguments())) {
    mapper.map(oldArg, newArg);
  }

  // Clone all operations in the loop body
  rewriter.setInsertionPointToStart(newLoopBody);
  for (Operation &op : oldLoopBody->without_terminator()) {
    rewriter.clone(op, mapper);
  }

  // Clone the yield operation
  auto oldYield = cast<scf::YieldOp>(oldLoopBody->getTerminator());
  SmallVector<Value> yieldOperands;
  for (Value operand : oldYield.getOperands()) {
    yieldOperands.push_back(mapper.lookupOrDefault(operand));
  }
  auto newYieldOp = rewriter.create<scf::YieldOp>(oldForOp.getLoc(), yieldOperands);

  // Find the cloned candidateSliceOp
  Operation *clonedOp = mapper.lookup(candidateSliceOp.getOperation());
  assert(clonedOp && "Failed to find cloned insert_slice in new loop");
  auto clonedInsertSlice = cast<tensor::InsertSliceOp>(clonedOp);

  return ClonedLoopBody{clonedInsertSlice, newYieldOp};
}

} // namespace

//===----------------------------------------------------------------------===//
// In-place Element-wise Fusion
//===----------------------------------------------------------------------===//

/// In-place fusion for element-wise consumers into spatial loops.
/// This strategy is designed for spatial (non-reduction) loops where the consumer
/// can be safely fused into every iteration without correctness issues.
///
/// Strategy:
/// 1. Create a new loop at the consumer's position
/// 2. Move the original loop's body to the new loop (using mergeBlocks)
/// 3. Fuse the consumer into the loop body
///
/// Example:
///   Original:
///     %0 = scf.for %i = 0 to 1024 step 32 iter_args(%arg = %init) {
///       %slice_A = tensor.extract_slice %A[%i, 0] [32, 1024]
///       %tiled = linalg.matmul ins(%slice_A, %B) outs(%slice_out)
///       %inserted = tensor.insert_slice %tiled into %arg[%i, 0]
///       scf.yield %inserted
///     }
///     %1 = linalg.add(%0, %D)  // consumer
///
///   After in-place fusion:
///     %0 = scf.for %i = 0 to 1024 step 32 iter_args(%arg = %init) {
///       %slice_A = tensor.extract_slice %A[%i, 0] [32, 1024]
///       %tiled = linalg.matmul ins(%slice_A, %B) outs(%slice_out)
///       %slice_D = tensor.extract_slice %D[%i, 0] [32, 1024]
///       %tiled_add = linalg.add ins(%tiled, %slice_D) outs(%tiled)
///       %inserted = tensor.insert_slice %tiled_add into %arg[%i, 0]
///       scf.yield %inserted
///     }
///
/// Key implementation detail:
/// - Uses mergeBlocks (not clone) to preserve Transform dialect handles
/// - This is critical because Transform operations may hold handles to operations
///   inside the loop, and cloning would break these handles
///
/// IMPORTANT: This is Layer 3 (IR Transformation Layer).
/// All preconditions MUST be validated by Layer 2 (FusionValidator)
/// before calling this function. This function trusts its inputs.
///
/// Required preconditions (validated by caller):
/// - candidateSliceOp is inside an scf.for loop (spatial loop)
/// - consumerOp is element-wise (all parallel, permutation maps)
/// - consumerOp uses the loop result as DPS init (in-place)
/// - loop result has only one user (the consumer)
FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseEltwiseConsumerInPlace(RewriterBase &rewriter,
                           tensor::InsertSliceOp candidateSliceOp,
                           Operation *consumerOp) {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting fuseEltwiseConsumerInPlace ===\n");

  // Analyze the fusion context
  FailureOr<FusionAnalysis> analysisResult =
      analyzeFusionContext(candidateSliceOp, consumerOp);
  if (failed(analysisResult)) {
    return failure();
  }
  FusionAnalysis &analysis = *analysisResult;

  auto forOp = analysis.forOp;
  unsigned resultNumber = analysis.resultNumber;
  unsigned consumerOperandNumber = analysis.consumerOperandNumber;

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = forOp.getLoc();

  // Step 1: Create a new scf.for loop at consumer's position
  // This ensures all consumer inputs are in the correct scope
  rewriter.setInsertionPoint(consumerOp);

  auto newForOp = rewriter.create<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), forOp.getInits());

  // Step 2: Move the old loop's body to the new loop
  // IMPORTANT: Use mergeBlocks (not clone) to preserve Transform dialect handles.
  // If Transform operations hold handles to operations inside the loop,
  // cloning would invalidate those handles when the old loop is erased.
  // mergeBlocks physically moves operations, preserving their identity.
  Block *oldLoopBody = forOp.getBody();
  Block *newLoopBody = newForOp.getBody();
  unsigned oldNumArguments = oldLoopBody->getNumArguments();

  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));

  // After mergeBlocks, candidateSliceOp is now in the new loop
  auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());

  LLVM_DEBUG({
    llvm::dbgs() << "Created new loop and moved old loop's body\n";
    llvm::dbgs() << "candidateSliceOp is now in new loop:\n";
    candidateSliceOp->dump();
  });

  // Step 3: Fuse the consumer into the loop using helper function
  // The helper will:
  // - Clone and tile the consumer
  // - Replace extract_slice(insert_slice_result) with producer's tiled result
  // - Create a new insert_slice with consumer's result
  FailureOr<FusionResult> fusionResult = fuseConsumerIntoLoop(
      rewriter, candidateSliceOp, consumerOp,
      forOp->getResult(resultNumber), consumerOperandNumber, loc);

  if (failed(fusionResult)) {
    return failure();
  }

  Operation *tiledConsumer = fusionResult->tiledConsumer;
  auto newInsertSliceOp = fusionResult->newInsertSlice;

  // Step 4: Update the yield operation to use the new insert_slice result
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp.setOperand(resultNumber, newInsertSliceOp.getResult());
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Updated yield to use new insert_slice result\n";
    yieldOp->dump();
  });

  // Save a pointer to the original consumer's operand before erasing
  OpOperand *origConsumerOperand = &consumerOp->getOpOperand(consumerOperandNumber);

  // Step 5: Cleanup - Replace old operations and erase them
  rewriter.replaceAllUsesWith(forOp->getResults(), newForOp->getResults());
  rewriter.replaceAllUsesWith(consumerOp->getResults(), newForOp->getResults());
  rewriter.eraseOp(forOp);
  rewriter.eraseOp(consumerOp);
  // Note: clonedConsumer is already erased by fuseConsumerIntoLoop

  LLVM_DEBUG({
    llvm::dbgs() << "=== Final fused loop ===\n";
    newForOp->dump();
  });

  // Return the fusion result
  // Note: origConsumerOperand points to deleted op, but that's how the original API works
  return scf::SCFFuseConsumerOfSliceResult{
      origConsumerOperand,
      &(tiledConsumer->getOpOperand(consumerOperandNumber)),
      fusionResult->tiledOps};
}


//===----------------------------------------------------------------------===//
// Split-Reduction Element-wise Fusion
//===----------------------------------------------------------------------===//

/// In-place fusion with split-reduction for element-wise consumers.
/// This strategy is designed for reduction loops where the consumer should only
/// execute after the reduction completes.
///
/// Strategy:
/// 1. Modify the original loop's upper bound to exclude the last iteration
/// 2. Create a second loop for the last iteration and clone the loop body
/// 3. Fuse the consumer into the second loop body
/// 4. Update the yield operation in the second loop
/// 5. Cleanup and replace uses
///
/// Example:
///   Original:
///     %0 = scf.for %i = 0 to 1024 step 8 { matmul reduction on k }
///     %1 = linalg.add(%0, %D)  // consumer
///
///   After split-reduction fusion:
///     %0 = scf.for %i = 0 to 1016 step 8 { matmul only }
///     %1 = scf.for %i = 1016 to 1024 step 8 iter_args(%arg = %0) {
///       matmul + add  // consumer fused into last iteration
///     }
///
/// IMPORTANT: This is Layer 3 (IR Transformation Layer).
/// All preconditions MUST be validated by Layer 2 (FusionValidator)
/// before calling this function. This function trusts its inputs.
///
/// Required preconditions (validated by caller):
/// - candidateSliceOp is inside an scf.for loop (reduction loop)
/// - consumerOp is element-wise (all parallel, permutation maps)
/// - consumerOp uses the loop result as DPS init (in-place)
/// - loop result has only one user (the consumer)
FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseEltwiseConsumerInPlaceWithSplitReduction(
    RewriterBase &rewriter,
    tensor::InsertSliceOp candidateSliceOp,
    Operation *consumerOp) {

  LLVM_DEBUG(llvm::dbgs() << "=== Starting fuseEltwiseConsumerInPlaceWithSplitReduction ===\n");

  // Analyze the fusion context
  FailureOr<FusionAnalysis> analysisResult =
      analyzeFusionContext(candidateSliceOp, consumerOp);
  if (failed(analysisResult)) {
    return failure();
  }
  FusionAnalysis &analysis = *analysisResult;

  auto forOp = analysis.forOp;
  unsigned resultNumber = analysis.resultNumber;
  unsigned consumerOperandNumber = analysis.consumerOperandNumber;

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = forOp.getLoc();

  // Step 1: Modify the original loop's upper bound to (ub - step)
  // This excludes the last iteration from the original loop
  rewriter.setInsertionPoint(forOp);

  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value newUpperBound1 = rewriter.create<arith::SubIOp>(loc, upperBound, step);

  rewriter.modifyOpInPlace(forOp, [&]() {
    forOp.getUpperBoundMutable().assign(newUpperBound1);
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Modified original loop upper bound to (ub - step)\n";
    forOp->dump();
  });

  // Step 2: Create the second loop [ub-step, ub) and clone the loop body
  // The second loop is created at the consumer's position
  rewriter.setInsertionPoint(consumerOp);

  // The second loop uses the first loop's results as initial values
  auto secondForOp = rewriter.create<scf::ForOp>(
      loc, newUpperBound1, upperBound, step, forOp.getResults());

  // Clone the original loop body into the second loop using helper function
  ClonedLoopBody clonedBody = cloneLoopBodyIntoNewLoop(
      rewriter, forOp, secondForOp, candidateSliceOp);

  auto secondLoopInsertSlice = clonedBody.clonedInsertSlice;
  auto secondYieldOp = clonedBody.newYieldOp;

  LLVM_DEBUG({
    llvm::dbgs() << "Created second loop and cloned loop body\n";
    secondForOp->dump();
  });

  // Step 3: Fuse the consumer into the second loop using helper function
  // The helper will:
  // - Clone and tile the consumer
  // - Replace extract_slice(insert_slice_result) with producer's tiled result
  // - Create a new insert_slice with consumer's result
  FailureOr<FusionResult> fusionResult = fuseConsumerIntoLoop(
      rewriter, secondLoopInsertSlice, consumerOp,
      forOp->getResult(resultNumber), consumerOperandNumber, loc);

  if (failed(fusionResult)) {
    return failure();
  }

  Operation *tiledConsumer = fusionResult->tiledConsumer;
  auto newInsertSliceOp = fusionResult->newInsertSlice;

  // Step 4: Update the yield operation in the second loop
  rewriter.modifyOpInPlace(secondYieldOp, [&]() {
    secondYieldOp.setOperand(resultNumber, newInsertSliceOp.getResult());
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Updated yield in second loop to use new insert_slice result\n";
    secondYieldOp->dump();
  });

  // Save a pointer to the original consumer's operand before erasing
  OpOperand *origConsumerOperand = &consumerOp->getOpOperand(consumerOperandNumber);

  // Step 5: Cleanup - Replace consumer uses and erase it
  rewriter.replaceAllUsesWith(consumerOp->getResults(),
                              secondForOp->getResults());

  rewriter.eraseOp(consumerOp);
  // Note: clonedConsumer is already erased by fuseConsumerIntoLoop

  LLVM_DEBUG({
    llvm::dbgs() << "=== Final result with split reduction ===\n";
    llvm::dbgs() << "First loop (no fusion):\n";
    forOp->dump();
    llvm::dbgs() << "Second loop (with fusion):\n";
    secondForOp->dump();
  });

  return scf::SCFFuseConsumerOfSliceResult{
      origConsumerOperand,
      &(tiledConsumer->getOpOperand(consumerOperandNumber)),
      fusionResult->tiledOps};
}


} // namespace mlir