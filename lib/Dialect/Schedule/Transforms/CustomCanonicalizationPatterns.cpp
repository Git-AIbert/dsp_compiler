//===- CustomCanonicalizationPatterns.cpp - Custom Canonicalization -----===//
//
// This file implements canonicalization patterns while preserving single-
// iteration loops by excluding the SimplifyTrivialLoops pattern.
//
// The ForOpIterArgsFolder and ForOpTensorCastFolder patterns are copied from
// LLVM's SCF dialect implementation to maintain full canonicalization support
// except for single-iteration loop simplification.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Schedule/Transforms/CustomCanonicalizationPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

// ============================================================================
// Patterns copied from LLVM's mlir/lib/Dialect/SCF/IR/SCF.cpp
// to preserve ForOp canonicalization except SimplifyTrivialLoops
// ============================================================================

/// Fold away ForOp iter arguments when:
/// 1) The op yields the iter arguments.
/// 2) The iter arguments have no use and the corresponding outer region
/// iterators (inputs) are yielded.
/// 3) The iter arguments have no use and the corresponding (operation) results
/// have no use.
struct ForOpIterArgsFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;

    int64_t numResults = forOp.getNumResults();
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(numResults);
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + numResults);
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getInitArgs().size());
    newYieldValues.reserve(numResults);
    newResultValues.reserve(numResults);
    for (auto it : llvm::zip(forOp.getInitArgs(),       // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             forOp.getYieldedValues()   // iter yield
                             )) {
      bool forwarded = ((std::get<1>(it) == std::get<3>(it)) ||
                        (std::get<1>(it).use_empty() &&
                         (std::get<0>(it) == std::get<3>(it) ||
                          std::get<2>(it).use_empty())));
      keepMask.push_back(!forwarded);
      canonicalize |= forwarded;
      if (forwarded) {
        newBlockTransferArgs.push_back(std::get<0>(it));
        newResultValues.push_back(std::get<0>(it));
        continue;
      }
      newIterArgs.push_back(std::get<0>(it));
      newYieldValues.push_back(std::get<3>(it));
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    if (!canonicalize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block &newBlock = newForOp.getRegion().front();

    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[1 + idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newForOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newForOp.getResult(collapsedIdx++);
      }
    }

    Block &oldBlock = forOp.getRegion().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");

    if (newIterArgs.empty()) {
      auto newYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      rewriter.inlineBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(forOp, newResultValues);
      return success();
    }

    auto cloneFilteredTerminator = [&](scf::YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<scf::YieldOp>(mergedTerminator.getLoc(),
                                    filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(forOp, newResultValues);
    return success();
  }
};

/// Helper function for ForOpTensorCastFolder.
static SmallVector<Value>
replaceTensorCastForOpIterArg(PatternRewriter &rewriter, OpOperand &operand,
                              Value replacement) {
  Type oldType = operand.get().getType(), newType = replacement.getType();
  assert(llvm::isa<RankedTensorType>(oldType) &&
         llvm::isa<RankedTensorType>(newType) &&
         "expected ranked tensor types");

  ForOp forOp = cast<ForOp>(operand.getOwner());
  assert(operand.getOperandNumber() >= forOp.getNumControlOperands() &&
         "expected an iter OpOperand");
  assert(operand.get().getType() != replacement.getType() &&
         "Expected a different type");
  SmallVector<Value> newIterOperands;
  for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
    if (opOperand.getOperandNumber() == operand.getOperandNumber()) {
      newIterOperands.push_back(replacement);
      continue;
    }
    newIterOperands.push_back(opOperand.get());
  }

  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands);
  newForOp->setAttrs(forOp->getAttrs());
  Block &newBlock = newForOp.getRegion().front();
  SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                             newBlock.getArguments().end());

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(&newBlock, newBlock.begin());
  BlockArgument newRegionIterArg = newForOp.getTiedLoopRegionIterArg(
      &newForOp->getOpOperand(operand.getOperandNumber()));
  Value castIn = rewriter.create<tensor::CastOp>(newForOp.getLoc(), oldType,
                                                 newRegionIterArg);
  newBlockTransferArgs[newRegionIterArg.getArgNumber()] = castIn;

  Block &oldBlock = forOp.getRegion().front();
  rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

  auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
  rewriter.setInsertionPoint(clonedYieldOp);
  unsigned yieldIdx =
      newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
  Value castOut = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), newType, clonedYieldOp.getOperand(yieldIdx));
  SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
  newYieldOperands[yieldIdx] = castOut;
  rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(clonedYieldOp);

  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> newResults = newForOp.getResults();
  newResults[yieldIdx] = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), oldType, newResults[yieldIdx]);

  return newResults;
}

/// Fold scf.for iter_arg/result pairs that go through tensor.cast ops.
struct ForOpTensorCastFolder : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    for (auto it : llvm::zip(op.getInitArgsMutable(), op.getResults())) {
      OpOperand &iterOpOperand = std::get<0>(it);
      auto incomingCast = iterOpOperand.get().getDefiningOp<tensor::CastOp>();
      if (!incomingCast ||
          incomingCast.getSource().getType() == incomingCast.getType())
        continue;
      if (!tensor::preservesStaticInformation(
              incomingCast.getDest().getType(),
              incomingCast.getSource().getType()))
        continue;
      if (!std::get<1>(it).hasOneUse())
        continue;

      rewriter.replaceOp(
          op, replaceTensorCastForOpIterArg(rewriter, iterOpOperand,
                                            incomingCast.getSource()));
      return success();
    }
    return failure();
  }
};

// ============================================================================
// End of copied patterns
// ============================================================================

/// Util function that tries to compute a constant diff between u and l.
/// Copied from LLVM's SCF.cpp.
static std::optional<int64_t> computeConstDiff(Value l, Value u) {
  IntegerAttr clb, cub;
  if (matchPattern(l, m_Constant(&clb)) && matchPattern(u, m_Constant(&cub))) {
    llvm::APInt lbValue = clb.getValue();
    llvm::APInt ubValue = cub.getValue();
    return (ubValue - lbValue).getSExtValue();
  }

  llvm::APInt diff;
  if (matchPattern(
          u, m_Op<arith::AddIOp>(matchers::m_Val(l), m_ConstantInt(&diff))) ||
      matchPattern(
          u, m_Op<arith::AddIOp>(m_ConstantInt(&diff), matchers::m_Val(l))))
    return diff.getSExtValue();
  return std::nullopt;
}

/// Rewriting pattern that erases loops that are known not to iterate, and
/// removes empty loops that iterate at least once and only return values
/// defined outside of the loop. Unlike SimplifyTrivialLoops, this does NOT
/// inline single-iteration loops.
///
/// This is essentially SimplifyTrivialLoops with the single-iteration inlining
/// logic removed.
struct SimplifyTrivialLoopsExceptSingleIteration : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    // If the upper bound is the same as the lower bound, the loop does not
    // iterate, just remove it.
    if (op.getLowerBound() == op.getUpperBound()) {
      rewriter.replaceOp(op, op.getInitArgs());
      return success();
    }

    std::optional<int64_t> diff =
        computeConstDiff(op.getLowerBound(), op.getUpperBound());
    if (!diff)
      return failure();

    // If the loop is known to have 0 iterations, remove it.
    if (*diff <= 0) {
      rewriter.replaceOp(op, op.getInitArgs());
      return success();
    }

    // NOTE: The original SimplifyTrivialLoops would check for single-iteration
    // loops here and inline them. We SKIP that check to preserve single-iteration
    // loops.

    // Now we are left with loops that have 1 or more iterations.
    // Only handle empty loops that return values defined outside.
    Block &block = op.getRegion().front();
    if (!llvm::hasSingleElement(block))
      return failure();

    // If the loop is empty and only returns values defined outside of the
    // loop, remove it.
    if (llvm::any_of(op.getYieldedValues(),
                     [&](Value v) { return !op.isDefinedOutsideOfLoop(v); }))
      return failure();

    rewriter.replaceOp(op, op.getYieldedValues());
    return success();
  }
};

} // namespace

namespace mlir {

void populateCustomCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  // Add canonicalization patterns from all dialects
  for (auto *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);

  // Add canonicalization patterns from all registered operations
  // except scf.for (we'll add custom patterns for it)
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    if (op.getStringRef() == "scf.for")
      continue;
    op.getCanonicalizationPatterns(patterns, ctx);
  }

  // Add our custom ForOp patterns that preserve single-iteration loops
  // These replace the standard ForOp canonicalization patterns
  patterns.add<ForOpIterArgsFolder>(ctx);
  patterns.add<ForOpTensorCastFolder>(ctx);
  patterns.add<SimplifyTrivialLoopsExceptSingleIteration>(ctx);
}

} // namespace mlir
