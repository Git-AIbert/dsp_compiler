//===- ChainSplitReductionPipelinesPass.cpp - Chain split loop pipelines -===//
//
// This pass connects the software pipelining across split reduction loops
// that were divided to enable consumer fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"
#include "Dialect/Schedule/Transforms/LoopPairUtils.h"

#define DEBUG_TYPE "chain-split-reduction-pipelines"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_CHAINSPLITREDUCTIONPIPELINES
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

// ============================================================================
// Data Structures
// ============================================================================

// /// Information about a split loop pair
// struct SplitLoopPair {
//   scf::ForOp firstLoop;   // The first loop segment (e.g., K=0 to 1536)
//   scf::ForOp secondLoop;  // The second loop segment (e.g., K=1536 to 2048)

//   SplitLoopPair(scf::ForOp first, scf::ForOp second)
//       : firstLoop(first), secondLoop(second) {}
// };

/// Pattern for prefetch logic within a loop
struct PrefetchPattern {
  arith::CmpIOp boundaryCheck;      // The comparison: cmpi slt, %next, %bound
  scf::IfOp conditionalPrefetch;    // The scf.if wrapping the prefetch
  Operation *dmaOp;                 // The DMA operation inside the if

  PrefetchPattern() : boundaryCheck(nullptr), conditionalPrefetch(nullptr),
                      dmaOp(nullptr) {}

  bool isValid() const {
    return boundaryCheck && conditionalPrefetch && dmaOp;
  }

  void dump() const {
    LDBG("PrefetchPattern {");
    if (boundaryCheck) {
      LDBG("  boundaryCheck: " << boundaryCheck);
    } else {
      LDBG("  boundaryCheck: nullptr");
    }
    if (conditionalPrefetch) {
      LDBG("  conditionalPrefetch: " << conditionalPrefetch);
    } else {
      LDBG("  conditionalPrefetch: nullptr");
    }
    if (dmaOp) {
      LDBG("  dmaOp: " << *dmaOp);
    } else {
      LDBG("  dmaOp: nullptr");
    }
    LDBG("}");
  }
};

// ============================================================================
// Pass Implementation
// ============================================================================

struct ChainSplitReductionPipelinesPass
    : public impl::ChainSplitReductionPipelinesBase<ChainSplitReductionPipelinesPass> {

private:
  // ============================================================================
  // Helper Functions
  // ============================================================================

  /// Check if two loops form a split reduction pair
  /// Requirements:
  /// 1. Same parent block
  /// 2. Sequential in the block (no other loops in between)
  /// 3. first.upperBound == second.lowerBound
  /// 4. Same step size
  /// 5. Both return DMA tokens (i32)
  bool isSplitPair(scf::ForOp loop1, scf::ForOp loop2);

  /// Find the prefetch pattern within a loop
  /// Looks for: arith.cmpi slt, %next, %bound -> scf.if -> dma_opt
  std::optional<PrefetchPattern> findPrefetchPattern(scf::ForOp loop);

  /// Find the standalone prefetch DMA before the second loop
  /// Returns nullptr if not found
  Operation* findStandalonePrefetch(scf::ForOp secondLoop);

  /// Extend the prefetch boundary in the first loop
  /// Changes: cmpi slt, %next, %c1536 -> cmpi slt, %next, %c2048
  void extendPrefetchBoundary(PrefetchPattern &pattern, Value newBound);

  /// Remove the standalone prefetch and its associated subviews
  void removeStandalonePrefetch(Operation *prefetchOp);

  /// Connect the second loop's iter_args to the first loop's result
  void connectLoopPipelines(scf::ForOp firstLoop, scf::ForOp secondLoop);

  /// Process a single split loop pair
  void processSplitLoopPair(scf::ForOp loop1, scf::ForOp loop2);

  /// Find and process all split loop pairs recursively from root operation
  void findAndProcessSplitLoops(Operation *root);

public:
  void runOnOperation() override;
};

// ============================================================================
// Helper Function Implementations
// ============================================================================

bool ChainSplitReductionPipelinesPass::isSplitPair(scf::ForOp loop1,
                                                    scf::ForOp loop2) {
  // 1. Must be in the same parent block
  if (loop1->getBlock() != loop2->getBlock()) {
    LDBG("Different parent blocks");
    return false;
  }

  // 2. Check if first.upperBound == second.lowerBound
  if (loop1.getUpperBound() != loop2.getLowerBound()) {
    LDBG("Bounds don't match: " << loop1.getUpperBound() << " vs "
                                << loop2.getLowerBound());
    return false;
  }

  // 3. Check if they have the same step
  if (loop1.getStep() != loop2.getStep()) {
    LDBG("Different steps");
    return false;
  }

  // 4. Check if both return DMA tokens (i32)
  if (loop1.getNumResults() == 0 || loop2.getNumResults() == 0) {
    LDBG("Missing return values");
    return false;
  }

  if (!loop1.getResult(0).getType().isInteger(32) ||
      !loop2.getResult(0).getType().isInteger(32)) {
    LDBG("Not returning i32 tokens");
    return false;
  }

  LDBG("✓ Found valid split pair:");
  LDBG("  Loop1: " << loop1.getLowerBound() << " to "
                   << loop1.getUpperBound() << " step " << loop1.getStep());
  LDBG("  Loop2: " << loop2.getLowerBound() << " to "
                   << loop2.getUpperBound() << " step " << loop2.getStep());

  return true;
}

std::optional<PrefetchPattern>
ChainSplitReductionPipelinesPass::findPrefetchPattern(scf::ForOp loop) {
  LDBG("");
  LDBG(">>> Entering findPrefetchPattern");

  PrefetchPattern pattern;

  // Walk through the loop to find the prefetch pattern
  loop.walk([&](arith::CmpIOp cmpOp) {
    // Look for: arith.cmpi slt, %next_iter, %upper_bound
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt) {
      return WalkResult::advance();
    }

    Value rhs = cmpOp.getRhs();

    // Check if this comparison is used by an scf.if
    for (Operation *user : cmpOp->getUsers()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
        // Check if the condition is the comparison result
        if (ifOp.getCondition() != cmpOp.getResult()) {
          continue;
        }

        // Check if the then-region contains a DMA operation
        Region &thenRegion = ifOp.getThenRegion();
        for (Operation &op : thenRegion.front()) {
          if (op.getName().getStringRef().contains("dma")) {
            // Found the pattern!
            pattern.boundaryCheck = cmpOp;
            pattern.conditionalPrefetch = ifOp;
            pattern.dmaOp = &op;

            LDBG("Found prefetch pattern");
            pattern.dump();

            return WalkResult::interrupt();
          }
        }
      }
    }

    return WalkResult::advance();
  });

  if (pattern.isValid()) {
    LDBG("<<< Exiting findPrefetchPattern");
    LDBG("");
    return pattern;
  }

  LDBG("No prefetch pattern found");
  LDBG("<<< Exiting findPrefetchPattern");
  LDBG("");
  return std::nullopt;
}

Operation* ChainSplitReductionPipelinesPass::findStandalonePrefetch(
    scf::ForOp secondLoop) {
  LDBG("");
  LDBG(">>> Entering findStandalonePrefetch");

  // Walk backwards from the second loop to find a DMA operation
  Operation *prevOp = secondLoop->getPrevNode();

  while (prevOp) {
    // Check if this is a DMA operation
    if (prevOp->getName().getStringRef().contains("dma")) {
      // Verify that this DMA's result is used as iter_args of the second loop
      if (!secondLoop.getInitArgs().empty()) {
        Value initArg = secondLoop.getInitArgs()[0];
        if (initArg == prevOp->getResult(0)) {
          LDBG("Found standalone prefetch: " << *prevOp);
          LDBG("<<< Exiting findStandalonePrefetch");
          LDBG("");
          return prevOp;
        }
      }
    }

    // Stop if we hit another loop or control flow
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(prevOp)) {
      break;
    }

    prevOp = prevOp->getPrevNode();
  }

  LDBG("Standalone prefetch not found");
  LDBG("<<< Exiting findStandalonePrefetch");
  LDBG("");
  return nullptr;
}

void ChainSplitReductionPipelinesPass::extendPrefetchBoundary(
    PrefetchPattern &pattern, Value newBound) {
  LDBG("");
  LDBG(">>> Entering extendPrefetchBoundary");

  LDBG("Extending prefetch boundary from " << pattern.boundaryCheck.getRhs()
                                            << " to " << newBound);

  // Simply replace the right-hand side of the comparison
  pattern.boundaryCheck.setOperand(1, newBound);

  LDBG("<<< Exiting extendPrefetchBoundary");
  LDBG("");
}

void ChainSplitReductionPipelinesPass::removeStandalonePrefetch(
    Operation *prefetchOp) {
  LDBG("");
  LDBG(">>> Entering removeStandalonePrefetch");

  if (!prefetchOp) return;

  LDBG("Removing standalone prefetch: " << *prefetchOp);

  // Collect operands that are subview operations with single use
  SmallVector<Operation*> toErase;
  for (Value operand : prefetchOp->getOperands()) {
    if (auto defOp = operand.getDefiningOp()) {
      if (isa<memref::SubViewOp>(defOp) && defOp->hasOneUse()) {
        toErase.push_back(defOp);
      }
    }
  }

  // Erase the prefetch operation
  prefetchOp->erase();

  // Erase the now-unused subviews
  for (Operation *op : toErase) {
    LDBG("  Removing subview: " << *op);
    op->erase();
  }

  LDBG("<<< Exiting removeStandalonePrefetch");
  LDBG("");
}

void ChainSplitReductionPipelinesPass::connectLoopPipelines(
    scf::ForOp firstLoop, scf::ForOp secondLoop) {

  LDBG("");
  LDBG(">>> Entering connectLoopPipelines");

  // Get the first loop's result (DMA token)
  Value firstLoopResult = firstLoop.getResult(0);

  LDBG("Connecting loop pipelines:");
  LDBG("  First loop result: " << firstLoopResult);
  LDBG("  Second loop old init: " << secondLoop.getInitArgs()[0]);

  // Replace the second loop's initial iter_arg
  secondLoop.getInitArgsMutable()[0].assign(firstLoopResult);

  LDBG("  ✓ Pipelines connected");

  LDBG("<<< Exiting connectLoopPipelines");
  LDBG("");
}

// ============================================================================
// Main Processing Logic
// ============================================================================

void ChainSplitReductionPipelinesPass::processSplitLoopPair(
    scf::ForOp loop1, scf::ForOp loop2) {

  LDBG("");
  LDBG(">>> Entering processSplitLoopPair");

  LDBG("First loop:  " << loop1.getLowerBound()
                       << " to " << loop1.getUpperBound());
  LDBG("Second loop: " << loop2.getLowerBound()
                       << " to " << loop2.getUpperBound());

  // Step 1: Find the prefetch pattern in the first loop
  auto patternOpt = findPrefetchPattern(loop1);
  if (!patternOpt) {
    LDBG("  No prefetch pattern found in first loop, skipping");
    LDBG("<<< Exiting processSplitLoopPair");
    LDBG("");
    return;
  }

  PrefetchPattern pattern = *patternOpt;

  // Step 2: Extend the prefetch boundary
  Value secondLoopUpperBound = loop2.getUpperBound();
  extendPrefetchBoundary(pattern, secondLoopUpperBound);

  // Step 3: Find the standalone prefetch (before removing it)
  Operation *standalonePrefetch = findStandalonePrefetch(loop2);

  // Step 4: Connect the loop pipelines FIRST (update iter_args)
  // This must happen BEFORE removing the standalone prefetch
  connectLoopPipelines(loop1, loop2);

  // Step 5: Now safe to remove the standalone prefetch
  if (standalonePrefetch) {
    removeStandalonePrefetch(standalonePrefetch);
  }

  LDBG("<<< Exiting processSplitLoopPair");
  LDBG("");
}

void ChainSplitReductionPipelinesPass::findAndProcessSplitLoops(
    Operation *root) {
  LDBG("");

  // Use the common implementation with custom debug logger
  auto debugLogger = [](const char* msg) {
    LDBG(msg);
  };

  auto isSplitPairFn = [this](scf::ForOp loop1, scf::ForOp loop2) {
    return isSplitPair(loop1, loop2);
  };

  auto processPairFn = [this](scf::ForOp loop1, scf::ForOp loop2) {
    processSplitLoopPair(loop1, loop2);
  };

  schedule::findAndProcessSplitLoopsImpl(root, isSplitPairFn, processPairFn, debugLogger);

  LDBG("");
}

// ============================================================================
// Pass Entry Point
// ============================================================================

void ChainSplitReductionPipelinesPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  LDBG("");
  LDBG("===============================================");
  LDBG("  ChainSplitReductionPipelines Pass Started");
  LDBG("===============================================");

  findAndProcessSplitLoops(funcOp);

  LDBG("===============================================");
  LDBG("  ChainSplitReductionPipelines Pass Completed");
  LDBG("===============================================");
  LDBG("");
}

}  // namespace

// ============================================================================
// Pass Registration
// ============================================================================

std::unique_ptr<Pass> mlir::createChainSplitReductionPipelinesPass() {
  return std::make_unique<ChainSplitReductionPipelinesPass>();
}
