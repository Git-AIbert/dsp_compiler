//===- BufferLoopSinkingPass.cpp - Sink deallocations out of loops --------===//
//
// This pass sinks deallocation operations out of loop nests to match
// hoisted allocation operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Dialect/Schedule/Transforms/Passes.h"

#define DEBUG_TYPE "buffer-loop-sinking"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_BUFFERLOOPSINKING
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Returns true if the given operation represents a loop.
static bool isLoop(Operation *op) {
  if (isa<LoopLikeOpInterface>(op))
    return true;

  auto regionInterface = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionInterface)
    return false;

  return regionInterface.hasLoop();
}

/// Get the end operation to place the given dealloc operation within the
/// specified placement block. Finds the last use of allocValue or any alias.
///
/// This function determines the safe insertion point for a dealloc operation
/// by finding the last operation that uses the allocated buffer or any of its
/// aliases. The dealloc must be placed after this operation to ensure all uses
/// complete before deallocation.
///
/// Precondition: placementBlock should be the post-dominator block where all
/// uses of allocValue have completed. At least one user must exist in this block.
///
/// Algorithm:
/// 1. Resolve all aliases of the allocated value (values that may alias)
/// 2. For each alias, iterate through all its users
/// 3. For users in nested blocks, find their ancestor operation in placementBlock
/// 4. Track the latest (furthest) ancestor operation in placementBlock
/// 5. Return this operation as the insertion point (asserts if no user found)
///
/// Example:
///   Block placementBlock:
///     %alloc = memref.alloc()
///     scf.for %i ... {        // op1: ancestor of user in nested block
///       use(%alloc)           // user in nested block
///     }
///     scf.for %j ... {        // op2: ancestor of another user (latest)
///       use(%alloc)           // another user
///     }
///     func.return             // terminator
///
/// Returns op2 (the scf.for at %j) because it's the latest ancestor of any user.
/// The dealloc will be inserted after op2, before the terminator.
static Operation *getEndOperation(Value allocValue, Block *placementBlock,
                                  const Liveness &liveness,
                                  const BufferViewFlowAnalysis &aliases) {
  // Resolve all possible aliases of the allocated value
  // Aliases are values that may point to the same memory (e.g., via memref.subview)
  auto allAliases = aliases.resolve(allocValue);

  // Track the latest operation in placementBlock that uses allocValue or its aliases
  Operation *lastAncestorInPlacementBlock = nullptr;

  // Iterate through all aliases and their users to find the last use
  for (Value alias : allAliases) {
    for (Operation *user : alias.getUsers()) {
      // Skip dealloc operations - we're looking for actual uses, not deallocations
      if (isa<memref::DeallocOp>(user))
        continue;

      Operation *ancestorOp = user;

      // Handle users in nested blocks (e.g., inside loops or if statements)
      // We need to find the top-level operation in placementBlock that contains this user
      if (user->getBlock() != placementBlock) {
        // Find the ancestor operation in placementBlock that contains this user
        // Example: if user is inside "scf.for", ancestorOp becomes the scf.for op
        ancestorOp = placementBlock->findAncestorOpInBlock(*user);
        if (!ancestorOp)
          continue;  // User is not in a nested region of placementBlock
      }

      // Update lastAncestorInPlacementBlock to track the furthest operation
      // "Furthest" = the operation that appears latest in the block
      if (!lastAncestorInPlacementBlock) {
        // First user found
        lastAncestorInPlacementBlock = ancestorOp;
      } else {
        // Check if ancestorOp comes after the current lastAncestorInPlacementBlock
        if (lastAncestorInPlacementBlock->isBeforeInBlock(ancestorOp)) {
          // ancestorOp is later in the block, so update our tracking
          lastAncestorInPlacementBlock = ancestorOp;
        }
      }
    }
  }

  // We should find at least one user in the post-dominator block
  assert(lastAncestorInPlacementBlock &&
         "Failed to find any user in post-dominator block");

  return lastAncestorInPlacementBlock;
}

//===----------------------------------------------------------------------===//
// BufferDeallocSinking
//===----------------------------------------------------------------------===//

/// Sinks dealloc operations out of loops. This is symmetric to
/// BufferAllocationHoisting but works in the opposite direction using
/// post-dominator analysis.
class BufferDeallocSinking : public BufferPlacementTransformationBase {
public:
  BufferDeallocSinking(Operation *op)
      : BufferPlacementTransformationBase(op), postDominators(op) {}

  /// Sinks dealloc operations downwards (out of loops).
  void sink() {
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      Operation *deallocOp = std::get<1>(entry);

      // Skip if there's no dealloc operation
      if (!deallocOp)
        continue;

      LDBG("Processing dealloc: " << *deallocOp);
      LDBG("  Alloc value: " << allocValue);

      // Get all aliases of this allocation
      auto resultAliases = aliases.resolve(allocValue);

      // Find the common post-dominator block of all aliases
      // This is symmetric to finding the common dominator for alloc hoisting
      Block *postDominatorBlock =
          findCommonDominator(allocValue, resultAliases, postDominators);

      if (!postDominatorBlock) {
        LDBG("  No post-dominator block found, skipping");
        continue;
      }

      // If the post-dominator block is the same as current block, no need to move
      if (postDominatorBlock == deallocOp->getBlock()) {
        LDBG("  Post-dominator block is same as current block, skipping");
        continue;
      }

      // Check if we're crossing loop boundaries
      // Only sink if we're moving the dealloc out of at least one loop
      Operation *currentParent = deallocOp->getParentOp();
      Operation *targetParent = postDominatorBlock->getParentOp();

      LDBG("  Current parent:");
      if (currentParent) {
        LDBG("    " << *currentParent);
      } else {
        LDBG("    null");
      }

      LDBG("  Target parent:");
      if (targetParent) {
        LDBG("    " << *targetParent);
      } else {
        LDBG("    null");
      }

      bool crossesLoop = false;
      Operation *op = currentParent;
      while (op && op != targetParent) {
        if (isLoop(op)) {
          LDBG("    Found loop to cross:");
          LDBG("      " << *op);
          crossesLoop = true;
          break;
        }
        op = op->getParentOp();
      }

      // Only perform sinking if we're crossing at least one loop
      // This is symmetric to loop hoisting
      if (!crossesLoop) {
        LDBG("  Not crossing any loops, skipping");
        continue;
      }

      // Verify that we're actually moving to a valid location
      // The target block should post-dominate the current block
      if (!postDominators.postDominates(postDominatorBlock, deallocOp->getBlock())) {
        LDBG("  Post-dominator block does not post-dominate current block, skipping");
        continue;
      }

      // Find the appropriate insertion point in the post-dominator block
      // This should be after the last use of allocValue or any of its aliases
      Operation *endOperation = getEndOperation(
          allocValue, postDominatorBlock, liveness, aliases);

      LDBG("  End operation:");
      LDBG("    " << *endOperation);

      // Move the dealloc operation after the last use
      deallocOp->moveAfter(endOperation);
      LDBG("  Successfully sunk dealloc operation");
    }
  }

private:
  /// Post-dominator info to find where to sink deallocs
  PostDominanceInfo postDominators;
};

//===----------------------------------------------------------------------===//
// BufferLoopSinkingPass
//===----------------------------------------------------------------------===//

struct BufferLoopSinkingPass
    : public impl::BufferLoopSinkingBase<BufferLoopSinkingPass> {

  void runOnOperation() override {
    // Sink all deallocs out of loops
    BufferDeallocSinking optimizer(getOperation());
    optimizer.sink();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> mlir::createBufferLoopSinkingPass() {
  return std::make_unique<BufferLoopSinkingPass>();
}
