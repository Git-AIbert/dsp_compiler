//===- FusionValidator.cpp - Eltwise fusion validation ---*- C++ -*-===//
//
// This file implements centralized validation logic for element-wise fusion.
// Layer 2: Semantic Validation Layer
//
//===----------------------------------------------------------------------===//

#include "FusionValidator.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "eltwise-fusion-validator"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] " << X << "\n")

namespace mlir {

//===----------------------------------------------------------------------===//
// Individual Validation Functions
//===----------------------------------------------------------------------===//

bool isEltwiseOperation(linalg::LinalgOp linalgOp) {
  // Check all iterator types are parallel
  for (auto iterType : linalgOp.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel) {
      LDBG("Operation has non-parallel iterator type");
      return false;
    }
  }

  // Check all indexing maps are permutations
  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    if (!map.isPermutation()) {
      LDBG("Operation has non-permutation indexing map: " << map);
      return false;
    }
  }

  return true;
}

LogicalResult checkSingleUse(Value value, Operation *expectedUser) {
  // Collect all unique operations that use this value
  llvm::SmallPtrSet<Operation *, 4> users;
  for (OpOperand &use : value.getUses()) {
    users.insert(use.getOwner());
  }

  // Check that only one unique operation uses the value
  if (users.size() != 1) {
    LDBG("Value is used by " << users.size()
         << " different operations, expected exactly 1");
    return failure();
  }

  // If expectedUser is provided, verify it matches
  if (expectedUser) {
    Operation *actualUser = *users.begin();
    if (actualUser != expectedUser) {
      LDBG("Value's single user is not the expected operation");
      return failure();
    }
  }

  return success();
}

LogicalResult checkInPlaceSemantics(Operation *consumerOp, Value loopResult) {
  // Consumer must be a DestinationStyleOp
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(consumerOp);
  if (!dstOp) {
    LDBG("Consumer is not a DestinationStyleOpInterface");
    return failure();
  }

  // Check if the loop result is used as one of the DPS inits
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  bool isInPlace = llvm::is_contained(dpsInits, loopResult);

  if (!isInPlace) {
    LDBG("Loop result is not used as a DPS init (not in-place)");
    return failure();
  }

  return success();
}

LogicalResult findConsumedLoopResult(Operation *consumerOp,
                                     scf::ForOp targetForOp,
                                     scf::ForOp &actualSourceLoop,
                                     Value &consumedLoopResult) {
  // Search through consumer's operands to find one from a loop result
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
          LDBG("Found consumed loop result from "
               << (sourceLoop == targetForOp ? "target loop" : "ancestor loop"));
          return success();
        }
      }
    }
  }

  LDBG("Consumer does not use any result from target loop or its ancestors");
  return failure();
}

FailureOr<tensor::InsertSliceOp>
findInsertSliceRecursive(Value loopResult, scf::ForOp sourceLoop,
                         scf::ForOp targetForOp) {
  // Verify that loopResult is indeed a result of sourceLoop
  auto opResult = dyn_cast<OpResult>(loopResult);
  if (!opResult || opResult.getOwner() != sourceLoop.getOperation()) {
    LDBG("loopResult is not a result of sourceLoop");
    return failure();
  }

  // Get the result index
  unsigned resultIndex = opResult.getResultNumber();

  // Get the corresponding yielded value from the loop's terminator
  auto yieldOp = cast<scf::YieldOp>(sourceLoop.getBody()->getTerminator());
  Value yieldedValue = yieldOp.getOperand(resultIndex);

  // CRITICAL: Check if we've reached the target loop level
  if (sourceLoop == targetForOp) {
    // At target loop level, the yielded value MUST be an insert_slice
    // It cannot be a nested loop, because we need the insert_slice to be
    // directly in targetForOp's loop body for proper fusion semantics
    if (auto insertSlice = yieldedValue.getDefiningOp<tensor::InsertSliceOp>()) {
      LDBG("Found insert_slice at target loop level");
      return insertSlice;
    } else {
      LDBG("Yielded value is not an insert_slice at target loop level");
      return failure();
    }
  }

  // Not at target loop level yet, continue recursing
  // The yielded value should be a result from a nested scf.for loop
  if (auto nestedForResult = dyn_cast<OpResult>(yieldedValue)) {
    if (auto nestedFor = dyn_cast<scf::ForOp>(nestedForResult.getOwner())) {
      LDBG("Recursing into nested loop");
      // Recurse into the nested loop
      return findInsertSliceRecursive(yieldedValue, nestedFor, targetForOp);
    }
  }

  // The yielded value is neither insert_slice (at target level) nor a nested loop
  LDBG("Unsupported pattern: yielded value is neither insert_slice nor nested loop");
  return failure();
}

//===----------------------------------------------------------------------===//
// Main Validation Entry Point
//===----------------------------------------------------------------------===//

FailureOr<EltwiseFusionPreconditions>
validateAndPrepareEltwiseFusion(Operation *consumerOp, scf::ForOp targetForOp) {
  LDBG("=== Starting validation for element-wise fusion ===");

  // Step 1: Validate that the consumer is a linalg operation
  auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumerOp);
  if (!consumerLinalgOp) {
    LDBG("Consumer is not a linalg operation");
    return failure();
  }

  // Step 2: Validate that the consumer is element-wise
  if (!isEltwiseOperation(consumerLinalgOp)) {
    LDBG("Consumer is not an element-wise operation");
    return failure();
  }
  LDBG("✓ Consumer is element-wise");

  // Step 3: Find which loop result the consumer uses
  scf::ForOp actualSourceLoop;
  Value consumedLoopResult;
  if (failed(findConsumedLoopResult(consumerOp, targetForOp, actualSourceLoop,
                                    consumedLoopResult))) {
    return failure();
  }
  LDBG("✓ Found consumed loop result");

  // Step 4: Validate single-use requirement
  if (failed(checkSingleUse(consumedLoopResult, consumerOp))) {
    LDBG("Loop result does not have single use by consumer");
    return failure();
  }
  LDBG("✓ Loop result has single use");

  // Step 5: Validate in-place semantics
  if (failed(checkInPlaceSemantics(consumerOp, consumedLoopResult))) {
    return failure();
  }
  LDBG("✓ In-place semantics validated");

  // Step 6: Find the insert_slice operation
  FailureOr<tensor::InsertSliceOp> candidateSliceOp =
      findInsertSliceRecursive(consumedLoopResult, actualSourceLoop, targetForOp);
  if (failed(candidateSliceOp)) {
    LDBG("Could not find insert_slice in target loop's body");
    return failure();
  }
  LDBG("✓ Found insert_slice operation");

  // Step 7: Find the result number and consumer operand number
  auto yieldOp = cast<scf::YieldOp>(targetForOp.getBody()->getTerminator());
  unsigned resultNumber = 0;
  bool found = false;
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    // Need to trace through potential nested loops
    Value yieldOperand = yieldOp.getOperand(i);

    // If sourceLoop == targetForOp, check directly
    if (actualSourceLoop == targetForOp) {
      if (yieldOperand.getDefiningOp() == candidateSliceOp->getOperation()) {
        resultNumber = i;
        found = true;
        break;
      }
    } else {
      // For nested cases, check if this yield operand leads to our consumed result
      if (auto opResult = dyn_cast<OpResult>(yieldOperand)) {
        if (opResult.getOwner()->getResult(opResult.getResultNumber()) ==
            consumedLoopResult.getDefiningOp()->getResult(
                cast<OpResult>(consumedLoopResult).getResultNumber())) {
          resultNumber = i;
          found = true;
          break;
        }
      }
    }
  }

  if (!found) {
    // Fallback: use the result number from consumed loop result
    if (auto opResult = dyn_cast<OpResult>(consumedLoopResult)) {
      resultNumber = opResult.getResultNumber();
    } else {
      LDBG("Could not determine result number");
      return failure();
    }
  }

  // Find consumer operand number
  unsigned consumerOperandNumber = 0;
  found = false;
  for (OpOperand &operand : consumerOp->getOpOperands()) {
    if (operand.get() == consumedLoopResult) {
      consumerOperandNumber = operand.getOperandNumber();
      found = true;
      break;
    }
  }

  if (!found) {
    LDBG("Could not find consumer operand that uses loop result");
    return failure();
  }

  LDBG("=== Validation successful ===");
  LDBG("  Result number: " << resultNumber);
  LDBG("  Consumer operand number: " << consumerOperandNumber);

  // Return validated preconditions
  return EltwiseFusionPreconditions{*candidateSliceOp, consumerOp, targetForOp,
                                    resultNumber, consumerOperandNumber,
                                    consumedLoopResult};
}

} // namespace mlir
