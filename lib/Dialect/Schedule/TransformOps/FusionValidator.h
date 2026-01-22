//===- FusionValidator.h - Validation for eltwise fusion -*- C++ -*-===//
//
// This file declares validation logic for element-wise consumer fusion.
// All semantic validations are centralized here to ensure clean separation
// between validation (Layer 2) and IR transformation (Layer 3).
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULE_TRANSFORMOPS_FUSIONVALIDATOR_H
#define SCHEDULE_TRANSFORMOPS_FUSIONVALIDATOR_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Preconditions Structure
//===----------------------------------------------------------------------===//

/// Validated preconditions for element-wise fusion.
/// This structure is returned after successful validation and contains
/// all necessary information to perform the fusion transformation.
struct EltwiseFusionPreconditions {
  /// The insert_slice operation in the producer loop
  tensor::InsertSliceOp candidateSliceOp;

  /// The element-wise consumer operation to fuse
  Operation *consumerOp;

  /// The containing loop where fusion will happen
  scf::ForOp containingLoop;

  /// The loop result number that the consumer uses
  unsigned resultNumber;

  /// The operand number in consumer that uses the loop result
  unsigned consumerOperandNumber;

  /// The loop result value consumed by the consumer
  Value consumedLoopResult;
};

//===----------------------------------------------------------------------===//
// Validation Functions
//===----------------------------------------------------------------------===//

/// Check if a linalg operation is element-wise.
/// An operation is element-wise if:
/// 1. All iterator types are "parallel" (no reduction dimensions)
/// 2. All indexing maps are permutations (no broadcasting or reduction)
///
/// \param linalgOp The linalg operation to check
/// \returns true if the operation is element-wise, false otherwise
bool isEltwiseOperation(linalg::LinalgOp linalgOp);

/// Check if a value has only one unique user operation.
/// Note: A value can be used multiple times by the same operation (e.g., as both
/// input and output in linalg ops), but should only be used by ONE operation.
///
/// \param value The value to check
/// \param expectedUser The operation we expect to be the only user (optional)
/// \returns success if the value has exactly one user, failure otherwise
LogicalResult checkSingleUse(Value value, Operation *expectedUser = nullptr);

/// Check if the consumer operation uses the loop result as a DPS init.
/// This is required for in-place element-wise fusion.
///
/// \param consumerOp The consumer operation (must be DestinationStyleOpInterface)
/// \param loopResult The loop result value
/// \returns success if the loop result is used as a DPS init, failure otherwise
LogicalResult checkInPlaceSemantics(Operation *consumerOp, Value loopResult);

/// Find which loop result the consumer actually uses.
/// The consumer might use an outer loop's result, not necessarily targetForOp.
/// This function searches through the consumer's operands and checks if any
/// comes from a loop that is an ancestor of or equal to targetForOp.
///
/// \param consumerOp The consumer operation
/// \param targetForOp The target containing loop
/// \param[out] actualSourceLoop The actual loop whose result is consumed
/// \param[out] consumedLoopResult The loop result value
/// \returns success if a valid loop result is found, failure otherwise
LogicalResult findConsumedLoopResult(Operation *consumerOp,
                                     scf::ForOp targetForOp,
                                     scf::ForOp &actualSourceLoop,
                                     Value &consumedLoopResult);

/// Recursively find the insert_slice operation by tracing through nested
/// scf.for loops, but STOP at targetForOp level.
///
/// This function handles multi-layer nested loops by recursively following
/// the data flow chain from sourceLoop down to targetForOp:
/// - Start with a value that is a result of sourceLoop
/// - Get the corresponding yielded value from sourceLoop's yield op
/// - If sourceLoop == targetForOp:
///     - The yielded value MUST be an insert_slice (not a nested loop)
///     - This ensures the insert_slice is directly in targetForOp's body
/// - If sourceLoop != targetForOp:
///     - The yielded value should be a nested scf.for result -> recurse
/// - Otherwise -> fail
///
/// \param loopResult The SSA value that is a result of sourceLoop
/// \param sourceLoop The current loop we're examining
/// \param targetForOp The target containing loop where fusion should happen
/// \returns The insert_slice operation, or failure if not found
FailureOr<tensor::InsertSliceOp>
findInsertSliceRecursive(Value loopResult, scf::ForOp sourceLoop,
                         scf::ForOp targetForOp);

//===----------------------------------------------------------------------===//
// Main Validation Entry Point
//===----------------------------------------------------------------------===//

/// Validate all preconditions for element-wise consumer fusion.
/// This is the unified entry point for Layer 2 validation.
///
/// Validates:
/// 1. Consumer is a linalg operation
/// 2. Consumer is element-wise (all parallel iterators, permutation maps)
/// 3. Consumer uses a loop result from targetForOp or its ancestor
/// 4. Loop result has only one user (the consumer)
/// 5. Consumer uses loop result as a DPS init (in-place requirement)
/// 6. insert_slice operation exists in targetForOp's direct body
///
/// \param consumerOp The consumer operation to validate and fuse
/// \param targetForOp The target containing loop
/// \returns Validated preconditions on success, failure otherwise
FailureOr<EltwiseFusionPreconditions>
validateAndPrepareEltwiseFusion(Operation *consumerOp, scf::ForOp targetForOp);

} // namespace mlir

#endif // SCHEDULE_TRANSFORMOPS_FUSIONVALIDATOR_H
