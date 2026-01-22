//===- ConsumerFusion.h - Local consumer fusion implementation --*- C++ -*-===//
//
// This file declares the local implementation of consumer fusion that can
// be customized for different fusion strategies.
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULE_TRANSFORMOPS_CONSUMERFUSION_H
#define SCHEDULE_TRANSFORMOPS_CONSUMERFUSION_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

/// Local implementation of consumer fusion.
/// This is a modified version of mlir::scf::tileAndFuseConsumerOfSlice that
/// accepts an explicit consumer operation instead of inferring it from the slice.
///
/// Key modifications:
/// 1. Accepts explicit consumer operation (bypasses single-use requirement)
/// 2. Does not use getUntiledConsumerFromSlice to infer consumer
/// 3. Allows fusion in multi-layer nested loop scenarios
///
/// \param rewriter The rewriter to use for IR modifications
/// \param candidateSliceOp The insert_slice operation to fuse consumer into
/// \param consumerOp The explicit consumer operation to fuse (must use the loop result)
/// \returns The fusion result containing fused consumer and new loop, or failure
FailureOr<scf::SCFFuseConsumerOfSliceResult>
localTileAndFuseConsumerOfSlice(RewriterBase &rewriter,
                                 Operation *candidateSliceOp,
                                 Operation *consumerOp);

/// In-place fusion for element-wise consumers into spatial loops (Layer 3).
/// This strategy is designed for spatial (non-reduction) loops where the consumer
/// can be safely fused into every iteration without correctness issues.
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
///
/// Transformation strategy:
/// 1. Create a new loop at the consumer's position
/// 2. Move the original loop's body to the new loop (using mergeBlocks)
/// 3. Fuse the consumer into the loop body
///
/// Example transformation:
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
/// \param rewriter The rewriter to use for IR modifications
/// \param candidateSliceOp The insert_slice operation in the producer loop (precondition: inside scf.for)
/// \param consumerOp The element-wise consumer operation to fuse (precondition: validated by Layer 2)
/// \returns The fusion result with the fused consumer and modified loop, or failure
FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseEltwiseConsumerInPlace(RewriterBase &rewriter,
                           tensor::InsertSliceOp candidateSliceOp,
                           Operation *consumerOp);

/// In-place fusion with split-reduction for element-wise consumers (Layer 3).
/// This strategy is designed for reduction loops where the consumer should only
/// execute after the reduction completes.
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
///
/// Transformation strategy:
/// 1. Modify the original loop's upper bound to exclude the last iteration
/// 2. Create a second loop for the last iteration and clone the loop body
/// 3. Fuse the consumer into the second loop body
/// 4. Update the yield operation in the second loop
/// 5. Cleanup and replace uses
///
/// Example transformation:
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
/// Key insight:
/// The loop is identified as a reduction loop when the insert_slice is doing
/// in-place updates (offsets = [0, 0], sizes = dest shape). This means every
/// iteration updates the same location, which is reduction semantics. The consumer
/// should only execute after the final reduction result is computed.
///
/// \param rewriter The rewriter to use for IR modifications
/// \param candidateSliceOp The insert_slice operation in the producer loop (precondition: inside scf.for)
/// \param consumerOp The element-wise consumer operation to fuse (precondition: validated by Layer 2)
/// \returns The fusion result with the fused consumer and modified loop, or failure
FailureOr<scf::SCFFuseConsumerOfSliceResult>
fuseEltwiseConsumerInPlaceWithSplitReduction(
    RewriterBase &rewriter,
    tensor::InsertSliceOp candidateSliceOp,
    Operation *consumerOp);

} // namespace mlir

#endif // SCHEDULE_TRANSFORMOPS_CONSUMERFUSION_H
