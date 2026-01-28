#ifndef SCHEDULE_TRANSOFORM_PASSES_H
#define SCHEDULE_TRANSOFORM_PASSES_H

#include "Dialect/Schedule/IR/ScheduleDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace func {
class FuncOp;
} // namespace func

} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "Dialect/Schedule/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createStaticizeTensorEmptyPass();

std::unique_ptr<mlir::Pass> createMultiBufferPass();

std::unique_ptr<mlir::Pass> createParallelPass();

std::unique_ptr<mlir::Pass> createUnrollPass();

std::unique_ptr<mlir::Pass> createOneShotBufferizeWithMemorySpacePass();

std::unique_ptr<mlir::Pass> createCustomCanonicializePass();

std::unique_ptr<mlir::Pass> createDeduplicateMultiBufferPass();

std::unique_ptr<mlir::Pass> createChainSplitReductionPipelinesPass();

std::unique_ptr<mlir::Pass> createBufferLoopSinkingPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Schedule/Transforms/Passes.h.inc"

} // namespace mlir

#endif // SCHEDULE_TRANSOFORM_PASSES_H