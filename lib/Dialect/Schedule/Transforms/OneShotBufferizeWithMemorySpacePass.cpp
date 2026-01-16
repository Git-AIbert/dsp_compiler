//===- OneShotBufferizeWithMemorySpacePass.cpp ---------------------------===//
//
// Implements One-Shot Bufferize with memory space preservation
//
//===----------------------------------------------------------------------===//

#include "Dialect/Schedule/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_ONESHOTBUFFERIZEWITHMEMORYSPACE
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct OneShotBufferizeWithMemorySpacePass
    : public impl::OneShotBufferizeWithMemorySpaceBase<
          OneShotBufferizeWithMemorySpacePass> {

  using OneShotBufferizeWithMemorySpaceBase::
      OneShotBufferizeWithMemorySpaceBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Configure One-Shot Bufferization options
    OneShotBufferizationOptions options;

    // Set options from pass parameters
    options.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
    options.allowUnknownOps = allowUnknownOps;
    options.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    options.dumpAliasSets = dumpAliasSets;
    options.testAnalysisOnly = testAnalysisOnly;
    options.printConflicts = printConflicts;
    options.checkParallelRegions = checkParallelRegions;

    // Set function argument type converter with identity layout map
    options.functionArgTypeConverterFn =
        [](TensorType tensorType, Attribute memorySpace,
           func::FuncOp funcOp,
           const BufferizationOptions &options) -> BaseMemRefType {
      return MemRefType::get(tensorType.getShape(),
                             tensorType.getElementType(),
                             /*layout=*/MemRefLayoutAttrInterface(),
                             memorySpace);
    };

    // Set memory copy function to use memref.copy
    options.memCpyFn = [](OpBuilder &b, Location loc, Value from, Value to) {
      b.create<memref::CopyOp>(loc, from, to);
      return success();
    };

    // Set unknown type converter to extract memory space from tensor.empty
    options.unknownTypeConverterFn =
        [](Value value, Attribute memorySpace,
           const BufferizationOptions &options) -> BaseMemRefType {
      auto tensorType = cast<TensorType>(value.getType());

      // If the value comes from tensor.empty, extract its memorySpace attribute
      if (auto emptyOp = value.getDefiningOp<tensor::EmptyOp>()) {
        if (auto addrSpaceAttr = emptyOp->getAttr("memorySpace")) {
          memorySpace = addrSpaceAttr;
        }
      }

      return MemRefType::get(tensorType.getShape(),
                             tensorType.getElementType(),
                             MemRefLayoutAttrInterface(),
                             memorySpace);
    };

    // Run One-Shot Bufferize
    if (failed(runOneShotBufferize(module, options))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createOneShotBufferizeWithMemorySpacePass() {
  return std::make_unique<OneShotBufferizeWithMemorySpacePass>();
}
