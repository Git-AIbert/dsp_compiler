//===- RemoveFunctionReturnsPass.cpp - Remove function return values -----===//
//
// This pass removes function return values, converting functions to use
// in-place modification semantics.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Dialect/Schedule/Transforms/Passes.h"

#define DEBUG_TYPE "remove-function-returns"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_REMOVEFUNCTIONRETURNS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// RemoveFunctionReturns Pass
//===----------------------------------------------------------------------===//

struct RemoveFunctionReturnsPass
    : public impl::RemoveFunctionReturnsBase<RemoveFunctionReturnsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each function in the module
    for (auto funcOp : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (failed(processFunction(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult processFunction(func::FuncOp funcOp) {
    // Get function type
    auto funcType = funcOp.getFunctionType();

    // Check if function has return values
    if (funcType.getNumResults() == 0) {
      LDBG("Function " << funcOp.getName() << " has no return values, skipping");
      return success();
    }

    LDBG("Processing function: " << funcOp.getName());

    // Find all return operations
    SmallVector<func::ReturnOp> returnOps;
    funcOp.walk([&](func::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
    });

    if (returnOps.empty()) {
      LDBG("No return operations found");
      return success();
    }

    // Update all return operations to return nothing
    for (auto returnOp : returnOps) {
      OpBuilder builder(returnOp);
      builder.create<func::ReturnOp>(returnOp.getLoc());
      returnOp->erase();
    }

    // Update function signature to have no return values
    auto newFuncType = FunctionType::get(
        funcOp.getContext(),
        funcType.getInputs(),
        TypeRange{}  // No return types
    );

    funcOp.setFunctionType(newFuncType);

    LDBG("Updated function signature for: " << funcOp.getName());

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createRemoveFunctionReturnsPass() {
  return std::make_unique<RemoveFunctionReturnsPass>();
}
