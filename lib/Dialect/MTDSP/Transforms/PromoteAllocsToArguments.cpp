//===-- PromoteAllocsToArguments.cpp - Promote allocs to function args ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that promotes mtdsp.alloc operations without
// memory space attribute or with Global memory space to function arguments.
//
//===----------------------------------------------------------------------===//

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/MTDSP/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_PROMOTEALLOCSTOARGUMENTS
#include "Dialect/MTDSP/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

class PromoteAllocsToArgumentsPass
    : public impl::PromoteAllocsToArgumentsBase<PromoteAllocsToArgumentsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* context = &getContext();

    // Step 1: Collect all mtdsp.alloc operations without memory space attribute
    // or with Global memory space
    SmallVector<mtdsp::AllocOp, 4> allocsToPromote;
    funcOp.walk([&](mtdsp::AllocOp allocOp) {
      MemRefType memrefTy = allocOp.getType();
      mlir::Attribute attr = memrefTy.getMemorySpace();

      // Check if alloc has no memory space attribute
      if (!attr) {
        allocsToPromote.push_back(allocOp);
        return;
      }

      // Check if alloc has Global memory space attribute
      if (auto addrSpaceAttr = mlir::dyn_cast<mtdsp::AddressSpaceAttr>(attr)) {
        if (addrSpaceAttr.getValue() == mtdsp::AddressSpace::Global) {
          allocsToPromote.push_back(allocOp);
        }
      }
    });

    if (allocsToPromote.empty())
      return;

    // Step 2: Create new function type with additional parameters
    FunctionType oldFuncType = funcOp.getFunctionType();
    SmallVector<Type, 8> newInputTypes(oldFuncType.getInputs().begin(),
                                      oldFuncType.getInputs().end());

    // Map from alloc operation to new parameter index
    llvm::DenseMap<Operation*, unsigned> allocToParamIndex;
    unsigned baseParamIndex = oldFuncType.getInputs().size();

    for (auto allocOp : allocsToPromote) {
      newInputTypes.push_back(allocOp.getType());
      allocToParamIndex[allocOp] = baseParamIndex++;
    }

    FunctionType newFuncType = FunctionType::get(context,
                                                newInputTypes,
                                                oldFuncType.getResults());

    // Step 3: Update function signature
    funcOp.setType(newFuncType);

    // Step 4: Add new block arguments to entry block
    Block& entryBlock = funcOp.getBody().front();
    for (auto allocOp : allocsToPromote) {
      entryBlock.addArgument(allocOp.getType(), allocOp.getLoc());
    }

    // Step 5: Replace each alloc's uses with corresponding function argument
    for (auto allocOp : allocsToPromote) {
      unsigned paramIndex = allocToParamIndex[allocOp];
      Value newArg = entryBlock.getArgument(paramIndex);
      allocOp.getResult().replaceAllUsesWith(newArg);
      allocOp.erase();
    }

    // Step 6: Check for call sites that need updating
    auto module = funcOp->getParentOfType<ModuleOp>();
    if (module) {
      SmallVector<func::CallOp, 4> callsToUpdate;
      module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == funcOp.getName()) {
          callsToUpdate.push_back(callOp);
        }
      });

      // Emit warning if there are call sites that need updating
      if (!callsToUpdate.empty()) {
        funcOp.emitWarning() << "Found " << callsToUpdate.size()
                             << " call site(s) that need to be updated to match "
                             << "the new function signature";
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createPromoteAllocsToArgumentsPass() {
  return std::make_unique<PromoteAllocsToArgumentsPass>();
}
