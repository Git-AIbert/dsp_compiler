//===-- RemoveMemRefAddressSpace.cpp - Remove MemRef address spaces ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that removes address space attributes from all
// MemRef types in the module.
//
//===----------------------------------------------------------------------===//

#include "Dialect/MTDSP/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_REMOVEMEMREFADDRESSSPACE
#include "Dialect/MTDSP/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

class RemoveMemRefAddressSpacePass
    : public impl::RemoveMemRefAddressSpaceBase<RemoveMemRefAddressSpacePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk through all operations and remove address spaces
    module.walk([&](Operation *op) {
      // Handle function operations specially to update signatures
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        updateFunctionSignature(funcOp);
      }

      // Process all operation results
      for (OpResult result : op->getResults()) {
        Type newType = removeAddressSpace(result.getType());
        if (newType != result.getType()) {
          result.setType(newType);
        }
      }

      // Process all block arguments in regions
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            Type newType = removeAddressSpace(arg.getType());
            if (newType != arg.getType()) {
              arg.setType(newType);
            }
          }
        }
      }
    });
  }

private:
  /// Remove address space from a type if it's a MemRef type
  Type removeAddressSpace(Type type) {
    if (auto memrefType = type.dyn_cast<MemRefType>()) {
      if (memrefType.getMemorySpace()) {
        return MemRefType::get(memrefType.getShape(),
                              memrefType.getElementType(),
                              memrefType.getLayout(),
                              nullptr);  // Remove address space
      }
    } else if (auto unrankedMemrefType = type.dyn_cast<UnrankedMemRefType>()) {
      if (unrankedMemrefType.getMemorySpace()) {
        return UnrankedMemRefType::get(unrankedMemrefType.getElementType(),
                                       nullptr);
      }
    }
    return type;
  }

  /// Update function signature by removing address spaces from argument and result types
  void updateFunctionSignature(func::FuncOp funcOp) {
    bool changed = false;

    // Process argument types
    SmallVector<Type> newArgTypes;
    for (Type argType : funcOp.getArgumentTypes()) {
      Type newType = removeAddressSpace(argType);
      newArgTypes.push_back(newType);
      if (newType != argType) {
        changed = true;
      }
    }

    // Process result types
    SmallVector<Type> newResultTypes;
    for (Type resultType : funcOp.getResultTypes()) {
      Type newType = removeAddressSpace(resultType);
      newResultTypes.push_back(newType);
      if (newType != resultType) {
        changed = true;
      }
    }

    // Update function type if changed
    if (changed) {
      auto newFuncType = FunctionType::get(
          funcOp.getContext(), newArgTypes, newResultTypes);
      funcOp.setType(newFuncType);

      // Update block argument types in entry block
      Block &entryBlock = funcOp.getBody().front();
      for (auto [arg, newType] : llvm::zip(entryBlock.getArguments(), newArgTypes)) {
        if (arg.getType() != newType) {
          arg.setType(newType);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createRemoveMemRefAddressSpacePass() {
  return std::make_unique<RemoveMemRefAddressSpacePass>();
}
