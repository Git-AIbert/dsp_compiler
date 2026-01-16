//===-- Passes.td - MTDSP dialect pass definition file --------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTDSP_TRANSFORM_PASSES
#define MTDSP_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def PromoteAllocsToArguments : Pass<"promote-allocs-to-arguments", "func::FuncOp"> {
  let summary = "Promote mtdsp.alloc operations without memory space or with Global space to function arguments";
  let description = [{
    This pass transforms mtdsp.alloc operations that either:
    1. Have no memory space attribute, or
    2. Have a Global memory space attribute

    into function parameters. This is a prerequisite pass that must run before
    the MTDSPToLLVM conversion pass.

    The pass performs the following transformations:
    - Identifies all mtdsp.alloc ops without memory space or with Global space
    - Extends the function signature to include these allocations as parameters
    - Replaces all uses of the alloc results with the new function parameters
    - Removes the original alloc operations
    - Emits warnings if call sites need to be updated

    Example:
    ```mlir
    // Before:
    func @kernel() {
      %0 = mtdsp.alloc() : memref<1024xf32>
      // use %0
    }

    // After:
    func @kernel(%arg0: memref<1024xf32>) {
      // use %arg0
    }
    ```
  }];

  let constructor = "mlir::createPromoteAllocsToArgumentsPass()";

  let dependentDialects = [
    "mtdsp::MTDSPDialect",
    "func::FuncDialect",
  ];
}

def RemoveMemRefAddressSpace : Pass<"remove-memref-address-space", "ModuleOp"> {
  let summary = "Remove address space attributes from all MemRef types";
  let description = [{
    This pass removes address space attributes from all MemRef types in the module.
    It processes:
    - Operation operands and results with MemRef types
    - Function signatures (argument and result types)
    - Block arguments with MemRef types
    - Both ranked MemRefType and UnrankedMemRefType

    This pass should be run after MTDSPToLLVM conversion as a post-processing step
    to clean up any remaining address space attributes that may cause issues
    in downstream compilation.

    Example:
    ```mlir
    // Before:
    func.func @foo(%arg0: memref<1024xf32, 1>) {
      %0 = memref.alloc() : memref<256xf32, 2>
      // ...
    }

    // After:
    func.func @foo(%arg0: memref<1024xf32>) {
      %0 = memref.alloc() : memref<256xf32>
      // ...
    }
    ```
  }];

  let constructor = "mlir::createRemoveMemRefAddressSpacePass()";

  let dependentDialects = [
    "func::FuncDialect",
  ];
}

#endif // MTDSP_TRANSFORM_PASSES
