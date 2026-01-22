//===-- Passes.td - Schedule dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULE_TRANSOFORM_PASSES
#define SCHEDULE_TRANSOFORM_PASSES

include "mlir/Pass/PassBase.td"

def StaticizeTensorEmpty : Pass<"staticize-tensor-empty", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::createStaticizeTensorEmptyPass()";
  let dependentDialects = [
    "schedule::ScheduleDialect",
  ];
}

def MultiBuffer : Pass<"multi-buffer", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::createMultiBufferPass()";
  let dependentDialects = [
    "schedule::ScheduleDialect",
  ];
}

def Parallel : Pass<"parallel", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::createParallelPass()";
  let dependentDialects = [
    "schedule::ScheduleDialect",
  ];
}

def Unroll : Pass<"unroll", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::createUnrollPass()";
  let dependentDialects = [
    "schedule::ScheduleDialect",
  ];
}

def OneShotBufferizeWithMemorySpace : Pass<"one-shot-bufferize-with-memory-space", "ModuleOp"> {
  let summary = "One-Shot Bufferize with memory space preservation from tensor.empty";
  let description = [{
    This pass applies One-Shot Bufferize with custom type converters that preserve
    memory space attributes from tensor.empty operations.

    Key features:
    1. Extracts memory space from tensor.empty's memorySpace attribute
    2. Applies it to the resulting memref type
    3. Uses identity layout map for function boundaries
    4. Uses memref.copy for buffer copies
  }];

  let constructor = "mlir::createOneShotBufferizeWithMemorySpacePass()";

  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
  ];

  let options = [
    Option<"allowReturnAllocsFromLoops", "allow-return-allocs-from-loops",
           "bool", /*default=*/"false",
           "Allows returning/yielding new allocations from a loop.">,
    Option<"allowUnknownOps", "allow-unknown-ops", "bool",
           /*default=*/"false",
           "Allows unknown (not bufferizable) ops in the input IR.">,
    Option<"bufferizeFunctionBoundaries", "bufferize-function-boundaries",
           "bool", /*default=*/"true",
           "Bufferize function boundaries.">,
    Option<"dumpAliasSets", "dump-alias-sets", "bool", /*default=*/"false",
           "Test only: Annotate tensor IR with alias sets">,
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Test only: Only run inplaceability analysis and annotate IR">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Test only: Annotate IR with RaW conflicts">,
    Option<"checkParallelRegions", "check-parallel-regions", "bool",
           /*default=*/"true", "Account for parallel regions in RaW analysis.">
  ];
}

def CustomCanonicalize : Pass<"custom-canonicalize"> {
  let summary = "Custom canonicalize pass that preserves single-iteration loops";
  let description = [{
    This pass applies all standard canonicalization patterns from all loaded
    dialects, but explicitly excludes the SimplifyTrivialLoops pattern for
    scf.for operations. This preserves single-iteration loops for subsequent
    optimization passes.

    The pass collects canonicalization patterns automatically from:
    - All loaded dialects
    - All registered operations

    Then filters out specific patterns that would unroll single-iteration loops.
  }];

  let constructor = "mlir::createCustomCanonicializePass()";
}

#endif // SCHEDULE_TRANSOFORM_PASSES