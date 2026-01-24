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

def DeduplicateMultiBuffer : Pass<"deduplicate-multi-buffer", "func::FuncOp"> {
  let summary = "Deduplicate redundant multi-buffer allocations across sibling loops";
  let description = [{
    This pass eliminates redundant multi-buffer allocations created by the
    MultiBufferPass when processing sequentially-executed sibling loops.

    After consumer fusion (e.g., matmul + elementwise operations), the IR often
    contains multiple sibling loops that execute sequentially. The MultiBufferPass
    processes each loop independently, creating duplicate multi-buffer allocations
    for the same data even though these allocations could be safely shared.

    The pass analyzes data flow through DMA operations to identify:
    1. Parent-child relationships based on memory hierarchy (Global → GSM → SM/AM)
    2. Sibling allocations with the same parent and type
    3. Allocations used in non-overlapping loop iterations

    Algorithm:
    - Constructs a parent→children map by analyzing DMA operations
    - Processes nodes layer by layer using BFS from function parameters
    - Merges sibling allocations of the same type within each layer
    - Inherits children from merged siblings to the canonical node

    Example transformation:
      %alloc   = memref.alloc() : memref<2x576x512xf32, gsm>  // Keep
      %alloc_3 = memref.alloc() : memref<2x576x512xf32, gsm>  // Delete (merge to alloc)
      // Both serve the same role but in different K-loop iterations
  }];

  let constructor = "mlir::createDeduplicateMultiBufferPass()";

  let dependentDialects = [
    "mtdsp::MTDSPDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
  ];
}

def ChainSplitReductionPipelines : Pass<"chain-split-reduction-pipelines", "func::FuncOp"> {
  let summary = "Chain prefetch pipelines across split reduction loops";
  let description = [{
    This pass connects the software pipelining across split reduction loops
    that were divided to enable consumer fusion.

    After consumer fusion with reduction axis splitting (e.g., matmul K-loop
    split into K=0:1536 and K=1536:2048), each loop segment has independent
    prefetch logic. This creates a pipeline gap between the two segments,
    introducing unnecessary DMA startup latency.

    Problem example:
      // First K-loop (0→1536): prefetches only until 1536
      %1 = scf.for %k = 0 to 1536 ... {
        %cond = cmpi slt, %next_k, 1536  // ⚠️ Stops at 1536
        scf.if %cond {
          dma_opt ...  // Prefetch next iteration
        }
      }

      // Standalone prefetch before second loop ⚠️ Redundant!
      %2 = dma_opt ...  // Prefetch k=1536

      // Second K-loop (1536→2048)
      %3 = scf.for %k = 1536 to 2048 iter_args(%token = %2) ...

    This pass performs three transformations:

    1. **Extend prefetch boundary**: Modify the first loop's prefetch condition
       from `cmpi slt, %next_k, 1536` to `cmpi slt, %next_k, 2048`, allowing
       it to prefetch for the second loop's first iteration.

    2. **Remove redundant prefetch**: Delete the standalone DMA operation
       before the second loop, as it's now covered by the first loop's
       extended prefetch.

    3. **Connect iter_args**: Change the second loop's initial token from
       the standalone prefetch result to the first loop's result, creating
       a continuous pipeline chain.

    This optimization works in conjunction with DeduplicateMultiBuffer:
    - DeduplicateMultiBuffer: Merges redundant buffer allocations (space)
    - ChainSplitReductionPipelines: Connects prefetch pipelines (time)

    Recommended pass ordering:
      1. FuseEltwiseConsumer (generates split loops)
      2. MultiBuffer (adds multi-buffering)
      3. DeduplicateMultiBuffer (merges buffers) ← Run before this pass
      4. ChainSplitReductionPipelines (chains pipelines)

    Benefits:
    - Eliminates pipeline bubble between split loops
    - Reduces DMA startup latency
    - Improves overall throughput for reduction-fused operations
  }];

  let constructor = "mlir::createChainSplitReductionPipelinesPass()";

  let dependentDialects = [
    "mtdsp::MTDSPDialect",
    "scf::SCFDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
  ];
}

#endif // SCHEDULE_TRANSOFORM_PASSES