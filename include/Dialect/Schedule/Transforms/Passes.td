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

def OptimizeDMA : Pass<"optimize-dma", "func::FuncOp"> {
  let summary = "Optimize DMA operations with channel allocation and point-to-point wait";
  let description = [{
    This pass optimizes DMA operations by allocating hardware channels and
    converting to optimized operation variants.

    The pass analyzes DMA operations created by the multi-buffer pass and performs:
    1. DMA channel allocation based on operation grouping and usage patterns
    2. Conversion of mtdsp.dma to mtdsp.dma_opt with explicit channel parameters
    3. Conversion of mtdsp.wait to mtdsp.wait_p2p for point-to-point synchronization

    The optimization relies on attributes set by the multi-buffer pass:
    - 'group' attribute: Groups related DMA operations for channel sharing
    - 'position' attribute: Indicates operation position ("initial", "next", "current", "fixed")

    This pass should run after the multi-buffer pass in the optimization pipeline.
  }];

  let constructor = "mlir::createOptimizeDMAPass()";

  let dependentDialects = [
    "mtdsp::MTDSPDialect",
    "arith::ArithDialect",
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

def BufferLoopSinking : Pass<"buffer-loop-sinking", "func::FuncOp"> {
  let summary = "Sinks deallocation operations out of loop nests to match hoisted allocations";
  let description = [{
    This pass is designed to run after buffer-loop-hoisting and moves deallocation
    operations out of loop nests to match the hoisted allocation operations.

    After buffer-loop-hoisting moves allocations upward out of loops, the
    corresponding deallocations may still remain inside the loop body or in
    nested control flow. This pass aggressively sinks these deallocations
    downward to the same scope level as their allocations, minimizing memory
    management overhead within loops.

    Example:
    ```mlir
    // After buffer-loop-hoisting:
    %alloc = memref.alloc() : memref<1024xf32>  // Hoisted to outer scope
    scf.for %i = 0 to 10 {
      scf.for %j = 0 to 20 {
        // use %alloc
      }
      memref.dealloc %alloc : memref<1024xf32>  // Still inside outer loop
    }

    // After buffer-loop-sinking:
    %alloc = memref.alloc() : memref<1024xf32>
    scf.for %i = 0 to 10 {
      scf.for %j = 0 to 20 {
        // use %alloc
      }
    }
    memref.dealloc %alloc : memref<1024xf32>  // Sunk to match allocation scope
    ```

    The pass performs the following:
    - Identifies deallocation operations inside loop bodies
    - Verifies safety of moving deallocations (no uses after the loop)
    - Sinks deallocations to after the loop nest
    - Maintains proper dominance relationships

    Recommended pass ordering:
      1. buffer-loop-hoisting (hoists allocations)
      2. buffer-loop-sinking (sinks deallocations to match)

    Benefits:
    - Reduces deallocation overhead inside loops
    - Balances allocation/deallocation scope for better code clarity
    - Enables better buffer reuse and optimization opportunities
  }];

  let constructor = "mlir::createBufferLoopSinkingPass()";

  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
  ];
}

def RemoveFunctionReturns : Pass<"remove-function-returns", "ModuleOp"> {
  let summary = "Remove function return values and convert to in-place modification";
  let description = [{
    Removes function return values, converting functions to use in-place
    modification semantics. Subsequent canonicalization and CSE passes
    will eliminate any dead values.
  }];

  let constructor = "mlir::createRemoveFunctionReturnsPass()";

  let dependentDialects = [
    "func::FuncDialect",
  ];
}

#endif // SCHEDULE_TRANSOFORM_PASSES