#ifndef MTDSP_OPS
#define MTDSP_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

include "Dialect/MTDSP/IR/MTDSPBase.td"

//===----------------------------------------------------------------------===//
// MTDSP Operations
//===----------------------------------------------------------------------===//

class MTDSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MTDSP_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def ThreadIdOp : MTDSP_Op<"thread_id", [
    Pure,
    // DeclareOpInterfaceMethods<InferIntRangeInterface>
]> {
  let description = [{
    Returns the thread id, i.e. the index of the current thread within the cluster.

    Example:

    ```mlir
    %tIdX = mtdsp.thread_id : i32
    ```
  }];
  let results = (outs I32:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getI32Type());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GroupSizeOp
//===----------------------------------------------------------------------===//

def GroupSizeOp : MTDSP_Op<"group_size", [
    Pure,
    // DeclareOpInterfaceMethods<InferIntRangeInterface>
]> {
  let description = [{
    Returns the number of threads within a group.

    Example:

    ```mlir
    %gSz = mtdsp.group_size : i32
    ```
  }];
  let results = (outs I32:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getI32Type());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// GroupBarrierOp
//===----------------------------------------------------------------------===//

def GroupBarrierOp : MTDSP_Op<"group_barrier", [
]> {
  let summary = "Thread group barrier synchronization operation";
  let description = [{
    The `mtdsp.group_barrier` operation synchronizes all threads within a thread group.
    All threads must reach the barrier before any thread can proceed past it.
    
    The barrier ID operand allows multiple distinct barriers to be used within the
    same thread group.

    Example:
    ```mlir
    // Wait at barrier 0 until all threads in group arrive
    mtdsp.group_barrier %0 : i32
    ```
  }];

  let arguments = (ins I32:$barrier_id);
  let assemblyFormat = [{
    $barrier_id `:` type($barrier_id) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def AllocOp : MTDSP_Op<"alloc", [
]> {
  let description = [{
    DSP memory allocation operation.

    Example:

    ```mlir
    %buf1 = mtdsp.alloc : memref<128xf32, 0>           // global memory
    %buf2 = mtdsp.alloc(%n) : memref<?xf32, 1>         // global shared memory
    %buf3 = mtdsp.alloc(%m, %n) : memref<?x?xf32, 2>   // private
    ```
  }];
  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs AnyMemRef:$memref);

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getMemref().getType()); }
  }];
  let assemblyFormat = [{
    (`(` $dynamicSizes^ `)`)? attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def DeallocOp : MTDSP_Op<"dealloc", [
]> {
  let summary = "Memory deallocation operation for different memory spaces";
  let description = [{
    The `mtdsp.dealloc` operation releases allocated memory in different memory spaces.
    Based on the memory space attribute of the memref, it will be lowered to different
    deallocation functions:

    * For scalar memory space: calls `int scalar_free(void *ptr)`
    * For vector memory space: calls `int vector_free(void *ptr)`

    Both functions return 0 on success and -1 on failure. Note that when deallocating 
    either scalar or vector cache areas, the data in the cache will not be automatically 
    written back to memory. Explicit write-back must be handled by the programmer if needed.

    Example:
    ```mlir 
    mtdsp.dealloc %memref : memref<1024x1024xf32, #mtdsp.address_space<sm>>
    ```
  }];
  let arguments = (ins AnyMemRef:$memref);
  let assemblyFormat = [{
    $memref attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//

def DMAOp : MTDSP_Op<"dma", [
]> {
  let summary = "DSP direct memory access transfer operation";
  let description = [{
    The `mtdsp.dma` operation initiates a DMA transfer between two memory spaces and 
    returns the logical channel number used for the transfer.

    The operation requires that source and destination addresses cannot both be in 
    core memory space (AM/SM) simultaneously. It returns an i32 value representing 
    the logical DMA channel allocated for this transfer.

    Example:
    ```mlir 
    %channel = mtdsp.dma %src, %dst : memref<1024x1024xf32>, memref<1024x1024xf32, #mtdsp.address_space<gsm>> -> i32
    ```
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let results = (outs I32:$channel);
  let assemblyFormat = [{
    $src`,` $dst `:` type($src)`,` type($dst) `->` type($channel) attr-dict
  }];
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst), [{
      build($_builder, $_state, $_builder.getI32Type(), src, dst);
    }]>
  ];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// DMAOptOp
//===----------------------------------------------------------------------===//

def DMAOptOp : MTDSP_Op<"dma_opt", [
]> {
  let summary = "DSP direct memory access transfer operation with specified channel";
  let description = [{
    The `mtdsp.dma_opt` operation initiates a DMA transfer between two memory spaces using 
    a specified channel number and returns the logical channel number used for the transfer.

    The operation requires that source and destination addresses cannot both be in 
    core memory space (AM/SM) simultaneously. It accepts a channel number as input and
    returns an i32 value representing the logical DMA channel used for this transfer.

    This operation is an optimized version of the standard DMA operation where the channel
    can be explicitly specified rather than automatically allocated.

    Example:
    ```mlir 
    %result = mtdsp.dma_opt %src, %dst, %channel : memref<1024x1024xf32>, memref<1024x1024xf32, #mtdsp.address_space<gsm>>, i32 -> i32
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$src,
    AnyMemRef:$dst,
    I32:$channel
  );
  
  let results = (outs I32:$result);
  
  let assemblyFormat = [{
    $src`,` $dst`,` $channel `:` type($src)`,` type($dst)`,` type($channel) `->` type($result) attr-dict
  }];
  
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst, "Value":$channel), [{
      build($_builder, $_state, $_builder.getI32Type(), src, dst, channel);
    }]>
  ];
  
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : MTDSP_Op<"wait", [
]> {
  let summary = "Wait for DMA transfer completion";
  let description = [{
    The `mtdsp.wait` operation blocks until DMA transfer on the specified logical 
    channel completes.

    The operation requires that an asynchronous DMA transfer has been initiated on 
    the specified channel before calling this operation. The channel ID must be 
    between 0-15.

    Example:
    ```mlir
    // Wait for DMA transfer on channel 1 to complete
    mtdsp.wait %1 : i32
    ```
  }];
  let arguments = (ins I32:$channel);
  let assemblyFormat = [{
    $channel `:` type($channel) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// MatmulR6C96Op
//===----------------------------------------------------------------------===//

def MatmulR6C96Op : MTDSP_Op<"matmul_r6c96", [
]> {
  let summary = "6x96 matrix multiplication microkernel operation";
  let description = [{
    Performs a 6x96 matrix multiplication microkernel operation. This operation represents
    a single microkernel call that processes a 6xK and Kx96 block of input matrices.
    The actual memref dimensions can be smaller or dynamic, as this operation handles
    the boundary cases correctly.

    Will be lowered to an optimized assembly implementation.
    
    Operand requirements:
    - lhs: Left matrix with dimensions (?xK) or (MxK), processes up to 6 rows per call
    - rhs: Right matrix with dimensions (Kx?) or (KxN), processes up to 96 columns per call
    - dst: Output matrix with dimensions (?x?) or (MxN)
    
    The operation will handle cases where the actual matrix dimensions are smaller
    than the 6x96 microkernel size by padding internally.

    Example:
    ```mlir
    // Dynamic shapes
    mtdsp.matmul_r6c96 %A, %B, %C : 
      memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>

    // Static shapes smaller than microkernel
    mtdsp.matmul_r6c96 %A, %B, %C :
      memref<4x8xf32>, memref<8x64xf32>, memref<4x64xf32>
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$lhs,
    AnyMemRef:$rhs, 
    AnyMemRef:$dst
  );

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst)
  }];
}

//===----------------------------------------------------------------------===//
// MatmulR6C128Op
//===----------------------------------------------------------------------===//

def MatmulR6C128Op : MTDSP_Op<"matmul_r6c128", [
]> {
  let summary = "6x128 matrix multiplication microkernel operation";
  let description = [{
    Performs a 6x128 matrix multiplication microkernel operation. This operation represents
    a single microkernel call that processes a 6xK and Kx128 block of input matrices.
    The actual memref dimensions can be smaller or dynamic, as this operation handles
    the boundary cases correctly.

    Will be lowered to an optimized assembly implementation.
    
    Operand requirements:
    - lhs: Left matrix with dimensions (?xK) or (MxK), processes up to 6 rows per call
    - rhs: Right matrix with dimensions (Kx?) or (KxN), processes up to 128 columns per call
    - dst: Output matrix with dimensions (?x?) or (MxN)
    
    The operation will handle cases where the actual matrix dimensions are smaller
    than the 6x128 microkernel size by padding internally.

    Example:
    ```mlir
    // Dynamic shapes
    mtdsp.matmul_r6c128 %A, %B, %C : 
      memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>

    // Static shapes smaller than microkernel
    mtdsp.matmul_r6c128 %A, %B, %C :
      memref<4x8xf32>, memref<8x96xf32>, memref<4x96xf32>
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$lhs,
    AnyMemRef:$rhs, 
    AnyMemRef:$dst
  );

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst)
  }];
}

#endif // MTDSP_OPS