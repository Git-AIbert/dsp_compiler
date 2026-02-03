#ifndef SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS
#define SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

def CacheReadOp : Op<Transform_Dialect, "structured.cache_read", [
    // FunctionalStyleTransformOpTrait,
    DeclareOpInterfaceMethods<TransformOpInterface>, 
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  ]> {

  let summary = "Create a cache read of original tensor for readers.";

  let description = [{
    Create a cache read of the values given by the `targets` op handle.
    For instance, given the input IR:

    ```mlir
    "some_op"(%a) : tensor<16x16xf16> -> ()
    ```

    If the `targets` handle points to `%a`, the IR after transformation is:

    ```mlir
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
    "some_op"(%cached) : tensor<16x16xf16> -> ()
    ```

    If `multi_buffer` is set to true, the generated linalg.copy will be marked with 
    a multi_buffer attribute:
    ```mlir
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) { multi_buffer }
    "some_op"(%cached) : tensor<16x16xf16> -> ()
    ```

    The `targets` op handle may be associated with one or more payload IR values,
    and cache read will be performed one by one.

    #### Return modes
    
    The return handle points to the defining ops of the cached values.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$targets,
                      AnyAttr:$memory_space,
                      DefaultValuedAttr<BoolAttr, "false">:$multi_buffer
                      );
  let results = (outs TransformHandleTypeInterface:$cached);
  let assemblyFormat = [{
    $targets (`multi_buffer` `=` $multi_buffer^)? attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

def CacheWriteOp : Op<Transform_Dialect, "structured.cache_write", [
    // FunctionalStyleTransformOpTrait,
    DeclareOpInterfaceMethods<TransformOpInterface>, 
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {

  let summary = "Create a cache write of original tensor, before storing into tensor.";

  let description = [{
    Create a cache write of the values given by the `targets` op handle.
    The `targets` payload values should be the SSA result of linalg/mtfusion operations.

    For instance, given the input IR:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    ```

    If the `targets` handle points to `%a`, the IR after transformation is:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
    ```

    If `multi_buffer` is set to true, the generated linalg.copy will be marked with 
    a multi_buffer attribute:
    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) { multi_buffer }
    ```

    If `cache_write_to` handle is provided and points to `%res`, 
    the IR after transformation is:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%res : tensor<16x16xf16>)
    ```

    The `targets` op handle may be associated with one or more payload IR values,
    and cache write will be performed one by one.

    #### Return modes
    
    The return handle points to the defining ops of the cached values.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$targets,
                      AnyAttr:$memory_space,
                      DefaultValuedAttr<BoolAttr, "false">:$multi_buffer,
                      Optional<TransformValueHandleTypeInterface>:$cache_write_to
                      );
  let results = (outs TransformHandleTypeInterface:$cached);
  let assemblyFormat = [{
    $targets (`,` $cache_write_to^)? (`multi_buffer` `=` $multi_buffer^)? attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MarkParallelOp
//===----------------------------------------------------------------------===//

def MarkParallelOp : Op<Transform_Dialect, "mark_parallel", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {

  let summary = "Mark the target loops for parallel execution.";

  let description = [{
    Mark the loops given by the `targets` op handle for parallel execution.
    The `targets` payload values should be loop operations (e.g., scf.for).
    The `num_threads` specifies the number of threads to use for parallel execution.

    For instance, given the input IR:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    }
    ```

    If the `targets` handle points to the scf.for op and num_threads is set to 4, 
    the IR after transformation will be:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    } { num_threads = 4 }
    ```

    The thread count will be used by subsequent passes to transform the loop 
    into a parallel loop with specified number of threads.

    #### Return modes
    
    The return handle points to the marked loop operations.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins 
    TransformHandleTypeInterface:$targets,
    I32Attr:$num_threads
  );
  
  let results = (outs TransformHandleTypeInterface:$transformed);
  
  let assemblyFormat = [{
    $targets `num_threads` `=` $num_threads attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MarkUnrollOp
//===----------------------------------------------------------------------===//

def MarkUnrollOp : Op<Transform_Dialect, "mark_unroll", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {

  let summary = "Mark the target loops for unrolling.";

  let description = [{
    Mark the loops given by the `targets` op handle for loop unrolling.
    The `targets` payload values should be loop operations (e.g., scf.for).
    The `unroll_factor` specifies the factor by which to unroll the loop.

    For instance, given the input IR:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    }
    ```

    If the `targets` handle points to the scf.for op and unroll_factor is set to 4, 
    the IR after transformation will be:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    } { unroll_factor = 4 }
    ```

    The unroll factor will be used by subsequent passes to transform the loop 
    into an unrolled version with the specified factor.

    #### Return modes
    
    The return handle points to the marked loop operations.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins 
    TransformHandleTypeInterface:$targets,
    I32Attr:$unroll_factor
  );
  
  let results = (outs TransformHandleTypeInterface:$transformed);
  
  let assemblyFormat = [{
    $targets `unroll_factor` `=` $unroll_factor attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MarkVectorizeOp
//===----------------------------------------------------------------------===//

def MarkVectorizeOp : Op<Transform_Dialect, "mark_vectorize", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {

  let summary = "Mark the target loops for vectorization.";

  let description = [{
    Mark the loops given by the `targets` op handle for vectorization.
    The `targets` payload values should be loop operations (e.g., scf.for).

    For instance, given the input IR:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    }
    ```

    If the `targets` handle points to the scf.for op, the IR after transformation
    will have the loop marked for vectorization (using attributes):

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    } { vectorize }
    ```

    The `targets` op handle may be associated with one or more payload IR values,
    and vectorization marking will be performed one by one.

    #### Return modes

    The return handle points to the loops marked for vectorization.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$targets);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = [{
    $targets attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// FuseConsumerIntoContainingOp
//===----------------------------------------------------------------------===//

def FuseConsumerIntoContainingOp : Op<Transform_Dialect, "structured.fuse_consumer_into_containing_op", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    ReportTrackingListenerFailuresOpTrait
  ]> {

  let summary = "Fuse consumer operations into a containing loop operation.";

  let description = [{
    Fuses a consumer operation into a containing loop operation (e.g., scf.for).
    This transformation is the counterpart to `fuse_into_containing_op` but works
    in the opposite direction: instead of fusing producers into loops, it fuses
    consumers into loops.

    This operation uses `scf::tileAndFuseConsumerOfSlice` to perform the fusion.
    The consumer operation is tiled and fused into the containing loop, improving
    data locality by bringing the consumer closer to where data is produced.

    For instance, given the input IR after tiling a matmul:

    ```mlir
    %result = scf.for %i = %c0 to %c1024 step %c32 iter_args(%arg = %init) -> (tensor<1024x1024xf32>) {
      %slice_out = tensor.extract_slice %arg[%i, 0] [32, 1024] [1, 1]
      %matmul_slice = linalg.matmul ins(%A_slice, %B : ...) outs(%slice_out : ...)
      %inserted = tensor.insert_slice %matmul_slice into %arg[%i, 0] [32, 1024] [1, 1]
      scf.yield %inserted : tensor<1024x1024xf32>
    }
    %output = linalg.add ins(%result, %C : ...) outs(%empty : ...)
    ```

    After applying `fuse_consumer_into_containing_op` with consumer=%add and containing_op=%for:

    ```mlir
    %result = scf.for %i = %c0 to %c1024 step %c32 iter_args(%arg = %init) -> (tensor<1024x1024xf32>) {
      %slice_out = tensor.extract_slice %arg[%i, 0] [32, 1024] [1, 1]
      %matmul_slice = linalg.matmul ins(%A_slice, %B : ...) outs(%slice_out : ...)
      %inserted = tensor.insert_slice %matmul_slice into %arg[%i, 0] [32, 1024] [1, 1]
      // Consumer is fused into the loop
      %C_slice = tensor.extract_slice %C[%i, 0] [32, 1024] [1, 1]
      %add_slice = linalg.add ins(%inserted, %C_slice : ...) outs(...)
      %add_inserted = tensor.insert_slice %add_slice into ...
      scf.yield %add_inserted : tensor<1024x1024xf32>
    }
    ```

    #### Return modes

    This operation returns two handles:
    - `fused_consumer`: Handle to the tiled and fused consumer operation
    - `new_containing_op`: Handle to the new containing loop operation

    This operation consumes the `consumer_op` handle and reads the `containing_op` handle.

    The return values mirror `fuse_into_containing_op` for symmetry:
    - `fuse_into_containing_op` returns (fused_producer, new_containing_op)
    - `fuse_consumer_into_containing_op` returns (fused_consumer, new_containing_op)
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$containing_op);
  let results = (outs TransformHandleTypeInterface:$fused_consumer,
                      TransformHandleTypeInterface:$new_containing_op);

  let assemblyFormat = "$consumer_op `into` $containing_op attr-dict "
                       " `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// FuseEltwiseConsumerOp
//===----------------------------------------------------------------------===//

def FuseEltwiseConsumerOp : Op<Transform_Dialect, "structured.fuse_eltwise_consumer", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    ReportTrackingListenerFailuresOpTrait
  ]> {

  let summary = "Fuse element-wise consumer into containing loop with in-place optimization.";

  let description = [{
    Fuses an element-wise consumer operation into a containing loop operation.
    This is a specialized version of `fuse_consumer_into_containing_op` that:

    1. **Requires the consumer to be element-wise** - this enables more aggressive optimization
    2. **Performs in-place fusion** - the consumer can directly reuse the producer's output
       buffer, eliminating the need to preserve intermediate results during iteration
    3. **Checks that the producer has only this element-wise consumer** - ensures safe
       elimination of intermediate storage

    An operation is considered element-wise if:
    - All iterator types are "parallel" (no reduction dimensions)
    - All indexing maps are permutations (no broadcasting or reduction)

    For instance, given the input IR with a tiled matmul:

    ```mlir
    %result = scf.for %i = %c0 to %c1024 step %c32 iter_args(%arg = %init) -> tensor<1024x1024xf32> {
      %slice_A = tensor.extract_slice %A[%i, 0] [32, 1024] [1, 1]
      %slice_out = tensor.extract_slice %arg[%i, 0] [32, 1024] [1, 1]
      %matmul = linalg.matmul ins(%slice_A, %B : ...) outs(%slice_out : ...)
      %inserted = tensor.insert_slice %matmul into %arg[%i, 0] [32, 1024] [1, 1]
      scf.yield %inserted
    }
    %output = linalg.add ins(%result, %C : ...) outs(%result : ...)
    ```

    After applying `fuse_eltwise_consumer` with consumer=%add and containing_op=%for:

    ```mlir
    %result = scf.for %i = %c0 to %c1024 step %c32 iter_args(%arg = %init) -> tensor<1024x1024xf32> {
      %slice_A = tensor.extract_slice %A[%i, 0] [32, 1024] [1, 1]
      %slice_out = tensor.extract_slice %arg[%i, 0] [32, 1024] [1, 1]
      %matmul = linalg.matmul ins(%slice_A, %B : ...) outs(%slice_out : ...)
      // Element-wise consumer fused in-place
      %slice_C = tensor.extract_slice %C[%i, 0] [32, 1024] [1, 1]
      %add = linalg.add ins(%matmul, %slice_C : ...) outs(%matmul : ...) // In-place!
      %inserted = tensor.insert_slice %add into %arg[%i, 0] [32, 1024] [1, 1]
      scf.yield %inserted
    }
    ```

    Note: The intermediate %result tensor is completely eliminated, and %add reuses %matmul's buffer.

    #### Return modes

    This operation returns two handles:
    - `fused_consumer`: Handle to the tiled and fused consumer operation
    - `new_containing_op`: Handle to the new containing loop operation

    This operation consumes the `consumer_op` handle and reads the `containing_op` handle.

    #### Failure modes

    The operation will fail if:
    - Consumer is not an element-wise operation (has reduction or broadcast)
    - Producer has multiple consumers (cannot safely eliminate intermediate storage)
    - Producer's only consumer is not the specified consumer operation
    - containing_op is not an scf.for loop
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
                       TransformHandleTypeInterface:$containing_op);
  let results = (outs TransformHandleTypeInterface:$fused_consumer,
                      TransformHandleTypeInterface:$new_containing_op);

  let assemblyFormat = "$consumer_op `into` $containing_op attr-dict "
                       " `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// ApplyCustomCanonicalizationPatternsOp
//===----------------------------------------------------------------------===//

def ApplyCustomCanonicalizationPatternsOp
    : Op<Transform_Dialect, "apply_patterns.custom_canonicalization",
        [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Populates custom canonicalization patterns that preserve single-iteration loops";

  let description = [{
    This operation populates all standard canonicalization patterns from all loaded
    dialects and registered operations, but excludes the `SimplifyTrivialLoops` pattern
    for `scf.for` operations. This preserves single-iteration loops for subsequent
    optimization passes.

    The operation is designed to be used within `transform.apply_patterns` to enable
    canonicalization during transformation pipelines while maintaining the structure
    of single-iteration loops that may be important for later optimizations.

    Example usage:
    ```mlir
    transform.apply_patterns to %target {
      transform.apply_patterns.custom_canonicalization
    } : !transform.any_op
    ```

    This is equivalent to `transform.apply_patterns.canonicalization` except that
    it will not inline single-iteration `scf.for` loops into their bodies.
  }];

  let assemblyFormat = "attr-dict";
}

// This is roughly similar to OpFoldResult assuming the handle produces a single
// value in the payload IR.
def TransformAnyParamTypeOrAnyHandle : Type<
    Or<[TransformHandleTypeInterface.predicate,
        TransformParamTypeInterface.predicate]>,
    "transform any param type or any handle type">;

//===----------------------------------------------------------------------===//
// ExtendedTileUsingForOp
//===----------------------------------------------------------------------===//

def ExtendedTileUsingForOp : Op<Transform_Dialect, "structured.extended_tile_using_for",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Extended tile operation with aggressive post-tiling optimizations";

  let description = [{
    This is an extended version of `tile_using_for` that applies aggressive
    optimizations after the standard tiling transformation. It performs the same
    basic tiling operation as `tile_using_for`, but additionally analyzes the
    tiled code and applies targeted optimizations for specific patterns.

    **Core Functionality:**
    Identical to `tile_using_for` - tiles the target operation with specified
    sizes, generating SCF for loops with the tiled operation in the loop body.

    **Additional Optimizations:**

    1. **In-place Element-wise Optimization**: When the original operation has
       in-place semantics (an input is also used as output), and the operation
       is element-wise (all parallel iterators), this transform applies an
       aggressive optimization to avoid extra memory allocations during
       subsequent bufferization.

       Specifically, for element-wise operations where `input[i] == init[i]`,
       the transform rewrites the tiled input slice to be extracted from the
       loop's iter_args instead of the original function arguments. While the
       standard tiling is semantically correct, this optimization enables
       bufferization to reuse buffers in-place rather than allocating
       additional temporary buffers.

    **Example Transformation:**

    Input IR (in-place add):
    ```mlir
    func.func @add(%A: tensor<2304x1024xf32>, %B: tensor<2304x1024xf32>)
        -> tensor<2304x1024xf32> {
      %result = linalg.add
          ins(%A, %B : tensor<2304x1024xf32>, tensor<2304x1024xf32>)
          outs(%A : tensor<2304x1024xf32>) -> tensor<2304x1024xf32>
      return %result : tensor<2304x1024xf32>
    }
    ```

    Standard `tile_using_for` with tile_sizes=[0, 128] (conservative):
    ```mlir
    %0 = scf.for %arg2 = %c0 to %c1024 step %c128 iter_args(%arg3 = %arg0) {
      %slice_0 = tensor.extract_slice %arg0[0, %arg2] [2304, 128] [1, 1]
      //                                 ^^^^^ Original argument - may cause extra allocation
      %slice_1 = tensor.extract_slice %arg1[0, %arg2] [2304, 128] [1, 1]
      %slice_2 = tensor.extract_slice %arg3[0, %arg2] [2304, 128] [1, 1]
      %add = linalg.add ins(%slice_0, %slice_1 : ...) outs(%slice_2 : ...)
      %inserted = tensor.insert_slice %add into %arg3[0, %arg2] [2304, 128] [1, 1]
      scf.yield %inserted
    }
    ```

    `extended_tile_using_for` with tile_sizes=[0, 128] (aggressive):
    ```mlir
    %0 = scf.for %arg2 = %c0 to %c1024 step %c128 iter_args(%arg3 = %arg0) {
      %slice_0 = tensor.extract_slice %arg3[0, %arg2] [2304, 128] [1, 1]
      //                                 ^^^^^ Iter arg - enables in-place bufferization
      %slice_1 = tensor.extract_slice %arg1[0, %arg2] [2304, 128] [1, 1]
      %slice_2 = tensor.extract_slice %arg3[0, %arg2] [2304, 128] [1, 1]
      %add = linalg.add ins(%slice_0, %slice_1 : ...) outs(%slice_2 : ...)
      %inserted = tensor.insert_slice %add into %arg3[0, %arg2] [2304, 128] [1, 1]
      scf.yield %inserted
    }
    ```

    **When to Use:**
    - Element-wise operations with in-place semantics
    - When minimizing memory allocation is critical for performance
    - When you want aggressive optimizations that enable better bufferization
    - As a general replacement for `tile_using_for` when you want all available
      optimizations applied automatically

    **Safety:**
    All optimizations are applied conservatively - they only trigger when the
    transformation is provably safe and semantics-preserving. The operation
    will never produce incorrect code; at worst it behaves identically to
    `tile_using_for`.

    **Future Extensibility:**
    This operation is designed to be extended with additional post-tiling
    optimizations as needed, while maintaining a single consistent interface.

    #### Arguments and Return Values

    Identical to `tile_using_for`:
    - Arguments: target operation, tile sizes (static/dynamic), interchange, scalable sizes
    - Returns: tiled operation handle and loop handles

    #### Return modes

    Same as `tile_using_for`: returns handles to co-indexed lists of tiled
    operations and their surrounding loops.

    #### Failure modes

    Same failure modes as `tile_using_for` - the additional optimizations do not
    introduce new failure cases.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$dynamic_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$interchange,
                   DefaultValuedOptionalAttr<DenseBoolArrayAttr, "{}">:$scalable_sizes);
  let results = (outs TransformHandleTypeInterface:$tiled_linalg_op,
                      Variadic<TransformHandleTypeInterface>:$loops);
  let builders = [
    OpBuilder<(ins "TypeRange":$loopTypes,
                   "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange,
                   CArg<"std::optional<ArrayRef<bool>>", "std::nullopt">:
                      $scalableSizes)>,
    OpBuilder<(ins "TypeRange":$loopTypes,
                   "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange,
                   CArg<"std::optional<ArrayRef<bool>>", "std::nullopt">:
                      $scalableSizes)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange,
                   CArg<"std::optional<ArrayRef<bool>>", "std::nullopt">:
                      $scalableSizes)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange,
                   CArg<"std::optional<ArrayRef<bool>>", "std::nullopt">:
                      $scalableSizes)>,
  ];

  let assemblyFormat = [{
    $target
      `tile_sizes` custom<DynamicIndexList>(
        $dynamic_sizes,
        $static_sizes,
        $scalable_sizes)
      (`interchange` `=` $interchange^)?
    attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

//===----------------------------------------------------------------------===//
// FuseElementwiseGenericOps
//===----------------------------------------------------------------------===//

def FuseElementwiseGenericOps : Op<Transform_Dialect, "structured.fuse_elementwise_generic", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    ReportTrackingListenerFailuresOpTrait
  ]> {

  let summary = "Fuse two consecutive element-wise linalg.generic operations";

  let description = [{
    Fuses a producer element-wise `linalg.generic` operation with its consumer
    element-wise `linalg.generic` operation into a single fused `linalg.generic`.

    This transformation merges two consecutive element-wise operations by:
    1. Combining their input operands (excluding the producer-consumer edge)
    2. Merging their computation bodies (region fusion)
    3. Concatenating their `op_label` attributes (e.g., "add" + "relu" = "add_relu")

    **Requirements:**
    - Both operations must be `linalg.generic` (not named ops like `linalg.add`)
    - Both must be element-wise: all iterators are "parallel", all maps are permutations
    - Consumer must directly use producer's result as one of its inputs
    - Indexing maps must be compatible for fusion

    **Example transformation:**

    Before:
    ```mlir
    %add = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"],
      op_label = "add"
    } ins(%A, %B : tensor<2304x1024xf32>, tensor<2304x1024xf32>)
      outs(%C : tensor<2304x1024xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<2304x1024xf32>

    %c0 = arith.constant 0.0 : f32
    %relu = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"],
      op_label = "relu"
    } ins(%add : tensor<2304x1024xf32>)
      outs(%add : tensor<2304x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<2304x1024xf32>
    ```

    After fusion:
    ```mlir
    %c0 = arith.constant 0.0 : f32
    %fused = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"],
      op_label = "add_relu"
    } ins(%A, %B : tensor<2304x1024xf32>, tensor<2304x1024xf32>)
      outs(%C : tensor<2304x1024xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      %max = arith.maximumf %sum, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<2304x1024xf32>
    ```

    **Benefits:**
    - Reduces memory traffic (eliminates intermediate tensor)
    - Enables better vectorization and optimization opportunities
    - Creates primitive operations that can be matched to custom kernels

    #### Return modes

    Returns a handle to the fused `linalg.generic` operation.
    This operation consumes both the producer and consumer handles.

    #### Failure modes

    The operation will fail if:
    - Producer or consumer is not a `linalg.generic` operation
    - Either operation is not element-wise (has reduction dimensions or broadcasting)
    - Consumer does not directly use producer's result as an input
    - Operations have incompatible shapes or indexing maps
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer,
                       TransformHandleTypeInterface:$consumer);
  let results = (outs TransformHandleTypeInterface:$fused_op);

  let assemblyFormat = "$producer `,` $consumer attr-dict "
                       "`:` functional-type(operands, results)";
}

#endif // SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS