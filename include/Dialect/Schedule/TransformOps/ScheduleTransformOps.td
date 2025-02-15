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
    $targets $cache_write_to (`multi_buffer` `=` $multi_buffer^)? attr-dict `:` functional-type(operands, results)
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

    For instance, given the input IR:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    }
    ```

    If the `targets` handle points to the scf.for op, the IR after transformation
    will have the loop marked with a parallel attribute:

    ```mlir
    scf.for %i = %c0 to %c16 step %c1 {
      "some_op"(%i) : (index) -> ()
    } { parallel }
    ```

    The parallel attribute can be used by subsequent passes to transform the loop
    into a parallel loop or perform other parallelization optimizations.

    The `targets` op handle may be associated with one or more payload IR values,
    and parallel marking will be performed one by one.

    #### Return modes
    
    The return handle points to the marked loop operations.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$targets);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = [{
    $targets attr-dict `:` functional-type(operands, results)
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

#endif // SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS