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

    The `targets` op handle may be associated with one or more payload IR values,
    and cache read will be performed one by one.

    #### Return modes
    
    The return handle points to the defining ops of the cached values.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$targets,
                       StrAttr:$memory_space
                       );
  let results = (outs TransformHandleTypeInterface:$cached);
  let assemblyFormat = [{
    $targets attr-dict `:` functional-type(operands, results)
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
                       StrAttr:$memory_space,
                       Optional<TransformValueHandleTypeInterface>:$cache_write_to
                       );
  let results = (outs TransformHandleTypeInterface:$cached);
  let assemblyFormat = [{
    $targets $cache_write_to attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS