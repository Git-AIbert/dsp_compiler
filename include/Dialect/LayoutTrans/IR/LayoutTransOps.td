#ifndef LAYOUTTRANS_OPS
#define LAYOUTTRANS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "Dialect/LayoutTrans/IR/LayoutTransBase.td"

//===----------------------------------------------------------------------===//
// LayoutTrans Operations
//===----------------------------------------------------------------------===//

class LayoutTrans_Op<string mnemonic, list<Trait> traits = []> :
    Op<LayoutTrans_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Img2ColOp
//===----------------------------------------------------------------------===//

def Img2ColOp : LayoutTrans_Op<"img2col", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "图像到列变换操作";
  let description = [{
    Img2Col 操作将输入图像张量 [N,C,H,W] 重排为列形式 [CxRxS, NxPxQ]，
    其中 R 和 S 是卷积核尺寸，P 和 Q 是输出特征图尺寸。

    这种变换允许将卷积操作转换为高效的矩阵乘法。

    示例：
    ```mlir
    %result = layout_trans.img2col ins(%input : tensor<2x3x226x226xf32>) 
                                 outs(%output : tensor<27x100352xf32>) 
                                 kernel_height(3) kernel_width(3) : tensor<27x100352xf32>
    ```
  }];

  let arguments = (ins
    AnyShaped:$input,
    AnyShaped:$output,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$kernel_height,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$kernel_width
  );
  let results = (outs
    Optional<AnyRankedTensor>:$result
  );
  
  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    `kernel_height` `(` $kernel_height `)`
    `kernel_width` `(` $kernel_width `)`
    attr-dict `:` type($result)
  }];

  // 保持原有的声明内容
  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output, "int32_t":$kernel_height, "int32_t":$kernel_width), [{
      build($_builder, $_state, output.getType(), input, output,
            $_builder.getI32IntegerAttr(kernel_height),
            $_builder.getI32IntegerAttr(kernel_width));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Col2ImgOp
//===----------------------------------------------------------------------===//

def Col2ImgOp : LayoutTrans_Op<"col2img", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "列到图像变换操作";
  let description = [{
    Col2Img 操作将列形式 [CxRxS, NxPxQ] 重排为图像张量 [N,C,H,W]，
    是 Img2Col 的逆操作，通常用于转置卷积或梯度计算。
    
    示例：
    ```mlir
    %result = layout_trans.col2img ins(%input : tensor<27x100352xf32>)
                                 outs(%output : tensor<2x3x226x226xf32>) : tensor<2x3x226x226xf32>
    ```
  }];

  let arguments = (ins
    AnyShaped:$input,
    AnyShaped:$output
  );
  let results = (outs
    Optional<AnyRankedTensor>:$result
  );
  
  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output), [{
      build($_builder, $_state, output.getType(), input, output);
    }]>
  ];
}

#endif // LAYOUTTRANS_OPS