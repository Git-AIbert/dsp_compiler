#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h" 

#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"
#include "Dialect/Schedule/Transforms/Passes.h"
#include "Dialect/LayoutTrans/IR/LayoutTransDialect.h"
#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Conversion/ConvertToMTDSP/ConvertToMTDSPPass.h"
#include "Conversion/MTDSPToLLVM/MTDSPToLLVMPass.h"

#define LOC builder.getUnknownLoc()

using namespace mlir;

// func::FuncOp createConv2DFunction(OpBuilder &builder, ModuleOp module) {
//     // 创建需要的类型
//     // 输入张量尺寸: N(batch) x C(输入通道) x H(高) x W(宽)
//     const int64_t N = 2;    // 批次大小
//     const int64_t C = 3;    // 输入通道数
//     const int64_t H = 224;  // 输入高度
//     const int64_t W = 224;  // 输入宽度
    
//     // 卷积核尺寸: K(输出通道) x C(输入通道) x KH(核高) x KW(核宽)
//     const int64_t K = 64;   // 输出通道数
//     const int64_t KH = 3;   // 卷积核高度
//     const int64_t KW = 3;   // 卷积核宽度
    
//     // 步长和膨胀
//     const int64_t strideH = 1;
//     const int64_t strideW = 1;
//     const int64_t dilationH = 1;
//     const int64_t dilationW = 1;
    
//     // 输出尺寸 (与输入相同)
//     const int64_t OH = H;  // 输出高度
//     const int64_t OW = W;  // 输出宽度
    
//     auto f32Type = builder.getF32Type();
    
//     // 创建输入、卷积核和输出张量类型
//     auto inputType = RankedTensorType::get({N, C, H, W}, f32Type);             // NCHW格式
//     auto kernelType = RankedTensorType::get({K, C, KH, KW}, f32Type);          // FCHW格式
//     auto outputType = RankedTensorType::get({N, K, OH, OW}, f32Type);          // NFHW格式
    
//     // 创建函数类型 (输入张量, 卷积核张量, 输出张量) -> 输出张量
//     auto functionType = builder.getFunctionType(
//         {inputType, kernelType, outputType},  // 输入类型
//         {outputType}                          // 结果类型
//     );
    
//     // 创建函数
//     builder.setInsertionPointToEnd(module.getBody());
//     auto funcOp = builder.create<func::FuncOp>(
//         LOC,                   // 位置
//         "conv2d",              // 函数名
//         functionType           // 函数类型
//     );
    
//     // 创建入口块并获取函数参数
//     auto* entryBlock = funcOp.addEntryBlock();
//     builder.setInsertionPointToStart(entryBlock);
    
//     // 获取参数
//     auto args = entryBlock->getArguments();
//     Value input = args[0];    // 输入张量
//     Value kernel = args[1];   // 卷积核张量
//     Value output = args[2];   // 输出张量
    
//     // 创建步长和膨胀属性
//     SmallVector<int64_t, 2> stridesValues = {strideH, strideW};
//     SmallVector<int64_t, 2> dilationsValues = {dilationH, dilationW};
    
//     auto stridesAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), stridesValues);
//     auto dilationsAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), dilationsValues);
    
//     // 计算需要的填充
//     const int64_t paddingH = (KH - 1) / 2;
//     const int64_t paddingW = (KW - 1) / 2;
    
//     // 创建填充常量（通常为0）
//     Value padConstant = builder.create<arith::ConstantOp>(
//         LOC, builder.getF32FloatAttr(0.0f));
    
//     // 创建填充操作 - 使用正确的构建方法
//     // 我们首先需要为填充操作创建静态大小信息
//     SmallVector<OpFoldResult> staticLow, staticHigh;
//     // N和C维度无需填充
//     staticLow.push_back(builder.getIndexAttr(0));
//     staticLow.push_back(builder.getIndexAttr(0));
//     // H和W维度需要填充
//     staticLow.push_back(builder.getIndexAttr(paddingH));
//     staticLow.push_back(builder.getIndexAttr(paddingW));
    
//     // 同样设置高位填充
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(paddingH));
//     staticHigh.push_back(builder.getIndexAttr(paddingW));
    
//     // 创建填充后的输入张量类型
//     auto paddedInputType = RankedTensorType::get(
//         {N, C, H + 2 * paddingH, W + 2 * paddingW}, f32Type);
    
//     // 使用正确的tensor::PadOp构建方式
//     Value paddedInput = builder.create<tensor::PadOp>(
//         LOC,
//         paddedInputType,
//         input,
//         staticLow,
//         staticHigh,
//         padConstant
//     );
    
//     // 创建linalg.conv_2d_nchw_fchw操作，使用填充后的输入
//     auto convOp = builder.create<linalg::Conv2DNchwFchwOp>(
//         LOC,                                // 位置
//         TypeRange{outputType},              // 结果类型
//         ValueRange{paddedInput, kernel},    // 输入操作数 (使用填充后的输入)
//         ValueRange{output},                 // 输出操作数
//         stridesAttr,                        // 步长属性
//         dilationsAttr                       // 膨胀属性
//     );
    
//     // 创建返回操作
//     builder.create<func::ReturnOp>(
//         LOC,
//         convOp.getResult(0)                 // 返回卷积结果
//     );
    
//     return funcOp;
// }

// func::FuncOp createConv2DFunction(OpBuilder &builder, ModuleOp module) {
//     // 创建需要的类型
//     // 输入张量尺寸: N(batch) x C(输入通道) x H(高) x W(宽)
//     const int64_t N = 2;    // 批次大小
//     const int64_t C = 3;    // 输入通道数
//     const int64_t H = 224;  // 输入高度
//     const int64_t W = 224;  // 输入宽度
    
//     // 卷积核尺寸: K(输出通道) x C(输入通道) x KH(核高) x KW(核宽)
//     const int64_t K = 64;   // 输出通道数
//     const int64_t KH = 3;   // 卷积核高度
//     const int64_t KW = 3;   // 卷积核宽度
    
//     // 步长和膨胀
//     const int64_t strideH = 1;
//     const int64_t strideW = 1;
//     const int64_t dilationH = 1;
//     const int64_t dilationW = 1;
    
//     // 输出尺寸 (与输入相同)
//     const int64_t OH = H;  // 输出高度
//     const int64_t OW = W;  // 输出宽度
    
//     auto f32Type = builder.getF32Type();
    
//     // 创建输入、卷积核和输出张量类型
//     auto inputType = RankedTensorType::get({N, C, H, W}, f32Type);             // NCHW格式
//     auto kernelType = RankedTensorType::get({K, C, KH, KW}, f32Type);          // FCHW格式
//     auto outputType = RankedTensorType::get({N, K, OH, OW}, f32Type);          // NFHW格式
    
//     // 创建函数类型 (输入张量, 卷积核张量, 输出张量) -> 输出张量
//     auto functionType = builder.getFunctionType(
//         {inputType, kernelType, outputType},  // 输入类型
//         {outputType}                          // 结果类型
//     );
    
//     // 创建函数
//     builder.setInsertionPointToEnd(module.getBody());
//     auto funcOp = builder.create<func::FuncOp>(
//         LOC,                   // 位置
//         "conv2d",              // 函数名
//         functionType           // 函数类型
//     );
    
//     // 创建入口块并获取函数参数
//     auto* entryBlock = funcOp.addEntryBlock();
//     builder.setInsertionPointToStart(entryBlock);
    
//     // 获取参数
//     auto args = entryBlock->getArguments();
//     Value input = args[0];    // 输入张量
//     Value kernel = args[1];   // 卷积核张量
//     Value output = args[2];   // 输出张量
    
//     // 创建步长和膨胀属性
//     SmallVector<int64_t, 2> stridesValues = {strideH, strideW};
//     SmallVector<int64_t, 2> dilationsValues = {dilationH, dilationW};
    
//     auto stridesAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), stridesValues);
//     auto dilationsAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), dilationsValues);
    
//     // 计算需要的填充
//     const int64_t paddingH = (KH - 1) / 2;
//     const int64_t paddingW = (KW - 1) / 2;
    
//     // 创建填充常量（通常为0）
//     Value padConstant = builder.create<arith::ConstantOp>(
//         LOC, builder.getF32FloatAttr(0.0f));
    
//     // 创建填充操作 - 使用正确的构建方法
//     // 我们首先需要为填充操作创建静态大小信息
//     SmallVector<OpFoldResult> staticLow, staticHigh;
//     // N和C维度无需填充
//     staticLow.push_back(builder.getIndexAttr(0));
//     staticLow.push_back(builder.getIndexAttr(0));
//     // H和W维度需要填充
//     staticLow.push_back(builder.getIndexAttr(paddingH));
//     staticLow.push_back(builder.getIndexAttr(paddingW));
    
//     // 同样设置高位填充
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(paddingH));
//     staticHigh.push_back(builder.getIndexAttr(paddingW));
    
//     // 创建填充后的输入张量类型
//     auto paddedInputType = RankedTensorType::get(
//         {N, C, H + 2 * paddingH, W + 2 * paddingW}, f32Type);
    
//     // 使用正确的tensor::PadOp构建方式
//     Value paddedInput = builder.create<tensor::PadOp>(
//         LOC,
//         paddedInputType,
//         input,
//         staticLow,
//         staticHigh,
//         padConstant
//     );

//     // 计算新的形状：将卷积核的后三个维度合为一个 [F, C*KH*KW]
//     auto reshapedKernelType = RankedTensorType::get({K, C * KH * KW}, f32Type);

//     // 创建新形状的常量张量
//     SmallVector<int64_t, 2> newKernelShape = {K, C * KH * KW};
//     auto newShapeType = RankedTensorType::get({2}, builder.getI64Type());
//     Value newShapeConstant = builder.create<arith::ConstantOp>(
//         LOC,
//         DenseIntElementsAttr::get(newShapeType, newKernelShape)
//     );

//     // 创建 ReshapeOp 将卷积核转换为新形状[64, 3x3x3]（[K, CxFHxFW]）
//     Value reshapedKernel = builder.create<tensor::ReshapeOp>(
//         LOC,
//         reshapedKernelType,    // 重塑后的张量类型
//         kernel,                // 原始卷积核
//         newShapeConstant       // 新形状常量
//     );

//     // img2col操作将paddedInput由[2,3,226,226]（[N,C,H,W]）转换为[3x3x3, 2x224x224]（[CxKHxKW, NxOHxOW]）
//     // 1. 首先创建 img2col 的输出类型
//     auto imgColOutType = RankedTensorType::get({C * KH * KW, N * OH * OW}, f32Type);

//     // 2. 创建一个空的张量作为 img2col 操作的输出缓冲区
//     Value imgColOutput = builder.create<tensor::EmptyOp>(
//         LOC,
//         imgColOutType.getShape(),
//         imgColOutType.getElementType()
//     );

//     // 3. 创建 layout_trans.img2col 操作
//     Value imgColResult = builder.create<layout_trans::Img2ColOp>(
//         LOC,
//         paddedInput,   // 输入张量 [2,3,226,226]
//         imgColOutput,  // 输出张量 [27,100352]
//         3, 3
//     ).getResult();

//     // 4. 现在可以使用 imgColResult 进行后续的矩阵乘法操作
//     // 创建矩阵乘法的输出缓冲区 [F, N*OH*OW] = [64, 100352]
//     auto matmulResultType = RankedTensorType::get({K, N * OH * OW}, f32Type);
//     Value matmulOutput = builder.create<tensor::EmptyOp>(
//         LOC,
//         matmulResultType.getShape(),
//         matmulResultType.getElementType()
//     );

//     // 5. 执行矩阵乘法: reshapedKernel [64,27] × imgColResult [27,100352]
//     Value matmulResult = builder.create<linalg::MatmulOp>(
//         LOC,
//         ValueRange{reshapedKernel, imgColResult},
//         ValueRange{matmulOutput}
//     ).getResult(0);
    
//     // 6. col2img操作将output由[64, 2x224x224]（[K, NxOHxOW]）转换为[2,64,224,224]（[N,K,OH,OW]）
//     Value colImgResult = builder.create<layout_trans::Col2ImgOp>(
//         LOC,
//         matmulResult,   // 输入张量 [64, 2x224x224]
//         output          // 输出张量 [2,64,224,224]
//     ).getResult();

//     // 返回转换后的输出
//     builder.create<func::ReturnOp>(
//         LOC,
//         colImgResult
//     );

//     // // 返回卷积结果
//     // builder.create<func::ReturnOp>(
//     //     LOC,
//     //     output
//     // );
    
//     return funcOp;
// }

// func::FuncOp createConv2DFunction(OpBuilder &builder, ModuleOp module) {
//     // 创建需要的类型
//     // 输入张量尺寸: N(batch) x C(输入通道) x H(高) x W(宽)
//     const int64_t N = 2;    // 批次大小
//     const int64_t C = 128;  // 输入通道数
//     const int64_t H = 56;   // 输入高度
//     const int64_t W = 56;   // 输入宽度
    
//     // 卷积核尺寸: K(输出通道) x C(输入通道) x KH(核高) x KW(核宽)
//     const int64_t K = 256;  // 输出通道数
//     const int64_t KH = 3;   // 卷积核高度
//     const int64_t KW = 3;   // 卷积核宽度
    
//     // 步长和膨胀
//     const int64_t strideH = 1;
//     const int64_t strideW = 1;
//     const int64_t dilationH = 1;
//     const int64_t dilationW = 1;
    
//     // 输出尺寸 (与输入相同)
//     const int64_t OH = H;  // 输出高度
//     const int64_t OW = W;  // 输出宽度
    
//     auto f32Type = builder.getF32Type();
    
//     // 创建输入、卷积核和输出张量类型
//     auto inputType = RankedTensorType::get({N, C, H, W}, f32Type);             // NCHW格式
//     auto kernelType = RankedTensorType::get({K, C, KH, KW}, f32Type);          // FCHW格式
//     auto outputType = RankedTensorType::get({N, K, OH, OW}, f32Type);          // NFHW格式
    
//     // 创建函数类型 (输入张量, 卷积核张量, 输出张量) -> 输出张量
//     auto functionType = builder.getFunctionType(
//         {inputType, kernelType, outputType},  // 输入类型
//         {outputType}                          // 结果类型
//     );
    
//     // 创建函数
//     builder.setInsertionPointToEnd(module.getBody());
//     auto funcOp = builder.create<func::FuncOp>(
//         LOC,                   // 位置
//         "conv2d",              // 函数名
//         functionType           // 函数类型
//     );
    
//     // 创建入口块并获取函数参数
//     auto* entryBlock = funcOp.addEntryBlock();
//     builder.setInsertionPointToStart(entryBlock);
    
//     // 获取参数
//     auto args = entryBlock->getArguments();
//     Value input = args[0];    // 输入张量
//     Value kernel = args[1];   // 卷积核张量
//     Value output = args[2];   // 输出张量
    
//     // 创建步长和膨胀属性
//     SmallVector<int64_t, 2> stridesValues = {strideH, strideW};
//     SmallVector<int64_t, 2> dilationsValues = {dilationH, dilationW};
    
//     auto stridesAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), stridesValues);
//     auto dilationsAttr = DenseIntElementsAttr::get(
//         RankedTensorType::get({2}, builder.getI64Type()), dilationsValues);
    
//     // 计算需要的填充
//     const int64_t paddingH = (KH - 1) / 2;
//     const int64_t paddingW = (KW - 1) / 2;
    
//     // 创建填充常量（通常为0）
//     Value padConstant = builder.create<arith::ConstantOp>(
//         LOC, builder.getF32FloatAttr(0.0f));
    
//     // 创建填充操作 - 使用正确的构建方法
//     // 我们首先需要为填充操作创建静态大小信息
//     SmallVector<OpFoldResult> staticLow, staticHigh;
//     // N和C维度无需填充
//     staticLow.push_back(builder.getIndexAttr(0));
//     staticLow.push_back(builder.getIndexAttr(0));
//     // H和W维度需要填充
//     staticLow.push_back(builder.getIndexAttr(paddingH));
//     staticLow.push_back(builder.getIndexAttr(paddingW));
    
//     // 同样设置高位填充
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(0));
//     staticHigh.push_back(builder.getIndexAttr(paddingH));
//     staticHigh.push_back(builder.getIndexAttr(paddingW));

//     // 获取填充后输入的尺寸
//     const int64_t paddedH = H + 2 * paddingH;
//     const int64_t paddedW = W + 2 * paddingW;
    
//     // 创建填充后的输入张量类型
//     auto paddedInputType = RankedTensorType::get(
//         {N, C, paddedH, paddedW}, f32Type);
    
//     // 使用正确的tensor::PadOp构建方式
//     Value paddedInput = builder.create<tensor::PadOp>(
//         LOC,
//         paddedInputType,
//         input,
//         staticLow,
//         staticHigh,
//         padConstant
//     );

//     // 创建img2col输出张量类型
//     // 修改形状为 [C, KH, KW, N, OH, OW]，这样更符合之后的矩阵乘法顺序
//     auto img2colType = RankedTensorType::get({C, KH, KW, N, OH, OW}, f32Type);
    
//     // 分配用于img2col结果的空张量
//     Value img2colOutput = builder.create<tensor::EmptyOp>(
//         LOC, 
//         img2colType.getShape(), 
//         img2colType.getElementType()
//     );
    
//     // 创建img2col操作的索引映射
//     // 输入索引映射: (c, kh, kw, n, oh, ow) -> (n, c, oh * strideH + kh * dilationH, ow * strideW + kw * dilationW)
//     // 输出索引映射: (c, kh, kw, n, oh, ow) -> (c, kh, kw, n, oh, ow)
    
//     // 构建输入和输出索引映射
//     MLIRContext* ctx = builder.getContext();
    
//     // 输入映射 - 注意，由于维度顺序的改变，索引位置也需要相应调整
//     AffineExpr c = getAffineDimExpr(0, ctx);
//     AffineExpr kh = getAffineDimExpr(1, ctx);
//     AffineExpr kw = getAffineDimExpr(2, ctx);
//     AffineExpr n = getAffineDimExpr(3, ctx);
//     AffineExpr oh = getAffineDimExpr(4, ctx);
//     AffineExpr ow = getAffineDimExpr(5, ctx);
    
//     // 构建输入索引表达式：oh * strideH + kh * dilationH 和 ow * strideW + kw * dilationW
//     AffineExpr hExpr = oh * strideH + kh * dilationH;
//     AffineExpr wExpr = ow * strideW + kw * dilationW;
    
//     // 构建完整的输入索引映射 - 注意顺序为(n, c, h, w)对应输入数据形状
//     AffineMap inputMap = AffineMap::get(
//         6,  // 维度数 (c, kh, kw, n, oh, ow)
//         0,  // 符号数
//         {n, c, hExpr, wExpr},  // 结果表达式
//         ctx
//     );
    
//     // 构建输出索引映射
//     AffineMap outputMap = AffineMap::get(
//         6,  // 维度数
//         0,  // 符号数
//         {c, kh, kw, n, oh, ow},  // 直接映射
//         ctx
//     );
    
//     // 创建迭代器类型数组，全部设为并行
//     SmallVector<utils::IteratorType, 6> iterTypes(6, utils::IteratorType::parallel);
    
//     // 创建linalg.generic操作实现img2col逻辑
//     Value img2col = builder.create<linalg::GenericOp>(
//         LOC,
//         TypeRange{img2colType},  // 结果类型
//         ValueRange{paddedInput},  // 张量输入
//         ValueRange{img2colOutput},  // 初始输出缓冲区
//         ArrayRef<AffineMap>{inputMap, outputMap},  // 输入和输出索引映射
//         iterTypes,  // 迭代器类型
//         [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
//             // 在img2col操作中，我们只是从填充后输入中提取值并直接输出
//             nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
//         }
//     ).getResult(0);
    
//     // 重塑img2col结果为适合矩阵乘法的形式 [C*KH*KW, N*OH*OW]
//     // 首先计算新的维度
//     const int64_t featureSize = C * KH * KW;         // 输入通道数 * 卷积核高度 * 卷积核宽度
//     const int64_t spatialSize = N * OH * OW;     // 批次大小 * 输出高度 * 输出宽度
    
//     // 创建重塑后的类型 - 注意这里维度是[featureSize, spatialSize]
//     auto reshapedImg2colType = RankedTensorType::get({featureSize, spatialSize}, f32Type);
    
//     // 执行重塑操作
//     SmallVector<ReassociationIndices, 2> img2colReassociation = {{0, 1, 2}, {3, 4, 5}};
    
//     Value reshapedImg2col = builder.create<tensor::CollapseShapeOp>(
//         LOC,
//         reshapedImg2colType,
//         img2col,
//         img2colReassociation  // 将[C,KH,KW]合并为一维，将[N,OH,OW]合并为一维
//     );
    
//     // 重塑卷积核为 [K, C*KH*KW] 形状
//     auto reshapedKernelType = RankedTensorType::get({K, featureSize}, f32Type);
    
//     SmallVector<ReassociationIndices, 2> kernelReassociation = {{0}, {1, 2, 3}};
    
//     Value reshapedKernel = builder.create<tensor::CollapseShapeOp>(
//         LOC,
//         reshapedKernelType,
//         kernel,
//         kernelReassociation  // 保持K不变，将[C,KH,KW]合并
//     );

//     // 创建用于矩阵乘法结果的空张量 - 注意结果形状为[K, N*OH*OW]
//     auto matmulResultType = RankedTensorType::get({K, spatialSize}, f32Type);
    
//     Value matmulOutput = builder.create<tensor::EmptyOp>(
//         LOC,
//         matmulResultType.getShape(),
//         matmulResultType.getElementType()
//     );

//     // 使用linalg.matmul执行矩阵乘法
//     // reshapedKernel[K, C*KH*KW] x reshapedImg2col[C*KH*KW, N*OH*OW] = matmulResult[K, N*OH*OW]
//     // 修正：根据错误信息，使用正确的构造方法
//     Value matmulResult = builder.create<linalg::MatmulOp>(
//         LOC,
//         TypeRange{matmulResultType},  // 返回类型
//         ValueRange{reshapedKernel, reshapedImg2col},  // 输入
//         ValueRange{matmulOutput}  // 输出
//     ).getResult(0);

//     // 重塑矩阵乘法结果回到 [K, N, OH, OW] 形状
//     auto reshapedMatmulType = RankedTensorType::get({K, N, OH, OW}, f32Type);
    
//     SmallVector<ReassociationIndices, 2> matmulReassociation = {{0}, {1, 2, 3}};
    
//     Value reshapedMatmul = builder.create<tensor::ExpandShapeOp>(
//         LOC,
//         reshapedMatmulType,
//         matmulResult,
//         matmulReassociation  // 保持K不变，将N*OH*OW拆分回[N,OH,OW]
//     );

//     // 转置结果，从 [K, N, outH, outW] 到 [N, K, outH, outW] (NCHW格式)
//     SmallVector<int64_t, 4> permutation = {1, 0, 2, 3};  // N, K, outH, outW

//     // 使用linalg.transpose操作进行转置
//     Value transposedResult = builder.create<linalg::TransposeOp>(
//         LOC,
//         reshapedMatmul,
//         output,
//         builder.getDenseI64ArrayAttr(permutation)
//     ).getResult()[0];
    
//     // 返回卷积结果
//     builder.create<func::ReturnOp>(
//         LOC,
//         transposedResult
//     );
    
//     return funcOp;
// }

func::FuncOp createConv2DFunction(OpBuilder &builder, ModuleOp module) {
    // 创建需要的类型
    // 输入张量尺寸: N(batch) x C(输入通道) x H(高) x W(宽)
    const int64_t N = 1;    // 批次大小
    const int64_t C = 64;   // 输入通道数
    const int64_t H = 58;   // 输入高度
    const int64_t W = 58;   // 输入宽度
    
    // 卷积核尺寸: K(输出通道) x C(输入通道) x KH(核高) x KW(核宽)
    const int64_t K = 64;   // 输出通道数
    const int64_t KH = 3;   // 卷积核高度
    const int64_t KW = 3;   // 卷积核宽度
    
    // 输出尺寸 (与输入相同)
    const int64_t OH = 56;  // 输出高度
    const int64_t OW = 56;  // 输出宽度
    
    auto f32Type = builder.getF32Type();

    // 卷积核分块大小
    const int64_t Ct = 64;
    const int64_t Kt = 64;
    
    // 创建输入、卷积核和输出张量类型
    auto inputType = RankedTensorType::get({N, C, H, W}, f32Type);             // NCHW格式
    auto kernelType = RankedTensorType::get({C/Ct, K/Kt, KH, KW, Ct, Kt}, f32Type);
    auto outputType = RankedTensorType::get({N, K, OH, OW}, f32Type);          // NFHW格式
    
    // 创建函数类型 (输入张量, 卷积核张量, 输出张量) -> 输出张量
    auto functionType = builder.getFunctionType(
        {inputType, kernelType, outputType},  // 输入类型
        {outputType}                          // 结果类型
    );
    
    // 创建函数
    builder.setInsertionPointToEnd(module.getBody());
    auto funcOp = builder.create<func::FuncOp>(
        LOC,                   // 位置
        "conv2d",       // 函数名
        functionType           // 函数类型
    );
    
    // 创建入口块并获取函数参数
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // 获取参数
    auto args = entryBlock->getArguments();
    Value input = args[0];    // 输入张量
    Value kernel = args[1];   // 卷积核张量
    Value output = args[2];   // 输出张量

    // 创建索引映射
    // 输入索引映射: (n, k, oh, ow, c, kh, kw) -> (n, c, oh + kh, ow + kw)
    // 卷积核索引映射: (n, k, oh, ow, c, kh, kw) -> (c / Ct, k / Kt, kh, kw, c % Ct, k % Kt)
    // 输出索引映射: (n, k, oh, ow, c, kh, kw) -> (n, k, oh, ow)
    
    // 构建输入和输出索引映射
    MLIRContext* ctx = builder.getContext();
    
    // 输入映射
    AffineExpr n = getAffineDimExpr(0, ctx);
    AffineExpr k = getAffineDimExpr(1, ctx);
    AffineExpr oh = getAffineDimExpr(2, ctx);
    AffineExpr ow = getAffineDimExpr(3, ctx);
    AffineExpr c = getAffineDimExpr(4, ctx);
    AffineExpr kh = getAffineDimExpr(5, ctx);
    AffineExpr kw = getAffineDimExpr(6, ctx);
    
    // 构建索引表达式
    // 输入索引表达式
    SmallVector<AffineExpr, 4> inputExprs = {
        n,                    // 批次维度
        c,                    // 通道维度
        oh + kh,              // 高度维度（输出高度 + 卷积核高度偏移）
        ow + kw               // 宽度维度（输出宽度 + 卷积核宽度偏移）
    };

    // 卷积核索引表达式
    SmallVector<AffineExpr, 6> kernelExprs = {
        c.floorDiv(Ct),       // 通道分块 (c / Ct)
        k.floorDiv(Kt),       // 卷积核分块 (k / Kt)
        kh,                   // 卷积核高度
        kw,                   // 卷积核宽度
        c % Ct,               // 通道内偏移
        k % Kt                // 卷积核内偏移
    };

    // 输出索引表达式
    SmallVector<AffineExpr, 4> outputExprs = {
        n,                    // 批次维度
        k,                    // 输出通道维度
        oh,                   // 输出高度
        ow                    // 输出宽度
    };

    // 构建索引映射
    // 创建输入索引映射
    AffineMap inputMap = AffineMap::get(
        7,  // 维度数 (n, k, oh, ow, c, kh, kw)
        0,  // 符号数
        inputExprs,  // 结果表达式 (n, c, oh + kh, ow + kw)
        ctx
    );
    
    // 创建卷积核索引映射
    AffineMap kernelMap = AffineMap::get(
        7,  // 维度数
        0,  // 符号数
        kernelExprs,  // 结果表达式 (c / Ct, k / Kt, kh, kw, c % Ct, k % Kt)
        ctx
    );
    
    // 创建输出索引映射
    AffineMap outputMap = AffineMap::get(
        7,  // 维度数
        0,  // 符号数
        outputExprs,  // 结果表达式 (n, k, oh, ow)
        ctx
    );

    // 创建迭代器类型数组
    SmallVector<utils::IteratorType, 7> iterTypes = {
        utils::IteratorType::parallel,   // n
        utils::IteratorType::parallel,   // k
        utils::IteratorType::parallel,   // oh
        utils::IteratorType::parallel,   // ow
        utils::IteratorType::reduction,  // c
        utils::IteratorType::reduction,  // kh
        utils::IteratorType::reduction   // kw
    };

    // 创建linalg.generic操作
    auto genericOp = builder.create<linalg::GenericOp>(
        LOC,
        TypeRange{outputType},           // 结果类型
        ValueRange{input, kernel},       // 输入值
        ValueRange{output},              // 输出值
        ArrayRef<AffineMap>{inputMap, kernelMap, outputMap},  // 索引映射
        iterTypes,                       // 迭代器类型
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
            // 获取输入、卷积核和输出值
            Value inputValue = args[0];
            Value kernelValue = args[1];
            Value outputValue = args[2];
            
            // 卷积计算: out += input * kernel
            auto mulOp = nestedBuilder.create<arith::MulFOp>(nestedLoc, inputValue, kernelValue);
            auto addOp = nestedBuilder.create<arith::AddFOp>(nestedLoc, outputValue, mulOp);
            
            // 将结果输出
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, ValueRange{addOp});
        }
    );

    // 添加标签以便transform::MatchOp精确定位
    genericOp->setAttr("transform.target_tag", 
                       StringAttr::get(ctx, "direct_conv2d"));

    // 返回卷积结果
    builder.create<func::ReturnOp>(
        LOC,
        genericOp.getResult(0)
    );
    
    return funcOp;
}

LogicalResult createAndApplyTransform2(ModuleOp module) {
    MLIRContext* context = module->getContext();
    OpBuilder builder(context);
    
    // 1. 创建转换模块
    ModuleOp transformModule = ModuleOp::create(LOC);
    builder.setInsertionPointToEnd(transformModule.getBody());

    // 2. 创建序列操作
    auto sequenceOp = builder.create<transform::SequenceOp>(
        LOC,                                     // location
        TypeRange{},                                      // result types
        transform::FailurePropagationMode::Propagate,     // failure mode
        builder.getType<transform::AnyOpType>(),          // block argument type
        [](OpBuilder &b, Location nested, Value rootH) {} // body builder function
    );

    // 3. 插入变换操作
    auto *sequenceBody = sequenceOp.getBodyBlock();
    Value arg0 = sequenceBody->getArgument(0);
    builder.setInsertionPointToEnd(sequenceBody);

    auto globalMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
        builder.getContext(), mtdsp::MTDSPDialect::getGlobalAddressSpace());
    auto workgroupMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
        builder.getContext(), mtdsp::MTDSPDialect::getWorkgroupAddressSpace());
    auto scalarMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
        builder.getContext(), mtdsp::MTDSPDialect::getScalarAddressSpace());
    auto vectorMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
        builder.getContext(), mtdsp::MTDSPDialect::getVectorAddressSpace());

    // 匹配名为"linalg.generic"且带有特定transform.target_tag的操作
    Value convOpHandle = builder.create<transform::MatchOp>(
        LOC,
        transform::AnyOpType::get(context),       // 任何操作类型
        arg0,                                     // 目标操作（在其中查找）
        builder.getArrayAttr({builder.getStringAttr("linalg.generic")}),  // 操作名称
        nullptr,                                  // 接口类型（未使用）
        // 使用DictionaryAttr指定要匹配的属性
        builder.getDictionaryAttr({
            builder.getNamedAttr("transform.target_tag", 
                                builder.getStringAttr("direct_conv2d"))
        }),
        nullptr,                       // 结果类型过滤器（未使用）
        nullptr                        // 操作数类型过滤器（未使用）
    ).getResult();

    builder.create<transform::PrintOp>(
        LOC,
        convOpHandle,
        /*name=*/builder.getStringAttr("Matched operations")  // 可选
    );

    // 对C维度进行分块
    SmallVector<int64_t, 6> tileSizes = {0, 0, 0, 0, 64};
    auto tileUsingForOp = builder.create<transform::TileUsingForOp>(
        LOC, 
        convOpHandle,  // target
        tileSizes      // static tile sizes
    );
    Value tiledConvHandle = tileUsingForOp.getTiledLinalgOp();
    ValueRange loopHandles = tileUsingForOp.getLoops();

    // 对K维度进行分块
    tileSizes = {0, 64};
    auto tileUsingForOp2 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledConvHandle,  // target
        tileSizes      // static tile sizes
    );
    Value tiledConvHandle2 = tileUsingForOp2.getTiledLinalgOp();
    ValueRange loopHandles2 = tileUsingForOp2.getLoops();

    auto convFilterHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledConvHandle2, 1);

    auto copyFilterHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        convFilterHandle,
        vectorMemoryAddressSpace,
        true);
        // false);

    // 对OW维度进行分块
    tileSizes = {0, 0, 0, 14};
    auto tileUsingForOp3 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledConvHandle2,  // target
        tileSizes      // static tile sizes
    );
    Value tiledConvHandle3 = tileUsingForOp3.getTiledLinalgOp();
    ValueRange loopHandles3 = tileUsingForOp3.getLoops();

    // 对OH维度进行分块
    tileSizes = {0, 0, 1};
    auto tileUsingForOp4 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledConvHandle3,  // target
        tileSizes      // static tile sizes
    );
    Value tiledConvHandle4 = tileUsingForOp4.getTiledLinalgOp();
    ValueRange loopHandles4 = tileUsingForOp4.getLoops();

    auto convInputHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledConvHandle4, 0);

    auto copyInputHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        convInputHandle,
        scalarMemoryAddressSpace,
        // true);
        false);

    auto convOutputHandle = builder.create<transform::GetResultOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledConvHandle4, 0);

    auto writeOutputHandle = builder.create<transform::CacheWriteOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        convOutputHandle,
        vectorMemoryAddressSpace,
        // true);
        false,
        nullptr);

    builder.create<transform::YieldOp>(LOC);

    llvm::outs() << transformModule << "\n";

    // 4. 应用转换
    transform::TransformOptions options;
    if (failed(transform::applyTransforms(
        module,      // payload root
        sequenceOp,    // transform operation
        {},         // extra mapping
        options                  // options
    ))) {
        llvm::errs() << "Transform application failed\n";
        return failure();
    }

    return success();
}

// LogicalResult createAndApplyTransform2(ModuleOp module) {
//     MLIRContext* context = module->getContext();
//     OpBuilder builder(context);
    
//     // 1. 创建转换模块
//     ModuleOp transformModule = ModuleOp::create(LOC);
//     builder.setInsertionPointToEnd(transformModule.getBody());

//     // 2. 创建序列操作
//     auto sequenceOp = builder.create<transform::SequenceOp>(
//         LOC,                                     // location
//         TypeRange{},                                      // result types
//         transform::FailurePropagationMode::Propagate,     // failure mode
//         builder.getType<transform::AnyOpType>(),          // block argument type
//         [](OpBuilder &b, Location nested, Value rootH) {} // body builder function
//     );

//     // 3. 插入变换操作
//     auto *sequenceBody = sequenceOp.getBodyBlock();
//     Value arg0 = sequenceBody->getArgument(0);
//     builder.setInsertionPointToEnd(sequenceBody);

//     auto globalMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
//         builder.getContext(), mtdsp::MTDSPDialect::getGlobalAddressSpace());
//     auto workgroupMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
//         builder.getContext(), mtdsp::MTDSPDialect::getWorkgroupAddressSpace());
//     auto scalarMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
//         builder.getContext(), mtdsp::MTDSPDialect::getScalarAddressSpace());
//     auto vectorMemoryAddressSpace = mtdsp::AddressSpaceAttr::get(
//         builder.getContext(), mtdsp::MTDSPDialect::getVectorAddressSpace());
    
//     SmallVector<StringRef, 1> opNames = {"linalg.conv_2d_nchw_fchw"};
//     auto convOpHandle = builder.create<transform::MatchOp>(
//         LOC,
//         arg0,              // target
//         opNames            // operation names to match
//     );

//     // 替代方案：一次性对 N 和 F 维度进行分块
//     SmallVector<int64_t, 6> tileSizes = {1, 1, 0, 0};  // [N=1, F=1, OH=0, OW=0]
//     auto tileUsingForOp = builder.create<transform::TileUsingForOp>(
//         LOC, 
//         convOpHandle,  // target
//         tileSizes      // static tile sizes
//     );
//     Value tiledConvHandle = tileUsingForOp.getTiledLinalgOp();
//     ValueRange loopHandles = tileUsingForOp.getLoops();

//     // 匹配所有函数操作
//     auto funcOp = builder.create<transform::MatchOp>(
//         LOC,
//         arg0,
//         SmallVector<StringRef, 1>{"func.func"}
//     );

//     // 应用 canonicalization 模式
//     builder.create<transform::ApplyPatternsOp>(
//         LOC,
//         funcOp.getResult(),  // target
//         [&](OpBuilder &b, Location loc) { // 即便是空，也会触发模式重写，其中包含了死代码消除的功能
//             // 在 patterns 区块中添加 canonicalization
//             b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
//         }
//     );

//     // 根据定义，LoopCoalesceOp 只需要一个参数：最外层循环句柄
//     // 它会自动合并该循环下的完美嵌套循环
//     auto loopCoalesceOp = builder.create<transform::LoopCoalesceOp>(
//         LOC,
//         builder.getType<transform::AnyOpType>(),  // 返回类型 - 需要指定一个 TransformHandleTypeInterface 类型
//         loopHandles[0]                         // 只需要传递最外层循环
//     );
    
//     // 获取合并后的循环句柄
//     Value coalescedLoopHandle = loopCoalesceOp.getResult();

//     // // 第三轮分块：将输出高度(OH)完全分块
//     // tileSizes = {0, 0, 1, 0};  // [N=0, F=0, OH=1, OW=0]
//     // auto tileUsingForOp3 = builder.create<transform::TileUsingForOp>(
//     //     LOC, 
//     //     tiledConvHandle,  // target
//     //     tileSizes          // static tile sizes
//     // );
//     // Value tiledConvHandle3 = tileUsingForOp3.getTiledLinalgOp();  // 分块后的操作
//     // ValueRange loopHandles3 = tileUsingForOp3.getLoops();         // 生成的循环

//     // // 第四轮分块：将输出宽度(OW)完全分块
//     // tileSizes = {0, 0, 0, 1};  // [N=0, F=0, OH=0, OW=1]
//     // auto tileUsingForOp4 = builder.create<transform::TileUsingForOp>(
//     //     LOC, 
//     //     tiledConvHandle3,  // target
//     //     tileSizes          // static tile sizes
//     // );
//     // Value tiledConvHandle4 = tileUsingForOp4.getTiledLinalgOp();  // 分块后的操作
//     // ValueRange loopHandles4 = tileUsingForOp4.getLoops();         // 生成的循环



//     // // 应用 CSE
//     // builder.create<transform::ApplyCommonSubexpressionEliminationOp>(
//     //     LOC,
//     //     funcOp.getResult()
//     // );

//     builder.create<transform::YieldOp>(LOC);

//     llvm::outs() << transformModule << "\n";

//     // 4. 应用转换
//     transform::TransformOptions options;
//     if (failed(transform::applyTransforms(
//         module,      // payload root
//         sequenceOp,    // transform operation
//         {},         // extra mapping
//         options                  // options
//     ))) {
//         llvm::errs() << "Transform application failed\n";
//         return failure();
//     }

//     return success();
// }