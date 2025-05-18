#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/LayoutTrans/IR/LayoutTransDialect.h"
#include "Conversion/ConvertToMTDSP/ConvertToMTDSPPass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MemrefToMTDSP RewritePatterns: AllocOp
//===----------------------------------------------------------------------===//

class ConvertMemrefAllocToMTDSP : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    auto allocOp = rewriter.create<mtdsp::AllocOp>(
      loc,
      op.getResult().getType(),
      ValueRange{}
    );

    rewriter.replaceOp(op, allocOp);

    auto memRefType = op.getResult().getType().cast<MemRefType>();
    if(mlir::Attribute attr = memRefType.getMemorySpace()){
      if(mtdsp::AddressSpaceAttr addrSpaceAttr = cast<mtdsp::AddressSpaceAttr>(attr)){
        // 在函数末尾插入mtdsp::DeallocOp
        // 获取当前函数
        auto parentFunc = allocOp->getParentOfType<func::FuncOp>();
        
        // 在函数末尾的返回操作之前插入 DeallocOp
        auto returnOp = parentFunc.getBlocks().back().getTerminator();
        rewriter.setInsertionPoint(returnOp);
        
        // 创建 DeallocOp
        rewriter.create<mtdsp::DeallocOp>(
          loc,
          allocOp.getResult()
        );
      }
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// MemrefToMTDSP RewritePatterns: CopyOp
//===----------------------------------------------------------------------===//

class ConvertMemrefCopyToMTDSP : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType().cast<MemRefType>();
    auto targetType = op.getTarget().getType().cast<MemRefType>();

    // 确保两个memref的形状匹配
    if (sourceType.getShape() != targetType.getShape()) {
      return failure();
    }

    // 如果src和tgt都存在偏移，返回失败
    bool sourceHasOffset = false;
    bool targetHasOffset = false;

    // 检查源memref是否有偏移
    if (auto sourceLayout = sourceType.getLayout().dyn_cast<StridedLayoutAttr>()) {
      ArrayRef<int64_t> sourceShape = sourceType.getShape();
      ArrayRef<int64_t> sourceStrides = sourceLayout.getStrides();
      
      // 检查每个维度的偏移情况
      for (unsigned i = 0; i < sourceType.getRank() - 1; i++) {
        // 如果该维度的大小乘以下一维度的步长小于当前维度的步长，则存在偏移
        if (sourceShape[i + 1] * sourceStrides[i + 1] < sourceStrides[i]) {
          sourceHasOffset = true;
          break;
        }
      }
    }

    // 检查目标memref是否有偏移
    if (auto targetLayout = targetType.getLayout().dyn_cast<StridedLayoutAttr>()) {
      ArrayRef<int64_t> targetShape = targetType.getShape();
      ArrayRef<int64_t> targetStrides = targetLayout.getStrides();
      
      // 检查每个维度的偏移情况
      for (unsigned i = 0; i < targetType.getRank() - 1; i++) {
        // 如果该维度的大小乘以下一维度的步长小于当前维度的步长，则存在偏移
        if (targetShape[i + 1] * targetStrides[i + 1] < targetStrides[i]) {
          targetHasOffset = true;
          break;
        }
      }
    }

    // 如果源和目标都存在偏移，返回失败
    if (sourceHasOffset && targetHasOffset) {
      return failure();
    }
    
    // 将最内两层使用DMA传输，外层使用循环
    // 确定最内层两个维度的位置
    unsigned innerDimStart = sourceType.getRank() - 2;

    // 构建循环变量
    SmallVector<Value, 4> lowerBounds;
    SmallVector<Value, 4> upperBounds;
    SmallVector<Value, 4> steps;

    // 为外层维度创建循环界限
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    for (unsigned i = 0; i < innerDimStart; i++) {
      lowerBounds.push_back(zero);
      upperBounds.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, sourceType.getShape()[i]));
      steps.push_back(one);
    }

    // 没有外层维度时，直接使用DMA
    if (lowerBounds.empty()) {
      auto dmaOp = rewriter.create<mtdsp::DMAOp>(
          loc, op.getSource(), op.getTarget());
      rewriter.create<mtdsp::WaitOp>(loc, dmaOp->getResult(0));
      rewriter.eraseOp(op);
      return success();
    }

    // 使用scf.for创建嵌套循环
    scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // 创建子视图的偏移
        SmallVector<OpFoldResult, 4> offsets;
        for (unsigned i = 0; i < sourceType.getRank(); i++) {
          if (i < innerDimStart) {
            offsets.push_back(ivs[i]);
          } else {
            offsets.push_back(nestedBuilder.getIndexAttr(0));
          }
        }

        // 创建子视图的大小
        SmallVector<OpFoldResult, 4> sizes;
        for (unsigned i = 0; i < sourceType.getRank(); i++) {
          if (i < innerDimStart) {
            sizes.push_back(nestedBuilder.getIndexAttr(1));
          } else {
            sizes.push_back(nestedBuilder.getIndexAttr(sourceType.getShape()[i]));
          }
        }

        // 创建步长（都使用单位步长）
        SmallVector<OpFoldResult, 4> strides;
        for (unsigned i = 0; i < sourceType.getRank(); i++) {
          strides.push_back(nestedBuilder.getIndexAttr(1));
        }

        // 推导出压缩后的结果类型
        MemRefType sourceSubviewType = memref::SubViewOp::inferRankReducedResultType(
            {sourceType.getDimSize(sourceType.getRank()-2), sourceType.getDimSize(sourceType.getRank()-1)}, 
            sourceType, 
            offsets, sizes, strides).cast<MemRefType>();
        
        MemRefType targetSubviewType = memref::SubViewOp::inferRankReducedResultType(
            {targetType.getDimSize(sourceType.getRank()-2), targetType.getDimSize(sourceType.getRank()-1)}, 
            targetType, 
            offsets, sizes, strides).cast<MemRefType>();

        // 创建源和目标子视图，并指定结果类型
        auto sourceSubview = nestedBuilder.create<memref::SubViewOp>(
            loc, sourceSubviewType, op.getSource(), offsets, sizes, strides);

        auto targetSubview = nestedBuilder.create<memref::SubViewOp>(
            loc, targetSubviewType, op.getTarget(), offsets, sizes, strides);
        
        // 对子视图使用DMA操作
        auto dmaOp = nestedBuilder.create<mtdsp::DMAOp>(
            loc, sourceSubview, targetSubview);
        
        // 等待DMA完成
        nestedBuilder.create<mtdsp::WaitOp>(loc, dmaOp->getResult(0));
      });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MemrefToLoadStore RewritePatterns: CopyOp
//===----------------------------------------------------------------------===//

class ConvertMemrefCopyToLoadStore : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType().cast<MemRefType>();
    auto targetType = op.getTarget().getType().cast<MemRefType>();

    // 确保两个memref的形状匹配
    if (sourceType.getShape() != targetType.getShape()) {
      return failure();
    }
    
    // 构建循环变量
    SmallVector<Value, 4> lowerBounds;
    SmallVector<Value, 4> upperBounds;
    SmallVector<Value, 4> steps;

    // 为所有维度创建循环界限
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    for (unsigned i = 0; i < sourceType.getRank(); i++) {
      lowerBounds.push_back(zero);
      upperBounds.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, sourceType.getShape()[i]));
      steps.push_back(one);
    }

    // 使用scf.for创建嵌套循环
    scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // 使用索引直接加载和存储元素
        Value loadedValue = nestedBuilder.create<memref::LoadOp>(
            loc, op.getSource(), ivs);
            
        nestedBuilder.create<memref::StoreOp>(
            loc, loadedValue, op.getTarget(), ivs);
      });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToMTDSP RewritePatterns: CopyOp
//===----------------------------------------------------------------------===//

class ConvertLinalgCopyToMTDSP : public OpConversionPattern<linalg::CopyOp> {
  using OpConversionPattern<linalg::CopyOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(linalg::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    auto dmaOp = rewriter.create<mtdsp::DMAOp>(
        loc,
        op.getInputs()[0],
        op.getOutputs()[0]);
    rewriter.create<mtdsp::WaitOp>(
        loc,
        dmaOp->getResult(0));
    
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToMTDSP RewritePatterns: MatmulOp
//===----------------------------------------------------------------------===//

class ConvertLinalgMatmulToMTDSP : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // 获取输入矩阵
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // // 获取矩阵维度
    // MemRefType lhsType = cast<MemRefType>(lhs.getType());
    // MemRefType outputType = cast<MemRefType>(output.getType());

    // // 获取原始subview操作
    // auto lhsOp = lhs.getDefiningOp<memref::SubViewOp>();
    // auto outputOp = output.getDefiningOp<memref::SubViewOp>();
    
    // // 创建前6行的subview，偏移量不变，形状取6行，步长不变
    // SmallVector<OpFoldResult> lhsOffsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    // SmallVector<OpFoldResult> lhsSizes;
    // SmallVector<OpFoldResult> lhsStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    
    // lhsSizes.push_back(rewriter.getIndexAttr(6));
    // if (ShapedType::isDynamic(lhsType.getShape()[1]))
    //   lhsSizes.push_back(rewriter.create<memref::DimOp>(loc, lhs, 1).getResult());
    // else
    //   lhsSizes.push_back(rewriter.getIndexAttr(lhsType.getShape()[1]));
    
    // Value firstHalfLhs = rewriter.create<memref::SubViewOp>(
    //     loc, lhs,
    //     lhsOffsets,  // offsets
    //     lhsSizes,    // sizes 
    //     lhsStrides   // strides
    // );
    
    // SmallVector<OpFoldResult> outputOffsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    // SmallVector<OpFoldResult> outputSizes;
    // SmallVector<OpFoldResult> outputStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    
    // outputSizes.push_back(rewriter.getIndexAttr(6));
    // if (ShapedType::isDynamic(outputType.getShape()[1]))
    //   outputSizes.push_back(rewriter.create<memref::DimOp>(loc, output, 1).getResult());
    // else
    //   outputSizes.push_back(rewriter.getIndexAttr(outputType.getShape()[1]));

    // Value firstHalfOutput = rewriter.create<memref::SubViewOp>(
    //     loc, output,
    //     outputOffsets,
    //     outputSizes,
    //     outputStrides
    // );
    
    // // 创建后6行的subview，偏移量6行，形状取6行，步长不变
    // SmallVector<OpFoldResult> lhsOffsetsSecond = {rewriter.getIndexAttr(6), rewriter.getIndexAttr(0)};
    
    // Value secondHalfLhs = rewriter.create<memref::SubViewOp>(
    //     loc, lhs,
    //     lhsOffsetsSecond,
    //     lhsSizes,
    //     lhsStrides
    // );
    
    // SmallVector<OpFoldResult> outputOffsetsSecond = {rewriter.getIndexAttr(6), rewriter.getIndexAttr(0)};
    
    // Value secondHalfOutput = rewriter.create<memref::SubViewOp>(
    //     loc, output, 
    //     outputOffsetsSecond,
    //     outputSizes,
    //     outputStrides
    // );

    // // // 创建前6行的MatmulR6C96Op
    // // rewriter.create<mtdsp::MatmulR6C96Op>(
    // //     loc, firstHalfLhs, rhs, firstHalfOutput
    // // );
    
    // // // 创建后6行的MatmulR6C96Op
    // // rewriter.create<mtdsp::MatmulR6C96Op>(
    // //     loc, secondHalfLhs, rhs, secondHalfOutput
    // // );

    // // 创建前6行的MatmulR6C128Op
    // rewriter.create<mtdsp::MatmulR6C128Op>(
    //     loc, firstHalfLhs, rhs, firstHalfOutput
    // );
    
    // // 创建后6行的MatmulR6C128Op
    // rewriter.create<mtdsp::MatmulR6C128Op>(
    //     loc, secondHalfLhs, rhs, secondHalfOutput
    // );

    // 创建MatmulR12C128Op
    rewriter.create<mtdsp::MatmulR12C128Op>(
        loc, lhs, rhs, output
    );
    
    // 删除原始op
    rewriter.eraseOp(op);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToMTDSP RewritePatterns: Generic Direct Conv2D
//===----------------------------------------------------------------------===//

class ConvertLinalgGenericConv2DToMTDSP : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 检查是否具有direct_conv2d标签
    auto targetAttr = op->getAttrOfType<StringAttr>("transform.target_tag");
    if (!targetAttr || targetAttr.getValue() != "direct_conv2d")
      return failure();

    auto loc = op.getLoc();
    
    // 检查操作数数量
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();
    
    // 获取输入和输出
    Value input = adaptor.getInputs()[0];    // 输入特征图
    Value kernel = adaptor.getInputs()[1];   // 卷积核
    Value output = adaptor.getOutputs()[0];  // 输出特征图
    
    // 获取类型
    auto inputType = cast<MemRefType>(input.getType());
    auto kernelType = cast<MemRefType>(kernel.getType());
    auto outputType = cast<MemRefType>(output.getType());
    
    // 验证输入类型是否匹配预期格式
    if (inputType.getRank() != 4)
      return failure();
    
    // 验证卷积核类型是否匹配预期格式
    if (kernelType.getRank() != 6)
      return failure();
    
    // 验证输出类型是否匹配预期格式
    if (outputType.getRank() != 4)
      return failure();
    
    // 检查内存空间属性
    auto inputMemSpace = inputType.getMemorySpace();
    auto kernelMemSpace = kernelType.getMemorySpace();
    auto outputMemSpace = outputType.getMemorySpace();
    
    // 验证内存空间属性是否存在
    if (!inputMemSpace || !kernelMemSpace || !outputMemSpace)
      return failure();
    
    // 确保输入在SM空间，卷积核和输出在AM空间
    auto inputAddrSpace = dyn_cast<mtdsp::AddressSpaceAttr>(inputMemSpace);
    auto kernelAddrSpace = dyn_cast<mtdsp::AddressSpaceAttr>(kernelMemSpace);
    auto outputAddrSpace = dyn_cast<mtdsp::AddressSpaceAttr>(outputMemSpace);
    
    if (!inputAddrSpace || !kernelAddrSpace || !outputAddrSpace)
      return failure();
    
    if (inputAddrSpace.getValue() != mtdsp::AddressSpace::Scalar ||
        kernelAddrSpace.getValue() != mtdsp::AddressSpace::Vector ||
        outputAddrSpace.getValue() != mtdsp::AddressSpace::Vector)
      return failure();
    
    // 创建专门的3x3卷积操作
    rewriter.create<mtdsp::Conv2d3x3S1N64M14Op>(
        loc,
        input,      // 输入特征图
        kernel,     // 卷积核
        output      // 输出特征图
    );
    
    // 删除原始操作
    rewriter.eraseOp(op);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LayoutTransToMTDSP RewritePatterns: Img2ColOp
//===----------------------------------------------------------------------===//

class ConvertImg2ColToMTDSP : public OpConversionPattern<layout_trans::Img2ColOp> {
  using OpConversionPattern<layout_trans::Img2ColOp>::OpConversionPattern;
public:
  LogicalResult 
  matchAndRewrite(layout_trans::Img2ColOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // 获取输入和输出值
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    
    // 获取卷积核大小
    int32_t kernelHeight = op.getKernelHeight();
    int32_t kernelWidth = op.getKernelWidth();
    
    // 获取输入和输出的类型
    auto inputType = cast<MemRefType>(input.getType());
    auto outputType = cast<MemRefType>(output.getType());
    
    // 获取输入维度
    auto inputShape = inputType.getShape();
    int64_t batchSize = inputShape[0]; // N
    int64_t channels = inputShape[1];  // C
    int64_t height = inputShape[2];    // H
    int64_t width = inputShape[3];     // W
    
    // 计算输出特征图的尺寸
    int64_t outputHeight = height - kernelHeight + 1; // P
    int64_t outputWidth = width - kernelWidth + 1;    // Q
    int64_t outputSize = outputHeight * outputWidth;  // P*Q
    
    // 创建循环嵌套
    // 外层循环：遍历 n, c, r, s（批次、通道、卷积核行、卷积核列）
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    // n循环
    Value nUpper = rewriter.create<arith::ConstantIndexOp>(loc, batchSize);
    auto nLoop = rewriter.create<scf::ForOp>(
        loc, zero, nUpper, one,
        /* 初始化迭代器变量 */ (ValueRange{}));
    
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value n = nLoop.getInductionVar();
    
    // c循环
    Value cUpper = rewriter.create<arith::ConstantIndexOp>(loc, channels);
    auto cLoop = rewriter.create<scf::ForOp>(
        loc, zero, cUpper, one,
        /* 初始化迭代器变量 */ (ValueRange{}));
    
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value c = cLoop.getInductionVar();
    
    // r循环（卷积核行）
    Value rUpper = rewriter.create<arith::ConstantIndexOp>(loc, kernelHeight);
    auto rLoop = rewriter.create<scf::ForOp>(
        loc, zero, rUpper, one,
        /* 初始化迭代器变量 */ (ValueRange{}));
    
    rewriter.setInsertionPointToStart(rLoop.getBody());
    Value r = rLoop.getInductionVar();
    
    // s循环（卷积核列）
    Value sUpper = rewriter.create<arith::ConstantIndexOp>(loc, kernelWidth);
    auto sLoop = rewriter.create<scf::ForOp>(
        loc, zero, sUpper, one,
        /* 初始化迭代器变量 */ (ValueRange{}));
    
    rewriter.setInsertionPointToStart(sLoop.getBody());
    Value s = sLoop.getInductionVar();
    
    // 在这个位置，我们在最内层循环内
    
    // 1. 计算输入子视图的起始位置和大小
    SmallVector<Value, 4> inputOffsets = {n, c, r, s};
    SmallVector<Value, 4> inputSizes = {
      one,
      one,
      rewriter.create<arith::ConstantIndexOp>(loc, outputHeight),
      rewriter.create<arith::ConstantIndexOp>(loc, outputWidth)
    };
    SmallVector<Value, 4> inputStrides(4, one);
    
    // 创建输入子视图
    Value inputSubview = rewriter.create<memref::SubViewOp>(
        loc, input, inputOffsets, inputSizes, inputStrides);
    
    // 2. 计算输出子视图的位置和大小
    // 计算CRS索引：对应于输出矩阵的第一维
    // 首先计算r*s项
    Value rs = rewriter.create<arith::MulIOp>(loc, r, s);
    // 然后计算c*r*s项
    Value crs = rewriter.create<arith::MulIOp>(loc, c, rs);
    
    // 输出偏移量：(c*r*s, n)
    SmallVector<Value, 2> outputOffsets = {crs, n};
    SmallVector<Value, 2> outputSizes = {
      one,
      rewriter.create<arith::ConstantIndexOp>(loc, outputSize)
    };
    SmallVector<Value, 2> outputStrides(2, one);
    
    // 创建输出子视图
    Value outputSubview = rewriter.create<memref::SubViewOp>(
        loc, output, outputOffsets, outputSizes, outputStrides);
    
    // 3. 创建DMA操作，将数据从输入复制到输出
    auto dmaOp = rewriter.create<mtdsp::DMAOp>(loc, inputSubview, outputSubview);

    rewriter.create<mtdsp::WaitOp>(
        loc,
        dmaOp->getResult(0));
    
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LayoutTransToMTDSP RewritePatterns: Col2ImgOp
//===----------------------------------------------------------------------===//

namespace {
struct ConvertToMTDSPPass : public PassWrapper<ConvertToMTDSPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToMTDSPPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final;
};
}

void ConvertToMTDSPPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         linalg::LinalgDialect, affine::AffineDialect,
                         arith::ArithDialect, memref::MemRefDialect, 
                         vector::VectorDialect, scf::SCFDialect,
                         LLVM::LLVMDialect, mtdsp::MTDSPDialect>();
  target.addIllegalOp<memref::AllocOp>();
  // target.addIllegalOp<memref::CopyOp>();
  target.addIllegalOp<linalg::CopyOp>();
  target.addIllegalOp<linalg::MatmulOp>();
  target.addIllegalOp<layout_trans::Img2ColOp>();
  target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
    auto targetAttr = op->getAttrOfType<StringAttr>("transform.target_tag");
    return !(targetAttr && targetAttr.getValue() == "direct_conv2d");
  });

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertMemrefAllocToMTDSP,
      // ConvertMemrefCopyToMTDSP,
      ConvertMemrefCopyToLoadStore,
      ConvertLinalgCopyToMTDSP,
      ConvertLinalgMatmulToMTDSP,
      ConvertImg2ColToMTDSP,
      ConvertLinalgGenericConv2DToMTDSP
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createConvertToMTDSPPass() {
  return std::make_unique<ConvertToMTDSPPass>();
};