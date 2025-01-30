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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
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
    
    // 获取矩阵维度
    MemRefType lhsType = cast<MemRefType>(lhs.getType());
    MemRefType outputType = cast<MemRefType>(output.getType());

    // 获取原始subview操作
    auto lhsOp = lhs.getDefiningOp<memref::SubViewOp>();
    auto outputOp = output.getDefiningOp<memref::SubViewOp>();
    
    // 创建前6行的subview，偏移量不变，形状取6行，步长不变
    SmallVector<OpFoldResult> lhsOffsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> lhsSizes;
    SmallVector<OpFoldResult> lhsStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    
    lhsSizes.push_back(rewriter.getIndexAttr(6));
    if (ShapedType::isDynamic(lhsType.getShape()[1]))
      lhsSizes.push_back(rewriter.create<memref::DimOp>(loc, lhs, 1).getResult());
    else
      lhsSizes.push_back(rewriter.getIndexAttr(lhsType.getShape()[1]));
    
    Value firstHalfLhs = rewriter.create<memref::SubViewOp>(
        loc, lhs,
        lhsOffsets,  // offsets
        lhsSizes,    // sizes 
        lhsStrides   // strides
    );
    
    SmallVector<OpFoldResult> outputOffsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> outputSizes;
    SmallVector<OpFoldResult> outputStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    
    outputSizes.push_back(rewriter.getIndexAttr(6));
    if (ShapedType::isDynamic(outputType.getShape()[1]))
      outputSizes.push_back(rewriter.create<memref::DimOp>(loc, output, 1).getResult());
    else
      outputSizes.push_back(rewriter.getIndexAttr(outputType.getShape()[1]));

    Value firstHalfOutput = rewriter.create<memref::SubViewOp>(
        loc, output,
        outputOffsets,
        outputSizes,
        outputStrides
    );
    
    // 创建后6行的subview，偏移量6行，形状取6行，步长不变
    SmallVector<OpFoldResult> lhsOffsetsSecond = {rewriter.getIndexAttr(6), rewriter.getIndexAttr(0)};
    
    Value secondHalfLhs = rewriter.create<memref::SubViewOp>(
        loc, lhs,
        lhsOffsetsSecond,
        lhsSizes,
        lhsStrides
    );
    
    SmallVector<OpFoldResult> outputOffsetsSecond = {rewriter.getIndexAttr(6), rewriter.getIndexAttr(0)};
    
    Value secondHalfOutput = rewriter.create<memref::SubViewOp>(
        loc, output, 
        outputOffsetsSecond,
        outputSizes,
        outputStrides
    );

    // 创建前6行的MatmulR6C96Op
    rewriter.create<mtdsp::MatmulR6C96Op>(
        loc, firstHalfLhs, rhs, firstHalfOutput
    );
    
    // 创建后6行的MatmulR6C96Op
    rewriter.create<mtdsp::MatmulR6C96Op>(
        loc, secondHalfLhs, rhs, secondHalfOutput
    );
    
    // 删除原始op
    rewriter.eraseOp(op);
    
    return success();
  }
};

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
  target.addIllegalOp<linalg::CopyOp>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertMemrefAllocToMTDSP,
      ConvertLinalgCopyToMTDSP,
      ConvertLinalgMatmulToMTDSP
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createConvertToMTDSPPass() {
  return std::make_unique<ConvertToMTDSPPass>();
};