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
#include "Conversion/LinalgToMTDSP/LinalgToMTDSPPass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOMTDSP
#include "Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

class CopyOpLowering : public OpConversionPattern<linalg::CopyOp> {
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
// MatmulOp
//===----------------------------------------------------------------------===//

class MatmulOpLowering : public OpConversionPattern<linalg::MatmulOp> {
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
    
    // 创建MatmulMicroKernelOp
    rewriter.create<mtdsp::MatmulMicroKernelOp>(
        loc, lhs, rhs, output
    );
    
    // 删除原始op
    rewriter.eraseOp(op);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

class AddOpLowering : public OpConversionPattern<linalg::AddOp> {
  using OpConversionPattern<linalg::AddOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(linalg::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 获取输入矩阵
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    // 创建AddMicroKernelOp
    rewriter.create<mtdsp::AddMicroKernelOp>(
        loc, lhs, rhs, output
    );

    // 删除原始op
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GenericOp with op_label (extensible for multiple activation functions)
//===----------------------------------------------------------------------===//

class GenericOpLowering : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  // 定义处理函数类型
  using HandlerFn = std::function<LogicalResult(
      linalg::GenericOp, OpAdaptor, ConversionPatternRewriter&, Location)>;

public:
  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 检查是否有 op_label 属性
    auto opLabelAttr = op->getAttrOfType<StringAttr>("op_label");
    if (!opLabelAttr) {
      return failure();
    }

    // 根据 op_label 分发到对应的处理函数
    StringRef opLabel = opLabelAttr.getValue();
    auto it = handlers.find(opLabel);
    if (it == handlers.end()) {
      return failure();
    }

    // 调用对应的处理函数
    return it->second(op, adaptor, rewriter, loc);
  }

private:
  // 处理 relu 的函数
  static LogicalResult handleReLU(linalg::GenericOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter, Location loc) {
    // 验证输入输出数量
    if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1) {
      return failure();
    }

    Value input = adaptor.getInputs()[0];
    Value output = adaptor.getOutputs()[0];

    // 创建 ReLUMicroKernelOp
    rewriter.create<mtdsp::ReLUMicroKernelOp>(loc, input, output);
    rewriter.eraseOp(op);

    return success();
  }

  // op_label 到处理函数的映射表
  inline static const llvm::StringMap<HandlerFn> handlers = {
    {"relu", handleReLU},
  };
};

namespace {
struct ConvertLinalgToMTDSPPass : public impl::ConvertLinalgToMTDSPBase<ConvertLinalgToMTDSPPass> {
  void runOnOperation() final;
};
}

void ConvertLinalgToMTDSPPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         affine::AffineDialect,
                         arith::ArithDialect, memref::MemRefDialect, 
                         vector::VectorDialect, scf::SCFDialect,
                         LLVM::LLVMDialect, mtdsp::MTDSPDialect>();
  target.addIllegalDialect<linalg::LinalgDialect>();

  RewritePatternSet patterns(context);
  patterns.add<
      CopyOpLowering,
      MatmulOpLowering,
      AddOpLowering,
      GenericOpLowering
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mtir::createConvertLinalgToMTDSPPass() {
  return std::make_unique<ConvertLinalgToMTDSPPass>();
};