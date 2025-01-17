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

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertMemrefAllocToMTDSP,
      ConvertLinalgCopyToMTDSP
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createConvertToMTDSPPass() {
  return std::make_unique<ConvertToMTDSPPass>();
};