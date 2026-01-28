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
#include "Conversion/MemRefToMTDSP/MemRefToMTDSPPass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOMTDSP
#include "Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

class AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
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
// DeallocOp
//===----------------------------------------------------------------------===//

class DeallocOpLowering : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 创建 mtdsp::DeallocOp
    rewriter.create<mtdsp::DeallocOp>(loc, adaptor.getMemref());

    // 删除原始的 memref::DeallocOp
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertMemRefToMTDSPPass : public impl::ConvertMemRefToMTDSPBase<ConvertMemRefToMTDSPPass> {
  void runOnOperation() final;
};
}

void ConvertMemRefToMTDSPPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         linalg::LinalgDialect, affine::AffineDialect,
                         arith::ArithDialect,
                         vector::VectorDialect, scf::SCFDialect,
                         LLVM::LLVMDialect, mtdsp::MTDSPDialect>();
  target.addIllegalOp<memref::AllocaOp, memref::DeallocOp>();

  RewritePatternSet patterns(context);
  patterns.add<
      AllocOpLowering,
      DeallocOpLowering
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mtir::createConvertMemRefToMTDSPPass() {
  return std::make_unique<ConvertMemRefToMTDSPPass>();
};