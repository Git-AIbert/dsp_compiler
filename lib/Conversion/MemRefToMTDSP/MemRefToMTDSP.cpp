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
  target.addIllegalOp<memref::AllocaOp>();

  RewritePatternSet patterns(context);
  patterns.add<
      AllocOpLowering
    >(context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mtir::createConvertMemRefToMTDSPPass() {
  return std::make_unique<ConvertMemRefToMTDSPPass>();
};