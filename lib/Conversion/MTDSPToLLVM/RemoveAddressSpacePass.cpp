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
#include "Conversion/MTDSPToLLVM/MTDSPToLLVMPass.h"

using namespace mlir;

namespace {
class RemoveAddressSpacePass : public PassWrapper<RemoveAddressSpacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveAddressSpacePass)

  void runOnOperation() final;
};
}

void RemoveAddressSpacePass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(&getContext());

  // 提取处理MemRefType的公共函数
  auto removeAddressSpace = [](MemRefType memrefType) -> MemRefType {
    if (!memrefType)
      return nullptr;
      
    auto memorySpace = memrefType.getMemorySpace();
    if (!memorySpace || !memorySpace.isa<mtdsp::AddressSpaceAttr>())
      return nullptr;

    return MemRefType::get(
        memrefType.getShape(),
        memrefType.getElementType(),
        memrefType.getLayout());
  };

  module.walk([&](Operation *op) {
    // 获取操作的结果类型
    MemRefType memrefType;
    Value oldResult;
    
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (castOp->getNumResults() != 1)
        return;
      memrefType = castOp->getResult(0).getType().dyn_cast<MemRefType>();
      oldResult = castOp->getResult(0);
    } 
    else if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      memrefType = subviewOp.getType();
      oldResult = subviewOp->getResult(0);
    }
    else {
      return;
    }

    // 使用公共函数处理MemRefType
    auto newMemRefType = removeAddressSpace(memrefType);
    if (!newMemRefType)
      return;

    // 创建新操作
    builder.setInsertionPoint(op);
    Value newResult;
    
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      newResult = builder.create<UnrealizedConversionCastOp>(
          castOp.getLoc(),
          newMemRefType,
          castOp->getOperand(0))->getResult(0);
    }
    else if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      newResult = builder.create<memref::SubViewOp>(
          subviewOp.getLoc(),
          newMemRefType,
          subviewOp.getSource(),
          subviewOp.getMixedOffsets(),
          subviewOp.getMixedSizes(),
          subviewOp.getMixedStrides())->getResult(0);
    }

    // 替换使用并删除旧操作
    oldResult.replaceAllUsesWith(newResult);
    op->erase();
  });
}

std::unique_ptr<mlir::Pass> mlir::createRemoveAddressSpacePass() {
  return std::make_unique<RemoveAddressSpacePass>();
}
