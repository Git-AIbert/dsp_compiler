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

  module.walk([&](Operation *op) {
    // 处理所有操作数
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto memrefType = operand.get().getType().dyn_cast<MemRefType>()) {
        if (memrefType.getMemorySpace()) {
          // 创建新的 MemRefType，使用默认地址空间
          auto newMemrefType = MemRefType::get(memrefType.getShape(),
                                               memrefType.getElementType(),
                                               memrefType.getLayout(), nullptr);

          // 直接修改操作数的类型
          operand.get().setType(newMemrefType);
        }
      }
    }

    // 处理所有结果
    for (OpResult result : op->getResults()) {
      if (auto memrefType = result.getType().dyn_cast<MemRefType>()) {
        if (memrefType.getMemorySpace()) {
          auto newMemrefType = MemRefType::get(memrefType.getShape(),
                                               memrefType.getElementType(),
                                               memrefType.getLayout(), nullptr);
          result.setType(newMemrefType);
        }
      }
    }
  });
}

std::unique_ptr<mlir::Pass> mlir::createRemoveAddressSpacePass() {
  return std::make_unique<RemoveAddressSpacePass>();
}
