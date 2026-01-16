#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Casting.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_UNROLL
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace
{
  struct UnrollPass
      : public impl::UnrollBase<UnrollPass>
  {
    void runOnOperation() override {
      func::FuncOp funcOp = getOperation();

      // 从内到外遍历for循环
      funcOp.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
        // 检查是否有 unroll_factor 属性
        if (auto unrollFactorAttr = forOp->getAttrOfType<IntegerAttr>("unroll_factor")) {
          // 使用 loopUnrollByFactor 进行展开
          LogicalResult result = loopUnrollByFactor(forOp, unrollFactorAttr.getInt());
          if (failed(result)) {
            forOp->emitError() << "failed to unroll loop by factor " 
                            << unrollFactorAttr.getInt();
          }
        }
      });
    }
  };
}

std::unique_ptr<Pass> mlir::createUnrollPass(){
  return std::make_unique<UnrollPass>();
}