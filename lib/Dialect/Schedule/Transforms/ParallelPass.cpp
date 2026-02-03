#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Casting.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_PARALLEL
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace
{
  struct ParallelPass
      : public impl::ParallelBase<ParallelPass>
  {
    void runOnOperation() override {
      func::FuncOp funcOp = getOperation();

      // 用于存储找到的循环及其线程数
      SmallVector<std::pair<scf::ForOp, uint32_t>> parallelLoops;

      funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
        // 检查是否有 num_threads 属性
        if (auto numThreadsAttr = forOp->getAttrOfType<IntegerAttr>("num_threads")) {
          // 记录循环及其线程数
          parallelLoops.push_back({forOp, numThreadsAttr.getInt()});
        }
      });

      // 如果不存在带num_threads属性的循环，跳过
      if (parallelLoops.empty()) {
        return;  // 没有找到需要并行化的循环
      }

      // 要求带num_threads属性的循环不存在嵌套关系

      // 在函数体开头插入获取线程 ID 的操作
      OpBuilder builder(funcOp);
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      Value threadId = builder.create<mtdsp::ThreadIdOp>(
          funcOp.getLoc());  // 使用函数的位置信息

      // 处理每个带有 num_threads 属性的循环
      for (auto &[forOp, numThreads] : parallelLoops) {
        // 设置插入点到循环之前
        builder.setInsertionPoint(forOp);

        // 创建 barrier_id 常量 0
        Value barrierId = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(),
            0,  // barrier_id = 0
            builder.getI32Type()
        );
        
        // 在循环前插入 barrier
        builder.create<mtdsp::GroupBarrierOp>(
            forOp.getLoc(),
            barrierId
        );

        // 将 threadId 转换为Index类型
        Value threadIdIndex = builder.create<arith::IndexCastOp>(
            forOp.getLoc(),
            builder.getIndexType(),
            threadId
        );
        
        // 计算当前线程的起始位置: tid * step
        Value threadStart = builder.create<arith::MulIOp>(
            forOp.getLoc(),  // 使用循环的位置信息
            threadIdIndex,   // 线程ID
            forOp.getStep()  // 循环步长
        );

        // 计算新的起始位置：originalStart + threadStart
        Value newStart = builder.create<arith::AddIOp>(
            forOp.getLoc(),
            forOp.getLowerBound(),
            threadStart
        );

        // 创建 num_threads 的常量值
        Value numThreadsVal = builder.create<arith::ConstantIndexOp>(
            forOp.getLoc(),
            numThreads
        );
        
        // 计算新的步长: num_threads * step
        Value newStep = builder.create<arith::MulIOp>(
            forOp.getLoc(),
            numThreadsVal,
            forOp.getStep()
        );

        // 直接修改原循环的参数
        forOp.setLowerBound(newStart);
        forOp.setStep(newStep);

        // 设置插入点到循环后
        builder.setInsertionPointAfter(forOp);
        
        // 在循环后插入 barrier
        builder.create<mtdsp::GroupBarrierOp>(
            forOp.getLoc(),
            barrierId
        );
      }
    }
  };
}

std::unique_ptr<Pass> mlir::createParallelPass(){
  return std::make_unique<ParallelPass>();
}