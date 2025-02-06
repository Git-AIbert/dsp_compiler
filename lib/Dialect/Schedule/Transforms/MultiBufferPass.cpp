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

using namespace mlir;

namespace
{
  struct MultiBufferPass
      : public PassWrapper<MultiBufferPass, OperationPass<mlir::func::FuncOp>>
  {
    // 追踪定义链以找到 AllocOp
    memref::AllocOp findAllocOp(Value memref){
      Value current = memref;
      while (current.getDefiningOp()){
        if (auto allocOp = dyn_cast<memref::AllocOp>(current.getDefiningOp())){
          return allocOp;
        }
        // TODO: 不同的操作使用的 memref 的位置可能不同，目前仅考虑了 subview 操作
        current = current.getDefiningOp()->getOperand(0);
      }
      return nullptr;
    }

    // // 获取操作所在的最内层循环
    // Operation *getInnermostLoop(Operation *op){
    //   Operation *parent = op->getParentOp();
    //   while (parent && !isa<scf::ForOp>(parent)){
    //     parent = parent->getParentOp();
    //   }
    //   return parent;
    // }

    // 检查操作是否在指定的循环内
    bool isInLoop(Operation *op, Operation *loop){
      Operation *parent = op->getParentOp();
      while (parent && parent != loop){
        parent = parent->getParentOp();
      }
      return parent == loop;
    }

    // 递归收集操作数的定义（只收集循环内的操作）
    void collectOperandDefs(Value value, Operation *loop,
                            llvm::SmallPtrSet<Operation *, 8> &visitedOps,
                            llvm::SmallPtrSet<Operation *, 8> &relatedOps){
      if (auto defOp = value.getDefiningOp()){
        // 检查操作是否在目标循环内
        if (!isInLoop(defOp, loop)){
          return;
        }

        // 如果这个操作已经访问过，直接返回
        if (!visitedOps.insert(defOp).second){
          return;
        }

        // 将当前操作加入相关操作集合
        relatedOps.insert(defOp);

        // 递归处理这个操作的所有操作数
        for (Value operand : defOp->getOperands()){
          collectOperandDefs(operand, loop, visitedOps, relatedOps);
        }
      }
    }

    // 获取操作所在的最内层循环
    scf::ForOp getInnermostLoop(Operation *op) {
      Operation *parent = op->getParentOp();
      while (parent && !isa<scf::ForOp>(parent)) {
        parent = parent->getParentOp();
      }
      return dyn_cast_or_null<scf::ForOp>(parent);
    }

    // 检查操作是否在指定的循环内
    bool isInLoop(Operation *op, scf::ForOp loop) {
      Operation *parent = op->getParentOp();
      while (parent && parent != loop.getOperation()) {
        parent = parent->getParentOp();
      }
      return parent == loop.getOperation();
    }

    // 递归收集操作数的定义（只收集循环内的操作）
    void collectOperandDefs(Value value, scf::ForOp loop,
                          llvm::SmallPtrSet<Operation *, 8> &visitedOps,
                          llvm::SmallPtrSet<Operation *, 8> &relatedOps) {
      if (auto defOp = value.getDefiningOp()) {
        // 检查操作是否在目标循环内
        if (!isInLoop(defOp, loop)) {
          return;
        }

        // 如果这个操作已经访问过，直接返回
        if (!visitedOps.insert(defOp).second) {
          return;
        }

        // 将当前操作加入相关操作集合
        relatedOps.insert(defOp);

        // 递归处理这个操作的所有操作数
        for (Value operand : defOp->getOperands()) {
          collectOperandDefs(operand, loop, visitedOps, relatedOps);
        }
      }
    }

    memref::AllocOp createMultiBufferAlloc(
      memref::AllocOp allocOp, OpBuilder &builder){
      // 获取原始alloc的类型
      auto memrefType = cast<MemRefType>(allocOp.getResult().getType());
      auto shape = memrefType.getShape();
      auto elementType = memrefType.getElementType();
      auto memorySpace = memrefType.getMemorySpace();

      // 创建新的形状，在最前面添加一个维度2
      SmallVector<int64_t> newShape{2};
      newShape.append(shape.begin(), shape.end());

      // 创建新的memref类型，注意这里不使用原始的layout
      auto newMemRefType = MemRefType::get(
          newShape,
          elementType,
          AffineMap(), // 使用默认layout
          memorySpace);

      // 创建新的alloc操作
      auto newAllocOp = builder.create<memref::AllocOp>(
          allocOp.getLoc(),
          newMemRefType,
          /*dynamicSizes=*/ValueRange{},
          /*symbolOperands=*/ValueRange{},
          /*alignmentAttr=*/allocOp.getAlignmentAttr());

      llvm::outs() << "Created new allocation:\n  ";
      newAllocOp.print(llvm::outs());
      llvm::outs() << "\n";

      return newAllocOp;
    }

    memref::SubViewOp getBufferSlice(Value position, Value stepSize, memref::AllocOp newAllocOp,
                                      Location loc, OpBuilder &builder){
      // 创建常量2
      auto c2 = builder.create<arith::ConstantOp>(
          loc,
          builder.getIndexType(),
          builder.getIndexAttr(2));

      // 1. 创建 divi 操作
      auto diviOp = builder.create<arith::DivSIOp>(
          loc,
          position, // 使用传入的位置参数
          stepSize);

      // 2. 创建 remsi 操作
      auto remsiOp = builder.create<arith::RemUIOp>(
          loc,
          diviOp.getResult(),
          c2);

      // 3. 创建 subview 操作
      auto sourceType = cast<MemRefType>(newAllocOp.getResult().getType());
      auto shape = sourceType.getShape();

      // 创建offset、size和stride
      SmallVector<OpFoldResult> offsets = {
          remsiOp.getResult(),     // 第一维度使用计算结果
          builder.getIndexAttr(0), // 第二维度offset为0
          builder.getIndexAttr(0)  // 第三维度offset为0
      };
      SmallVector<OpFoldResult> sizes = {
          builder.getIndexAttr(1),        // 第一维度大小为1
          builder.getIndexAttr(shape[1]), // 第二维度使用原始大小
          builder.getIndexAttr(shape[2])  // 第三维度使用原始大小
      };
      SmallVector<OpFoldResult> strides(3, builder.getIndexAttr(1)); // 所有步长都为1

      // 使用inferRankReducedResultType推导降维后的类型
      auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                            {sourceType.getDimSize(1), sourceType.getDimSize(2)}, // 新的形状
                            sourceType,                                           // 源类型
                            offsets,                                              // 偏移
                            sizes,                                                // 大小
                            strides                                               // 步长
                            ));

      return builder.create<memref::SubViewOp>(
          loc,
          resultType,
          newAllocOp.getResult(),
          offsets,
          sizes,
          strides);
    }

    void updateResultType(Operation *op) {
      // 处理 SubViewOp
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        // 获取所有必要的参数
        SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = subviewOp.getMixedStrides();

        // 获取源类型
        auto sourceType = cast<MemRefType>(subviewOp.getSource().getType());

        // 推导正确的结果类型
        auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                              sourceType.getShape(),
                              sourceType,
                              mixedOffsets,
                              mixedSizes,
                              mixedStrides));

        // 更新结果类型
        subviewOp.getResult().setType(resultType);
      }
      // TODO: 添加其他操作类型的处理逻辑
    }

    Value createPrefetch(
        IRMapping &mapping,                      // 预先创建好的映射关系
        scf::ForOp &forOp,                             // 原始的for循环
        llvm::SmallPtrSet<Operation *, 8> &relatedOps, // 相关操作集合
        linalg::CopyOp copyOp,                         // 原始的copy操作
        OpBuilder &builder){

      // clone 原始 for 循环中集合中的操作，使用传入的映射
      for (Operation &op : forOp.getRegion().front()){
        // 检查当前操作是否在我们收集的集合中
        if (relatedOps.count(&op)){
          // clone 操作并使用传入的映射
          Operation *clonedOp = builder.clone(op, mapping);

          // 更新结果类型
          updateResultType(clonedOp);

          // 更新映射关系
          mapping.map(&op, clonedOp);

          // 输出 clone 的信息（用于调试）
          llvm::outs() << "Cloned operation for prefetch:\n";
          llvm::outs() << "  Original: ";
          op.print(llvm::outs());
          llvm::outs() << "\n  Cloned: ";
          clonedOp->print(llvm::outs());
          llvm::outs() << "\n";
        }
      }

      // 创建一个 dma 操作，使用映射后的值
      return builder.create<mtdsp::DMAOp>(
          copyOp.getLoc(),
          mapping.lookup(copyOp.getInputs()[0]),
          mapping.lookup(copyOp.getOutputs()[0]));
    }

    Value createBufferSliceAndPrefetch(
        Value curIV,                                   // 当前迭代变量（可以是 getLowerBound 或 nextIV）
        memref::AllocOp allocOp,                       // 原始的 alloc 操作
        memref::AllocOp newAllocOp,                    // 新创建的 alloc 操作
        scf::ForOp forOp,                              // 原始的 for 循环
        llvm::SmallPtrSet<Operation *, 8> &relatedOps, // 相关操作集合
        linalg::CopyOp copyOp,                         // 原始的 copy 操作
        OpBuilder &builder){                           // 用于创建操作的 builder

      // 获取缓冲切片
      auto bufferSlice = getBufferSlice(
          curIV,           // 当前迭代位置
          forOp.getStep(), // 步长
          newAllocOp,
          forOp.getLoc(),
          builder);

      // 创建映射关系
      IRMapping prefetchMapping;
      prefetchMapping.map(forOp.getInductionVar(), curIV);
      prefetchMapping.map(allocOp.getResult(), bufferSlice.getResult());

      // 创建预取操作并返回
      return createPrefetch(
          prefetchMapping,
          forOp,
          relatedOps,
          copyOp,
          builder);
    }

    Value createIsNotLastIterCheck(
        Value nextIV, // 下一次迭代的位置
        Value step,   // 步长
        Value ub,     // 上界
        Location loc, // 位置信息
        OpBuilder &builder){ // 用于创建新操作的 builder

      // 检查步长是否是常量
      if (auto constantOp = step.getDefiningOp<arith::ConstantOp>()){
        // 如果步长是常量，直接判断
        int64_t stepValue = cast<IntegerAttr>(constantOp.getValue()).getInt();

        // 根据步长正负选择比较谓词
        arith::CmpIPredicate pred = stepValue > 0 ? arith::CmpIPredicate::slt : // 步长为正，用小于
                                        arith::CmpIPredicate::sgt;              // 步长为负，用大于

        // 创建比较操作并返回结果
        return builder.create<arith::CmpIOp>(
            loc,
            pred,
            nextIV, // 下一次迭代位置
            ub      // 上界
        );
      } else {
        // 如果步长不是常量，需要动态判断

        // 1. 先判断步长的符号
        Value isPositiveStep = builder.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::sgt,
            step,
            builder.create<arith::ConstantIndexOp>(loc, 0));

        // 2. 分别创建两种情况的比较
        Value ltCmp = builder.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::slt,
            nextIV,
            ub);

        Value gtCmp = builder.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::sgt,
            nextIV,
            ub);

        // 3. 根据步长符号选择并返回正确的比较结果
        return builder.create<arith::SelectOp>(
            loc,
            isPositiveStep,
            ltCmp,
            gtCmp);
      }
    }

    // 创建循环内的预取条件分支
    scf::IfOp createLoopPrefetch(
        scf::ForOp newForOp,                           // 新创建的 for 循环
        memref::AllocOp allocOp,                       // 原始的 alloc 操作
        memref::AllocOp newAllocOp,                    // 新创建的 alloc 操作
        scf::ForOp forOp,                              // 原始的 for 循环
        llvm::SmallPtrSet<Operation *, 8> &relatedOps, // 相关操作集合
        linalg::CopyOp copyOp,                         // 原始的 copy 操作
        OpBuilder &builder){ // 用于创建操作的 builder

      // 计算下一次迭代的位置
      Value nextIV = builder.create<arith::AddIOp>(
          forOp.getLoc(),
          newForOp.getInductionVar(),    // 直接从 newForOp 获取当前迭代变量
          forOp.getStep()                // 直接使用 forOp.getStep()
      );

      // 判断是否是最后一次迭代
      Value isNotLastIter = createIsNotLastIterCheck(
          nextIV,
          forOp.getStep(),
          forOp.getUpperBound(),
          forOp.getLoc(),
          builder);

      // 创建条件分支操作
      Type channelType = builder.getIntegerType(32);
      scf::IfOp ifOp = builder.create<scf::IfOp>(
          forOp.getLoc(),
          TypeRange{channelType},
          isNotLastIter,
          /*withElseRegion=*/true);

      // 设置 then 分支的预取操作
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      auto nextChannel = createBufferSliceAndPrefetch(
          nextIV,       // 使用下一次迭代位置
          allocOp,
          newAllocOp,
          forOp,
          relatedOps,
          copyOp,
          builder
      );
      
      builder.create<scf::YieldOp>(forOp.getLoc(), nextChannel);

      // 设置 else 分支返回常量
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto zero = builder.create<arith::ConstantIntOp>(
                             forOp.getLoc(),
                             0,
                             channelType)
                      .getResult();
      builder.create<scf::YieldOp>(forOp.getLoc(), zero);

      return ifOp;
    }

    void cloneLoopBody(
        IRMapping &mapping,    // 值映射关系
        scf::ForOp &forOp,     // 原始的 for 循环
        linalg::CopyOp copyOp, // 需要排除的 copy 操作
        OpBuilder &builder){   // 用于创建新操作的 builder
      // clone原始for循环中的所有操作，除了当前处理的copy
      for (Operation &op : forOp.getRegion().front()){
        // 如果是终结符(scf.yield)或者是我们正在处理的copy，跳过
        if (isa<scf::YieldOp>(op) || &op == copyOp.getOperation())
          continue;

        // clone操作并记录映射
        Operation *clonedOp = builder.clone(op, mapping);

        // 更新结果类型
        updateResultType(clonedOp);

        // 更新映射关系
        mapping.map(&op, clonedOp);

        // 输出debug信息
        llvm::outs() << "Cloned operation for current iteration:\n";
        llvm::outs() << "  Original: ";
        op.print(llvm::outs());
        llvm::outs() << "\n  Cloned: ";
        clonedOp->print(llvm::outs());
        llvm::outs() << "\n";
      }
    }

    void runOnOperation() override {
      func::FuncOp funcOp = getOperation();
      OpBuilder builder(funcOp.getContext());

      // 用一个vector存储需要删除的操作
      std::vector<Operation *> opsToErase;

      // 遍历所有copyOp
      // 找到copyOp的outs最终使用的memref来自allocOp的copyOp
      // 找到该copyOp所在的for循环
      // 将在for循环内的copyOp的所有的操作数放入集合

      // 用于存储已访问过的操作
      llvm::SmallPtrSet<Operation *, 8> visitedOps;
      // 用于存储收集到的相关操作
      llvm::SmallPtrSet<Operation *, 8> relatedOps;

      // 遍历函数体中的所有copyOp
      funcOp.walk([&](linalg::CopyOp copyOp){
        // 获取原始的 allocOp
        auto allocOp = findAllocOp(copyOp.getOutputs()[0]);
        // 如果这个 copy 操作的输出不来自 allocOp，跳过该 copy
        if (!allocOp)
          return;

        // 找到copy操作所在的最内层循环
        auto forOp = getInnermostLoop(copyOp);
        // 如果这个 copy 操作不在循环中，跳过该 copy
        if (!forOp)
          return;

        // 清空之前的结果
        visitedOps.clear();
        relatedOps.clear();

        // 收集与该 copy 位于同一循环中的 copy 的相关操作
        for (Value input : copyOp->getOperands()) {
          collectOperandDefs(input, forOp, visitedOps, relatedOps);
        }

        // 输出收集到的 copy 相关操作
        llvm::outs() << "Found related operations for CopyOp:\n";
        for (Operation *op : relatedOps) {
          llvm::outs() << "  ";
          op->print(llvm::outs());
          llvm::outs() << "\n";
        }
        llvm::outs() << "\n";

        // 设置插入点到原始alloc之后
        builder.setInsertionPointAfter(allocOp);
        // 创建多缓冲alloc
        auto newAllocOp = createMultiBufferAlloc(allocOp, builder);

        // 将插入位置设置为for循环之前
        builder.setInsertionPoint(forOp);
        auto channel = createBufferSliceAndPrefetch(
            forOp.getLowerBound(),  // 使用循环的起始位置
            allocOp,
            newAllocOp,
            forOp,
            relatedOps,
            copyOp,
            builder
        );

        // 创建一个与原for循环相同的循环，但是添加迭代参数，迭代变量的值等于dma的返回值
        scf::ForOp newForOp = builder.create<scf::ForOp>(
            forOp.getLoc(),
            forOp.getLowerBound(),  // lower bound
            forOp.getUpperBound(),  // upper bound
            forOp.getStep(),        // step
            ValueRange{channel}     // 初始值,使用DMA的channel作为迭代参数
        );
        // 获取channel参数
        Value waitChannel = newForOp.getRegionIterArgs()[0];

        // 将插入位置设置为for循环之中
        builder.setInsertionPointToStart(&newForOp.getRegion().front());
        auto ifOp = createLoopPrefetch(
          newForOp,
          allocOp, 
          newAllocOp, 
          forOp, 
          relatedOps, 
          copyOp, 
          builder);

        // 将插入位置设置为if之后
        builder.setInsertionPointAfter(ifOp);
        // 创建wait操作，操作数为迭代参数
        builder.create<mtdsp::WaitOp>(
            forOp.getLoc(),
            waitChannel
        );

        // 获取缓冲切片
        auto bufferSlice = getBufferSlice(
            newForOp.getInductionVar(), // 当前迭代位置
            forOp.getStep(),            // 步长
            newAllocOp,
            forOp.getLoc(),
            builder
        );

        // 创建映射关系
        IRMapping currentMapping;
        currentMapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
        currentMapping.map(allocOp.getResult(), bufferSlice.getResult());
        
        // clone循环体内除copyOp之外的操作
        cloneLoopBody(currentMapping, forOp, copyOp, builder);

        // scf.yield if的返回值
        builder.create<scf::YieldOp>(forOp.getLoc(), ifOp.getResult(0));

        // 不直接删除forOp，而是将其加入待处理列表
        opsToErase.push_back(forOp);

        llvm::outs() << "funcOp:\n" << funcOp << "\n\n";
      });

      // 遍历结束后再执行删除/替换操作
      for (auto forOp : opsToErase){
        forOp->erase();
      }
    }
  };
}

std::unique_ptr<Pass> mlir::createMultiBufferPass(){
  return std::make_unique<MultiBufferPass>();
}