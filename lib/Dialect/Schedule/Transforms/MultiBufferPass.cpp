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

#define USE_DMA_OPT 1

namespace
{
  struct MultiBufferPass
      : public PassWrapper<MultiBufferPass, OperationPass<mlir::func::FuncOp>>
  {
    // 追踪channel的起始值
    int channelStart = 0;

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

    void findCopyOps(scf::ForOp forOp, 
                    llvm::SmallVector<linalg::CopyOp> &readCopyOps,
                    llvm::SmallVector<linalg::CopyOp> &writeCopyOps,
                    llvm::DenseMap<linalg::CopyOp, memref::AllocOp> &allocOps,
                    llvm::DenseMap<memref::AllocOp, int> &bufferFactors) {
      for (Operation &op : forOp.getRegion().front()) {
        auto copyOp = dyn_cast_or_null<linalg::CopyOp>(&op);
        if(!copyOp) continue;

        // 检查是否有 multi_buffer 属性，如果没有则跳过
        if (!copyOp->hasAttr("multi_buffer")) continue;

        auto inputType = cast<MemRefType>(copyOp.getInputs()[0].getType());
        auto outputType = cast<MemRefType>(copyOp.getOutputs()[0].getType());
        
        auto inputSpace = inputType.getMemorySpace() ? 
            cast<mtdsp::AddressSpaceAttr>(inputType.getMemorySpace()).getValue() : 
            mtdsp::AddressSpace::Global;
        auto outputSpace = outputType.getMemorySpace() ? 
            cast<mtdsp::AddressSpaceAttr>(outputType.getMemorySpace()).getValue() : 
            mtdsp::AddressSpace::Global;

        if (static_cast<unsigned>(inputSpace) < static_cast<unsigned>(outputSpace)) {
          readCopyOps.push_back(copyOp);
          auto allocOp = findAllocOp(copyOp.getOutputs()[0]);
          allocOps[copyOp] = allocOp;
          if(!bufferFactors.contains(allocOp)){
            bufferFactors[allocOp] = 2;
          } else {
            bufferFactors[allocOp] |= 2;
          }
        } else if (static_cast<unsigned>(inputSpace) > static_cast<unsigned>(outputSpace)) {
          writeCopyOps.push_back(copyOp);
          auto allocOp = findAllocOp(copyOp.getInputs()[0]);
          allocOps[copyOp] = allocOp;
          if(!bufferFactors.contains(allocOp)){
            bufferFactors[allocOp] = 4;
          } else {
            bufferFactors[allocOp] |= 4;
          }
        }
      }

      for (auto &[op, value] : bufferFactors) {
        value = 1 + __builtin_popcount(value);
      }

      // llvm::outs() << "Read operations:\n";
      // for (auto copyOp : readCopyOps) {
      //   copyOp->print(llvm::outs());
      //   llvm::outs() << "\n";
      // }

      // llvm::outs() << "Write operations:\n"; 
      // for (auto copyOp : writeCopyOps) {
      //   copyOp->print(llvm::outs());
      //   llvm::outs() << "\n";
      // }

      // llvm::outs() << "allocOps:\n";
      // for (auto &[copyOp, allocOp] : allocOps) {
      //   llvm::outs() << "copy:  " << copyOp << "\n";
      //   llvm::outs() << "alloc: " << allocOp << "\n";
      // }

      // llvm::outs() << "bufferFactors:\n";
      // for (auto &[op, value] : bufferFactors) {
      //   llvm::outs() << "alloc: " << op << "\n";
      //   llvm::outs() << "value: " << value << "\n";
      // }
      // llvm::outs() << "\n";
    }

    // 检查操作是否在指定的循环内
    bool isInLoop(Operation *op, scf::ForOp loop) {
      return op->getParentOp() == loop.getOperation();
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
      memref::AllocOp allocOp, OpBuilder &builder, int factor){
      // 获取原始alloc的类型
      auto memrefType = cast<MemRefType>(allocOp.getResult().getType());
      auto shape = memrefType.getShape();
      auto elementType = memrefType.getElementType();
      auto memorySpace = memrefType.getMemorySpace();

      // 创建新的形状，在最前面添加一个维度
      SmallVector<int64_t> newShape{factor};
      newShape.append(shape.begin(), shape.end());

      // 创建新的memref类型，注意这里不使用原始的layout
      auto newMemRefType = MemRefType::get(
          newShape,
          elementType,
          AffineMap(), // 使用默认layout
          memorySpace);

      // 设置插入点到原始alloc之后
      builder.setInsertionPointAfter(allocOp);

      // 创建新的alloc操作
      auto newAllocOp = builder.create<memref::AllocOp>(
          allocOp.getLoc(),
          newMemRefType,
          /*dynamicSizes=*/ValueRange{},
          /*symbolOperands=*/ValueRange{},
          /*alignmentAttr=*/allocOp.getAlignmentAttr());

      // llvm::outs() << "Created new allocation:\n  ";
      // newAllocOp.print(llvm::outs());
      // llvm::outs() << "\n";

      return newAllocOp;
    }

    memref::SubViewOp getBufferSlice(int factor, Value position, Value stepSize, memref::AllocOp newAllocOp,
                                      Location loc, OpBuilder &builder){
      // 创建缓冲因子常量
      auto cFactor = builder.create<arith::ConstantOp>(
          loc,
          builder.getIndexType(),
          builder.getIndexAttr(factor));

      // 1. 创建 divi 操作
      auto diviOp = builder.create<arith::DivSIOp>(
          loc,
          position, // 使用传入的位置参数
          stepSize);

      // 2. 创建 remsi 操作
      auto remsiOp = builder.create<arith::RemUIOp>(
          loc,
          diviOp.getResult(),
          cFactor);

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

    IRMapping createMapping(
        scf::ForOp forOp,
        Value curIV, // 当前迭代变量
        llvm::SmallVector<linalg::CopyOp> &readCopyOps,
        llvm::DenseMap<linalg::CopyOp, memref::AllocOp> &allocOps,
        llvm::DenseMap<memref::AllocOp, memref::AllocOp> &newAllocMap,
        llvm::DenseMap<memref::AllocOp, int> &bufferFactors,
        OpBuilder &builder){
      
      // 创建映射关系
      IRMapping prefetchMapping;
      prefetchMapping.map(forOp.getInductionVar(), curIV);

      for (linalg::CopyOp copyOp : readCopyOps){
        auto allocOp = allocOps[copyOp];
        auto newAllocOp = newAllocMap[allocOp];
        auto factor = bufferFactors[allocOp];

        // 获取缓冲切片
        auto bufferSlice = getBufferSlice(
            factor,
            curIV,           // 当前迭代位置
            forOp.getStep(), // 步长
            newAllocOp,
            forOp.getLoc(),
            builder);
        prefetchMapping.map(allocOp.getResult(), bufferSlice.getResult());
      }

      return prefetchMapping;
    }

    llvm::SmallVector<Operation *> cloneRelatedOps(
        IRMapping &mapping,
        scf::ForOp forOp,
        llvm::SmallPtrSet<Operation *, 8> &relatedOps,
        OpBuilder &builder) {
      // 用于存储所有克隆出的操作
      llvm::SmallVector<Operation *> clonedOps;

      // 遍历原始for循环中的所有操作
      for (Operation &op : forOp.getRegion().front()) {
        // 检查当前操作是否在我们收集的相关操作集合中
        if (relatedOps.count(&op)) {
          // 使用提供的映射关系克隆操作
          Operation *clonedOp = builder.clone(op, mapping);

          // 更新结果类型
          updateResultType(clonedOp);

          // 更新映射关系
          mapping.map(&op, clonedOp);

          // 将克隆的操作添加到返回列表中
          clonedOps.push_back(clonedOp);

          // // 输出 clone 的信息（用于调试）
          // llvm::outs() << "Cloned operation for prefetch:\n";
          // llvm::outs() << "  Original: ";
          // op.print(llvm::outs());
          // llvm::outs() << "\n  Cloned: ";
          // clonedOp->print(llvm::outs());
          // llvm::outs() << "\n";
        }
      }

      return clonedOps;
    }

    llvm::SmallVector<Value> createDMAOps(
        IRMapping &prefetchMapping,
        Value position,
        Value stepSize,
        llvm::DenseMap<linalg::CopyOp, Value> &copyOpsChannelStart,
        llvm::SmallVector<linalg::CopyOp> &readCopyOps,
        OpBuilder &builder) {
      llvm::SmallVector<Value> channels;

      // 创建常量2
      auto c2 = builder.create<arith::ConstantOp>(
          builder.getUnknownLoc(),
          builder.getIndexType(),
          builder.getIndexAttr(2));

      // 创建 divi 操作
      auto diviOp = builder.create<arith::DivSIOp>(
          builder.getUnknownLoc(),
          position, // 使用传入的位置参数
          stepSize);

      // 创建 remsi 操作
      auto remsiOp = builder.create<arith::RemUIOp>(
          builder.getUnknownLoc(),
          diviOp.getResult(),
          c2);

      // 使用映射为readCopyOps中的每个CopyOp创建对应的DMAOp
      for (linalg::CopyOp copyOp : readCopyOps) {

        Value inputChannel = builder.create<arith::AddIOp>(
            copyOp.getLoc(),
            copyOpsChannelStart[copyOp],
            remsiOp.getResult()
        );

        // 将index转换为i32
        Value inputChannelI32 = builder.create<arith::IndexCastOp>(
            copyOp.getLoc(),
            builder.getI32Type(),  // 目标类型
            inputChannel           // 源值
        );

#if USE_DMA_OPT
        auto channel = builder.create<mtdsp::DMAOptOp>(
            copyOp.getLoc(),
            prefetchMapping.lookup(copyOp.getInputs()[0]),
            prefetchMapping.lookup(copyOp.getOutputs()[0]),
            inputChannelI32);
#else
        auto channel = builder.create<mtdsp::DMAOp>(
            copyOp.getLoc(),
            prefetchMapping.lookup(copyOp.getInputs()[0]),
            prefetchMapping.lookup(copyOp.getOutputs()[0]));
#endif
        channels.push_back(channel);
      }

      return channels;
    }

    llvm::SmallVector<Value> createPrefetch(
        Value curIV,                                   // 当前迭代变量（可以是 getLowerBound 或 nextIV）
        llvm::SmallVector<linalg::CopyOp> &readCopyOps,
        llvm::DenseMap<linalg::CopyOp, memref::AllocOp> &allocOps,
        llvm::DenseMap<memref::AllocOp, memref::AllocOp> &newAllocMap,
        llvm::DenseMap<memref::AllocOp, int> &bufferFactors,
        scf::ForOp forOp,                              // 原始的 for 循环
        llvm::SmallPtrSet<Operation *, 8> &relatedOps, // 相关操作集合
        llvm::DenseMap<linalg::CopyOp, Value> &copyOpsChannelStart,
        OpBuilder &builder){                           // 用于创建操作的 builder

      // 创建缓冲切片并映射
      IRMapping prefetchMapping = createMapping(
          forOp,
          curIV,
          readCopyOps,
          allocOps,
          newAllocMap,
          bufferFactors,
          builder);

      // clone所有相关操作
      auto clonedOps = cloneRelatedOps(
          prefetchMapping,
          forOp,
          relatedOps,
          builder);

      // 创建DMA操作并返回channels
      return createDMAOps(
          prefetchMapping,
          curIV,
          forOp.getStep(),
          copyOpsChannelStart,
          readCopyOps,
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

    Value createIsNotFirstIterCheck(
        Value curIV,  // 当前迭代的位置
        Value lb,     // 下界
        Location loc, // 位置信息
        OpBuilder &builder){ // 用于创建新操作的 builder
      return builder.create<arith::CmpIOp>(
          loc,
          arith::CmpIPredicate::ne, // 使用不等于判断
          curIV,                    // 当前迭代位置
          lb                        // 下界
      );
    }

    // 创建循环内的预取条件分支
    scf::IfOp createLoopPrefetch(
        scf::ForOp newForOp,                           // 新创建的 for 循环
        llvm::SmallVector<linalg::CopyOp> &readCopyOps,
        llvm::DenseMap<linalg::CopyOp, memref::AllocOp> &allocOps,
        llvm::DenseMap<memref::AllocOp, memref::AllocOp> &newAllocMap,
        llvm::DenseMap<memref::AllocOp, int> &bufferFactors,
        scf::ForOp forOp,                              // 原始的 for 循环
        llvm::SmallPtrSet<Operation *, 8> &relatedOps, // 相关操作集合
        llvm::DenseMap<linalg::CopyOp, Value> &copyOpsChannelStart,
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
      SmallVector<Type> resultTypes(readCopyOps.size(), channelType);
      scf::IfOp ifOp = builder.create<scf::IfOp>(
          forOp.getLoc(),
          TypeRange(resultTypes),
          isNotLastIter,
          /*withElseRegion=*/true);

      // 设置 then 分支的预取操作
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      auto nextChannels = createPrefetch(
          nextIV,       // 使用下一次迭代位置
          readCopyOps,
          allocOps,
          newAllocMap,
          bufferFactors,
          forOp,
          relatedOps,
          copyOpsChannelStart,
          builder);
      
      builder.create<scf::YieldOp>(forOp.getLoc(), nextChannels);

      // 设置 else 分支返回常量
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto zero = builder.create<arith::ConstantIntOp>(
                             forOp.getLoc(),
                             0,
                             channelType)
                      .getResult();
      llvm::SmallVector<mlir::Value> zeros(readCopyOps.size(), zero);
      builder.create<scf::YieldOp>(forOp.getLoc(), zeros);

      return ifOp;
    }

    llvm::SmallVector<Value> cloneLoopOperations(
        scf::ForOp forOp,                                      // 原始的for循环
        IRMapping &currentMapping,                             // 当前的映射关系
        const llvm::SmallVector<linalg::CopyOp> &readCopyOps,  // 读操作列表
        const llvm::SmallVector<linalg::CopyOp> &writeCopyOps, // 写操作列表
        llvm::DenseMap<linalg::CopyOp, Value> &copyOpsChannelStart,
        OpBuilder &builder) {                                  // 操作构建器
      
      llvm::SmallVector<Value> writeChannels;

      // 遍历for循环中的所有操作
      for (Operation &op : forOp.getRegion().front()) {
        // 如果是终结符(scf.yield)，跳过
        if (isa<scf::YieldOp>(op))
          continue;
        
        if (auto currentCopyOp = dyn_cast<linalg::CopyOp>(&op)) {
          // 判断是否存在于 readCopyOps 中
          bool isReadCopy = llvm::is_contained(readCopyOps, currentCopyOp);
          if (isReadCopy) {
            continue;
          }
          bool isWriteCopy = llvm::is_contained(writeCopyOps, currentCopyOp);
          if (isWriteCopy) {

            // 创建常量2
            auto c2 = builder.create<arith::ConstantOp>(
                currentCopyOp.getLoc(),
                builder.getIndexType(),
                builder.getIndexAttr(2));

            // 创建 divi 操作
            auto diviOp = builder.create<arith::DivSIOp>(
                currentCopyOp.getLoc(),
                currentMapping.lookup(forOp.getInductionVar()),
                forOp.getStep());

            // 创建 remsi 操作
            auto remsiOp = builder.create<arith::RemUIOp>(
                currentCopyOp.getLoc(),
                diviOp.getResult(),
                c2);

            Value inputChannel = builder.create<arith::AddIOp>(
                currentCopyOp.getLoc(),
                copyOpsChannelStart[currentCopyOp],
                remsiOp.getResult()
            );

            // 将index转换为i32
            Value inputChannelI32 = builder.create<arith::IndexCastOp>(
                currentCopyOp.getLoc(),
                builder.getI32Type(),  // 目标类型
                inputChannel           // 源值
            );

#if USE_DMA_OPT
            auto channel = builder.create<mtdsp::DMAOptOp>(
                currentCopyOp.getLoc(),
                currentMapping.lookup(currentCopyOp.getInputs()[0]),
                currentMapping.lookup(currentCopyOp.getOutputs()[0]),
                inputChannelI32);
#else
            auto channel = builder.create<mtdsp::DMAOp>(
                currentCopyOp.getLoc(),
                currentMapping.lookup(currentCopyOp.getInputs()[0]),
                currentMapping.lookup(currentCopyOp.getOutputs()[0]));
#endif
            writeChannels.push_back(channel);
            continue;
          }
        }

        // clone操作并记录映射
        Operation *clonedOp = builder.clone(op, currentMapping);

        // 更新结果类型
        updateResultType(clonedOp);

        // 更新映射关系
        currentMapping.map(&op, clonedOp);

        // // 输出debug信息
        // llvm::outs() << "Cloned operation for current iteration:\n";
        // llvm::outs() << "  Original: ";
        // op.print(llvm::outs());
        // llvm::outs() << "\n  Cloned: ";
        // clonedOp->print(llvm::outs());
        // llvm::outs() << "\n";
      }

      return writeChannels;
    }

    scf::ForOp multiBufferize(scf::ForOp forOp, int level) {
      // if(
      //   level == 1 
      //   || 
      //   level == 2 
      //   || 
      //   level == 3 
      //   || 
      //   level == 4
      // )
      //   return forOp;

      scf::ForOp newForOp = forOp;
      OpBuilder builder(forOp);

      // 遍历当前for循环中的所有copyOp，将其记录在readCopyOps和writeCopyOps中
      llvm::SmallVector<linalg::CopyOp> readCopyOps;
      llvm::SmallVector<linalg::CopyOp> writeCopyOps;
      llvm::DenseMap<linalg::CopyOp, memref::AllocOp> allocOps;
      llvm::DenseMap<memref::AllocOp, int> bufferFactors;

      findCopyOps(forOp, readCopyOps, writeCopyOps, allocOps, bufferFactors);

      if(readCopyOps.empty() && writeCopyOps.empty())
        return newForOp;

      // 创建每个copyOp到channel的映射，从常量0开始，每次递增2（区分本次循环和上次循环）
      llvm::DenseMap<linalg::CopyOp, Value> copyOpsChannelStart;
      for (linalg::CopyOp copyOp : readCopyOps) {
        // 创建常量
        auto constant = builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(),
            builder.getIndexType(),
            builder.getIndexAttr(channelStart));
        copyOpsChannelStart[copyOp] = constant;
        channelStart += 2;
      }
      for (linalg::CopyOp copyOp : writeCopyOps) {
        // 创建常量
        auto constant = builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(),
            builder.getIndexType(),
            builder.getIndexAttr(channelStart));
        copyOpsChannelStart[copyOp] = constant;
        channelStart += 2;
      }

      // 创建多缓冲的 allocOp
      llvm::DenseMap<memref::AllocOp, memref::AllocOp> newAllocMap;
      for (auto [allocOp, factor] : bufferFactors) {
        auto newAllocOp = createMultiBufferAlloc(allocOp, builder, factor);
        newAllocMap[allocOp] = newAllocOp;
      }

      llvm::SmallPtrSet<Operation *, 8> visitedOps;
      llvm::SmallPtrSet<Operation *, 8> relatedOps;
      if (!readCopyOps.empty()) {
        // 收集readCopyOps中所有 read 的相关操作
        for (linalg::CopyOp copyOp : readCopyOps) {
          for (Value input : copyOp->getOperands()) {
            collectOperandDefs(input, forOp, visitedOps, relatedOps);
          }
        }
        // // 输出收集到的 read 相关操作
        // llvm::outs() << "Found related operations for CopyOp:\n";
        // for (Operation *op : relatedOps) {
        //   llvm::outs() << "  ";
        //   op->print(llvm::outs());
        //   llvm::outs() << "\n";
        // }
        // llvm::outs() << "\n";
      }

      // 将插入位置设置为for循环之前
      builder.setInsertionPoint(forOp);

      // 添加原始 forOp 的迭代参数初始值
      llvm::SmallVector<Value> initIterArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
      if (!writeCopyOps.empty()) {
        mlir::Value zero = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(),
            0,
            builder.getIntegerType(32));
        // 创建写通道的初始通道值（全0）并添加到初始通道列表中
        llvm::SmallVector<Value> initWriteChannels(writeCopyOps.size(), zero);
        initIterArgs.append(initWriteChannels);
      }

      if (!readCopyOps.empty()) {
        // 创建初始预取并记录预取返回的channel
        auto initReadChannels = createPrefetch(
            forOp.getLowerBound(),
            readCopyOps,
            allocOps,
            newAllocMap,
            bufferFactors,
            forOp,
            relatedOps,
            copyOpsChannelStart,
            builder);
        initIterArgs.append(initReadChannels.begin(), initReadChannels.end());
      }

      // 创建一个与原for循环相同的循环，但是添加迭代参数，迭代变量的值等于dma的返回值
      newForOp = builder.create<scf::ForOp>(
          forOp.getLoc(),
          forOp.getLowerBound(),  // lower bound
          forOp.getUpperBound(),  // upper bound
          forOp.getStep(),        // step
          initIterArgs            // 初始值
      );
      // 复制原循环的所有属性
      for (const NamedAttribute &attr : forOp->getAttrs()) {
        newForOp->setAttr(attr.getName(), attr.getValue());
      }
      // 获取for循环迭代实参
      auto iterArgs = newForOp.getRegionIterArgs();
      auto writeChannelsStart = iterArgs.begin() + forOp.getInitArgs().size();
      auto readChannelsStart = writeChannelsStart + writeCopyOps.size();
      // 分别获取写和读的channel实参
      llvm::SmallVector<Value> writeChannelArgs(
          writeChannelsStart,
          readChannelsStart);
      llvm::SmallVector<Value> readChannelArgs(
          readChannelsStart,
          readChannelsStart + readCopyOps.size());

      // 将插入位置设置为for循环之中
      builder.setInsertionPointToStart(&newForOp.getRegion().front());

      llvm::SmallVector<Value> readChannels;
      if(!readCopyOps.empty()){
        // 创建循环预取并记录预取返回的channel
        auto ifOpForRead = createLoopPrefetch(
            newForOp,
            readCopyOps,
            allocOps,
            newAllocMap,
            bufferFactors,
            forOp, 
            relatedOps, 
            copyOpsChannelStart,
            builder);
        for (OpResult result : ifOpForRead.getResults()) {
          readChannels.push_back(result);
        }

        // 将插入位置设置为if之后
        builder.setInsertionPointAfter(ifOpForRead);
        // 创建read的wait操作
        for (Value readChannel : readChannelArgs) {
          builder.create<mtdsp::WaitOp>(
            forOp.getLoc(),
            readChannel
          );
        }
      }

      // 创建映射关系
      IRMapping currentMapping = createMapping(
          forOp,
          newForOp.getInductionVar(),
          readCopyOps,
          allocOps,
          newAllocMap,
          bufferFactors,
          builder);

      // clone原始for循环中的操作
      llvm::SmallVector<Value> writeChannels = cloneLoopOperations(
          forOp, 
          currentMapping, 
          readCopyOps, 
          writeCopyOps, 
          copyOpsChannelStart,
          builder);

      if(!writeCopyOps.empty()){
        // 判断是否是第一次迭代，如果不是，wait对应位置的write的迭代参数
        Value isNotFirstIter = createIsNotFirstIterCheck(
            newForOp.getInductionVar(),
            forOp.getLowerBound(),
            forOp.getLoc(),
            builder);
        // 创建wait操作，操作数为迭代参数
        scf::IfOp ifOpForWrite = builder.create<scf::IfOp>(
            forOp.getLoc(),
            isNotFirstIter,
            [&](OpBuilder &builder, Location loc){
              // 创建write的wait操作
              for (Value writeChannel : writeChannelArgs){
                builder.create<mtdsp::WaitOp>(loc, writeChannel);
              }
              builder.create<scf::YieldOp>(forOp.getLoc());
            },
            nullptr);
      }

      // 创建 yield 操作
      llvm::SmallVector<Value> allYieldArgs;
      // 获取原始 forOp 的 yield 操作的参数
      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
      allYieldArgs.append(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
      // 添加所有的 channels
      allYieldArgs.append(writeChannels.begin(), writeChannels.end());
      allYieldArgs.append(readChannels.begin(), readChannels.end());
      builder.create<scf::YieldOp>(forOp.getLoc(), allYieldArgs);

      if (!writeCopyOps.empty()) {
        // 将插入位置设置为 newForOp 之后
        builder.setInsertionPointAfter(newForOp);
        // 获取写通道的最终返回值，用于最后的同步等待
        auto writeChannelsStart = 
            newForOp.getResults().begin()
            + forOp.getInitArgs().size();
        llvm::SmallVector<Value> finalWriteChannels(
            writeChannelsStart,
            writeChannelsStart + writeCopyOps.size());
        // 为每个写通道创建最后的wait操作，确保所有写操作完成
        for (Value finalChannel : finalWriteChannels) {
            builder.create<mtdsp::WaitOp>(
                forOp.getLoc(),
                finalChannel
            );
        }
      }

      // // 在删除 forOp 之前
      // llvm::outs() << "Function content before erasing original forOp:\n";
      // // 获取包含 forOp 的函数
      // if (auto parentFunc = forOp->getParentOfType<func::FuncOp>()) {
      //     parentFunc.print(llvm::outs());
      //     llvm::outs() << "\n";
      // } else {
      //     llvm::outs() << "Could not find parent function\n";
      // }

      // 删除forOp
      forOp.erase();

      return newForOp;
    }

    void recursiveTraverseForOps(scf::ForOp forOp, int level){
      // llvm::outs() << "Level " << level << " loop:\n";
      // forOp->print(llvm::outs());
      // llvm::outs() << "\n";

      scf::ForOp newForOp = multiBufferize(forOp, level);

      // 先收集所有子循环
      llvm::SmallVector<scf::ForOp> childLoops;
      for (Operation &op : newForOp.getRegion().front()) {
        if (auto childForOp = dyn_cast<scf::ForOp>(&op)) {
          childLoops.push_back(childForOp);
        }
      }

      // 再递归处理子循环
      for (auto childForOp : childLoops) {
        recursiveTraverseForOps(childForOp, level + 1);
      }
    }

    void runOnOperation() override {
      func::FuncOp funcOp = getOperation();

      funcOp.walk([&](scf::ForOp forOp) {
        if (isa<func::FuncOp>(forOp->getParentOp())) {
          recursiveTraverseForOps(forOp, 0);
        }
      });

      // 处理剩余的 CopyOp
      OpBuilder builder(funcOp);
      funcOp.walk([&](linalg::CopyOp copyOp) {
        builder.setInsertionPoint(copyOp);
        // 为每个 CopyOp 创建一个 channel 常量
        auto channelConst = builder.create<arith::ConstantOp>(
            copyOp.getLoc(),
            builder.getIndexType(),
            builder.getIndexAttr(channelStart));
        
        // 将 index 类型转换为 i32 类型
        auto channelI32 = builder.create<arith::IndexCastOp>(
            copyOp.getLoc(),
            builder.getI32Type(),
            channelConst);

        // 创建 DMAOptOp 替换 CopyOp
        auto dmaOp = builder.create<mtdsp::DMAOptOp>(
            copyOp.getLoc(),
            copyOp.getInputs()[0],    // 输入操作数
            copyOp.getOutputs()[0],   // 输出操作数
            channelI32                 // channel 参数
        );

        // 创建 WaitOp
        builder.create<mtdsp::WaitOp>(
            copyOp.getLoc(),
            dmaOp->getResult(0)
        );

        // 增加 channelStart 的值，为下一个转换准备
        channelStart++;

        // 删除原始的 CopyOp
        copyOp.erase();
      });
    }
  };
}

std::unique_ptr<Pass> mlir::createMultiBufferPass(){
  return std::make_unique<MultiBufferPass>();
}