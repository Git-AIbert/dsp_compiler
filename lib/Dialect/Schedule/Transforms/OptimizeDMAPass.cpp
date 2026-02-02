//===----------------------------------------------------------------------===//
// OptimizeDMAPass - DMA Channel 优化分配
//===----------------------------------------------------------------------===//
//
// 功能概述：
//   为多缓冲优化后的 DMA 操作分配硬件 channel，并设置最内层循环的优先级。
//
// 主要步骤：
//   1. 收集所有 DMA group 和其所属循环的信息
//   2. 为叶子循环的 groups 分配 channels（从 0 开始）
//   3. 为非叶子循环的 groups 分配 channels（复用或递增）
//   4. 为循环外的 groups 分配 channels
//   5. 将 mtdsp.dma -> mtdsp.dma_opt，mtdsp.wait -> mtdsp.wait_p2p
//   6. 在函数入口插入 mtdsp.set_prir 设置最内层循环优先级
//
// DMA Position 类型说明：
//   - "fixed"   : 不进行多缓冲的 DMA，分配 1 个 channel
//   - "initial" : 多缓冲的初始预取，分配 2 个 channels (ping-pong)
//   - "next"    : 多缓冲的预取下一块，分配 2 个 channels (ping-pong)
//   - "current" : 多缓冲的写回当前块，分配 2 个 channels (ping-pong)
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"

#define DEBUG_TYPE "optimize-dma"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZEDMA
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace
{
  //===--------------------------------------------------------------------===//
  // 数据结构定义
  //===--------------------------------------------------------------------===//

  /// DMA Group 信息：包含同一个 group 的所有 DMA 和 Wait 操作
  struct GroupInfo {
    int groupId;                           // Group ID
    SmallVector<mtdsp::DMAOp> dmaOps;      // 该 group 的所有 DMA 操作
    SmallVector<mtdsp::WaitOp> waitOps;    // 该 group 的所有 Wait 操作
    StringRef position;                    // Position 类型 (fixed/initial/next/current)
    scf::ForOp ownerLoop;                  // 所属的循环（循环外的为 nullptr）
    int channelBase = -1;                  // 分配的 channel 基地址（-1 表示未分配）
  };

  //===--------------------------------------------------------------------===//
  // OptimizeDMAPass 主类
  //===--------------------------------------------------------------------===//

  struct OptimizeDMAPass
      : public impl::OptimizeDMABase<OptimizeDMAPass>
  {
    //===------------------------------------------------------------------===//
    // 成员变量
    //===------------------------------------------------------------------===//

    DenseMap<int, GroupInfo> allGroups;                // 所有 DMA groups 的信息
    DenseMap<scf::ForOp, SmallVector<int>> loopToGroups; // 循环 -> groups 映射
    int reservedChannels = 0;                          // 叶子循环使用的最大 channel 数
    unsigned long prirMask = 0;                        // 最内层循环的优先级掩码

    //===------------------------------------------------------------------===//
    // 辅助函数
    //===------------------------------------------------------------------===//

    /// 找到包含指定操作的最内层循环
    scf::ForOp findLoopContainingOp(Operation *op) {
      Operation *parent = op->getParentOp();
      while (parent) {
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          return forOp;
        }
        parent = parent->getParentOp();
      }
      return nullptr;
    }

    /// 判断是否是叶子循环
    /// 定义：该循环包含 groups，且其所有子循环都不包含 groups
    bool isLeafLoop(scf::ForOp loop) {
      // 检查是否有 groups
      if (loopToGroups.find(loop) == loopToGroups.end()) {
        return false;
      }

      // 检查子循环是否包含 groups
      for (Operation &op : loop.getBody()->getOperations()) {
        if (auto childLoop = dyn_cast<scf::ForOp>(&op)) {
          if (loopToGroups.find(childLoop) != loopToGroups.end()) {
            return false;  // 有子循环包含 groups，不是叶子循环
          }
        }
      }

      return true;
    }

    /// 收集函数中的所有顶层循环
    SmallVector<scf::ForOp> getTopLevelLoops(func::FuncOp funcOp) {
      SmallVector<scf::ForOp> topLoops;
      for (Operation &op : funcOp.getBody().front().getOperations()) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          topLoops.push_back(forOp);
        }
      }
      return topLoops;
    }

    //===------------------------------------------------------------------===//
    // 阶段 0：信息收集
    //===------------------------------------------------------------------===//

    /// 收集所有 DMA group 信息和循环映射关系
    void collectGroupInfo(func::FuncOp funcOp) {
      // 收集所有 DMAOp
      funcOp.walk([&](mtdsp::DMAOp dmaOp) {
        auto groupAttr = dmaOp->getAttrOfType<IntegerAttr>("group");
        auto positionAttr = dmaOp->getAttrOfType<StringAttr>("position");

        if (!groupAttr || !positionAttr) {
          return;  // 跳过没有属性的 DMA
        }

        int groupId = groupAttr.getInt();
        StringRef position = positionAttr.getValue();

        // 创建或获取 GroupInfo
        if (allGroups.find(groupId) == allGroups.end()) {
          allGroups[groupId] = GroupInfo{groupId, {}, {}, position, nullptr, -1};
        }

        allGroups[groupId].dmaOps.push_back(dmaOp);
        // 优先使用非 initial 的 position（因为 initial 在循环外）
        if (position != "initial") {
          allGroups[groupId].position = position;
        }

        // 为非 initial 的 DMA 找到所属循环
        // initial 是循环外的初始预取，其他类型都可能在循环内
        if (position != "initial") {
          scf::ForOp ownerLoop = findLoopContainingOp(dmaOp);
          if (ownerLoop) {
            allGroups[groupId].ownerLoop = ownerLoop;
            loopToGroups[ownerLoop].push_back(groupId);
          }
        }
      });

      // 收集所有 WaitOp
      funcOp.walk([&](mtdsp::WaitOp waitOp) {
        auto groupAttr = waitOp->getAttrOfType<IntegerAttr>("group");
        if (!groupAttr) {
          return;
        }

        int groupId = groupAttr.getInt();
        if (allGroups.find(groupId) != allGroups.end()) {
          allGroups[groupId].waitOps.push_back(waitOp);
        }
      });

      // Debug: 打印收集到的信息
      LDBG("========== Collected Group Information ==========");
      LDBG("Total groups: " << allGroups.size());
      for (auto &[groupId, info] : allGroups) {
        LDBG("  Group " << groupId << ":");
        LDBG("    Position: " << info.position);
        LDBG("    DMAOps count: " << info.dmaOps.size());
        for (size_t i = 0; i < info.dmaOps.size(); ++i) {
          LDBG("      DMA[" << i << "]: " << info.dmaOps[i]);
        }
        LDBG("    WaitOps count: " << info.waitOps.size());
        for (size_t i = 0; i < info.waitOps.size(); ++i) {
          LDBG("      Wait[" << i << "]: " << info.waitOps[i]);
        }
        if (info.ownerLoop) {
          LDBG("    Owner loop: " << info.ownerLoop);
        } else {
          LDBG("    Owner loop: <none>");
        }
        LDBG("    Channel base: " << info.channelBase);
      }

      LDBG("========== Loop to Groups Mapping ==========");
      LDBG("Total loops with groups: " << loopToGroups.size());
      for (auto &[loop, groups] : loopToGroups) {
        LDBG("  Loop " << loop << " contains " << groups.size() << " groups:");
        for (int gid : groups) {
          LDBG("    Group " << gid);
        }
      }
      LDBG("=================================================");
    }

    //===------------------------------------------------------------------===//
    // 阶段 1：为叶子循环分配 channels
    //===------------------------------------------------------------------===//

    /// 为叶子循环的 groups 分配 channels
    ///
    /// 策略：
    ///   - DFS 递归遍历循环树
    ///   - 只处理叶子循环（最内层包含 groups 的循环）
    ///   - 从 channel 0 开始分配
    ///   - fixed 类型分配 1 个 channel，多缓冲类型分配 2 个 channels
    ///   - 计算所有叶子循环使用的最大 channel 数 (reservedChannels)
    ///   - 将叶子循环使用的 channels 加入优先级掩码
    void allocateLeafLoopChannels(scf::ForOp loop) {
      // 递归处理所有子循环
      for (Operation &op : loop.getBody()->getOperations()) {
        if (auto childLoop = dyn_cast<scf::ForOp>(&op)) {
          allocateLeafLoopChannels(childLoop);
        }
      }

      // 只处理叶子循环
      if (!isLeafLoop(loop)) {
        return;
      }

      // 为叶子循环的 groups 分配 channels
      int nextChannel = 0;
      for (int groupId : loopToGroups.find(loop)->second) {
        allGroups[groupId].channelBase = nextChannel;

        // 根据 position 类型确定需要的 channel 数量
        StringRef position = allGroups[groupId].position;
        int channelCount;

        if (position == "fixed") {
          channelCount = 1;  // fixed 不进行多缓冲，只需 1 个 channel
        } else {
          channelCount = 2;  // initial/next/current 多缓冲，需要 2 个 channels
        }

        // 将使用的 channels 加入优先级掩码（最内层循环优先级高）
        for (int i = 0; i < channelCount; i++) {
          prirMask |= (1UL << (nextChannel + i));
        }

        nextChannel += channelCount;
      }

      // 更新全局预留 channels（取所有叶子循环的最大值）
      if (nextChannel > reservedChannels) {
        reservedChannels = nextChannel;
      }

      // Debug: 打印叶子循环处理结果
      LDBG("---------- Leaf Loop Processed ----------");
      LDBG("  Loop: " << loop);
      LDBG("  Groups in this loop: " << loopToGroups.find(loop)->second.size());
      for (int groupId : loopToGroups.find(loop)->second) {
        LDBG("    Group " << groupId << " -> channelBase: " << allGroups[groupId].channelBase);
      }
      LDBG("  Reserved channels (max so far): " << reservedChannels);
      LDBG("-----------------------------------------");
    }

    //===------------------------------------------------------------------===//
    // 阶段 2：为非叶子循环分配 channels
    //===------------------------------------------------------------------===//

    /// 为非叶子循环的 groups 分配 channels
    ///
    /// 策略：
    ///   - DFS 后序遍历（先处理子循环，再处理父循环）
    ///   - 跳过叶子循环（已在阶段 1 处理）
    ///   - 从 reservedChannels 开始分配（避免与叶子循环冲突）
    ///   - 子循环之间可以复用 channels
    ///   - 返回当前循环及其子循环使用的最大 channel 号
    ///
    /// 返回值：
    ///   当前循环树使用的最大 channel 号 + 1
    int allocateNonLeafLoopChannels(scf::ForOp loop) {
      LDBG("========== allocateNonLeafLoopChannels ==========");
      LDBG("  Loop: " << loop);

      int nextChannel = reservedChannels;

      // 叶子循环已处理，直接返回
      if (isLeafLoop(loop)) {
        LDBG("  Leaf loop, skipping");
        LDBG("  Returning: " << nextChannel);
        LDBG("=================================================");
        return nextChannel;
      }

      // 收集所有子循环
      SmallVector<scf::ForOp> childLoops;
      for (Operation &op : loop.getBody()->getOperations()) {
        if (auto childLoop = dyn_cast<scf::ForOp>(&op)) {
          childLoops.push_back(childLoop);
        }
      }

      // 递归处理子循环，取最大的 channel 号
      for (auto childLoop : childLoops) {
        int childMax = allocateNonLeafLoopChannels(childLoop);
        if (childMax > nextChannel) {
          nextChannel = childMax;
        }
      }

      // 为当前循环的 groups 分配 channels
      auto it = loopToGroups.find(loop);
      if (it != loopToGroups.end()) {
        LDBG("  Non-leaf loop, allocating channels for " << it->second.size() << " groups");

        for (int groupId : it->second) {
          // 检查是否超出 channel 上限
          if (nextChannel >= 20) {
            loop->emitError("DMA channel allocation exceeded maximum limit (0-19)");
            signalPassFailure();
            LDBG("  ERROR: Channel limit exceeded");
            LDBG("  Returning: " << nextChannel);
            LDBG("=================================================");
            return nextChannel;
          }

          allGroups[groupId].channelBase = nextChannel;
          LDBG("    Group " << groupId << " -> channelBase: " << nextChannel);

          // 根据 position 类型确定需要的 channel 数量
          StringRef position = allGroups[groupId].position;
          int channelCount = (position == "fixed") ? 1 : 2;
          nextChannel += channelCount;
        }
      } else {
        LDBG("  Non-leaf loop with no groups");
      }

      LDBG("  Returning: " << nextChannel);
      LDBG("=================================================");
      return nextChannel;
    }

    //===------------------------------------------------------------------===//
    // 阶段 3：为循环外的 groups 分配 channels
    //===------------------------------------------------------------------===//

    /// 为循环外的 groups 分配 channels
    ///
    /// 策略：
    ///   - 查找 channelBase == -1 的 groups（未在 DFS 中分配）
    ///   - 从 reservedChannels 开始分配
    ///   - 通常只有循环外的 initial position DMA 会到这里
    void allocateOutOfLoopChannels() {
      int nextChannel = reservedChannels;

      for (auto &[groupId, info] : allGroups) {
        // 只处理未分配 channel 的 groups
        if (info.channelBase == -1) {
          // 检查是否超出 channel 上限
          if (nextChannel >= 20) {
            info.dmaOps[0]->emitError("DMA channel allocation exceeded maximum limit (0-19)");
            signalPassFailure();
            return;
          }

          info.channelBase = nextChannel;

          // 根据 position 类型确定需要的 channel 数量
          StringRef position = info.position;
          int channelCount = (position == "fixed") ? 1 : 2;
          nextChannel += channelCount;
        }
      }
    }

    //===------------------------------------------------------------------===//
    // 阶段 4：代码转换
    //===------------------------------------------------------------------===//

    /// 将所有 DMA 和 Wait 操作转换为优化版本
    ///
    /// 转换规则：
    ///   - mtdsp.dma  -> mtdsp.dma_opt (添加 channel 参数)
    ///   - mtdsp.wait -> mtdsp.wait_p2p
    void transformOperations() {
      OpBuilder builder(&getContext());

      for (auto &[groupId, info] : allGroups) {
        int channelBase = info.channelBase;

        if (channelBase == -1) {
          continue;  // 未分配 channel，跳过
        }

        // 转换所有 DMAOp
        for (mtdsp::DMAOp dmaOp : info.dmaOps) {
          builder.setInsertionPoint(dmaOp);

          // 创建 channel 值
          Value channel = createChannelValue(dmaOp, channelBase, info.ownerLoop, builder);
          if (!channel) {
            continue;  // 错误已在 createChannelValue 中报告
          }

          // 创建 DMAOptOp 替换 DMAOp
          auto dmaOptOp = builder.create<mtdsp::DMAOptOp>(
              dmaOp.getLoc(),
              dmaOp.getOperand(0),  // src
              dmaOp.getOperand(1),  // dst
              channel               // channel
          );

          dmaOp.getResult().replaceAllUsesWith(dmaOptOp.getResult());
          dmaOp.erase();
        }

        // 转换所有 WaitOp
        for (mtdsp::WaitOp waitOp : info.waitOps) {
          builder.setInsertionPoint(waitOp);

          // 创建 WaitP2POp 替换 WaitOp
          builder.create<mtdsp::WaitP2POp>(
              waitOp.getLoc(),
              waitOp.getChannel()
          );

          waitOp.erase();
        }
      }
    }

    /// 为 DMA 操作创建 channel 值
    ///
    /// 计算规则：
    ///   - fixed:   channel = channelBase (常量)
    ///   - initial: channel = channelBase + (lowerBound / step) % 2
    ///   - current: channel = channelBase + (IV / step) % 2
    ///   - next:    channel = channelBase + ((IV + step) / step) % 2
    Value createChannelValue(mtdsp::DMAOp dmaOp, int channelBase, scf::ForOp ownerLoop, OpBuilder &builder) {
      Location loc = dmaOp.getLoc();

      auto positionAttr = dmaOp->getAttrOfType<StringAttr>("position");
      StringRef position = positionAttr.getValue();

      // Fixed: 返回固定 channel 常量（无论在循环内外）
      if (position == "fixed") {
        auto channelConst = builder.create<arith::ConstantOp>(
            loc, builder.getIndexType(), builder.getIndexAttr(channelBase));
        return builder.create<arith::IndexCastOp>(
            loc, builder.getI32Type(), channelConst);
      }

      // 多缓冲 DMA 需要循环信息
      if (!ownerLoop) {
        dmaOp->emitError("Multi-buffered DMA missing owner loop");
        return Value();
      }

      Value step = ownerLoop.getStep();
      Value iv;

      // 根据 position 选择迭代变量
      if (position == "initial") {
        iv = ownerLoop.getLowerBound();  // 初始预取使用 lowerBound
      } else if (position == "current") {
        iv = ownerLoop.getInductionVar();  // 当前块使用 IV
      } else if (position == "next") {
        iv = builder.create<arith::AddIOp>(  // 下一块使用 IV + step
            loc, ownerLoop.getInductionVar(), step);
      } else {
        dmaOp->emitError("Unknown position attribute: ") << position;
        return Value();
      }

      // 计算 channel = channelBase + (iv / step) % 2
      auto c2 = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(2));
      auto div = builder.create<arith::DivSIOp>(loc, iv, step);
      auto rem = builder.create<arith::RemUIOp>(loc, div, c2);
      auto base = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(channelBase));
      auto channel = builder.create<arith::AddIOp>(loc, base, rem);
      return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), channel);
    }

    //===------------------------------------------------------------------===//
    // 阶段 5：插入优先级设置
    //===------------------------------------------------------------------===//

    /// 在函数入口插入 SetPrirOp 设置最内层循环的优先级
    void insertSetPrirOp(func::FuncOp funcOp) {
      if (prirMask == 0) {
        return;  // 没有需要设置优先级的 channel
      }

      OpBuilder builder(&getContext());
      builder.setInsertionPointToStart(&funcOp.getBody().front());

      auto prirConst = builder.create<arith::ConstantOp>(
          funcOp.getLoc(),
          builder.getI64Type(),
          builder.getIntegerAttr(builder.getI64Type(), prirMask));

      builder.create<mtdsp::SetPrirOp>(funcOp.getLoc(), prirConst);
    }

    //===------------------------------------------------------------------===//
    // Pass 入口
    //===------------------------------------------------------------------===//

    void runOnOperation() override {
      func::FuncOp funcOp = getOperation();

      // 重置状态
      allGroups.clear();
      loopToGroups.clear();
      reservedChannels = 0;
      prirMask = 0;

      // 阶段 0：收集信息
      collectGroupInfo(funcOp);

      // 阶段 1：为叶子循环分配 channels
      SmallVector<scf::ForOp> topLoops = getTopLevelLoops(funcOp);
      for (auto topLoop : topLoops) {
        allocateLeafLoopChannels(topLoop);
      }

      // 阶段 2：为非叶子循环分配 channels
      for (auto topLoop : topLoops) {
        allocateNonLeafLoopChannels(topLoop);
        // 注意：兄弟循环可以复用 channels
      }

      // Debug: 打印最终分配结果
      LDBG("========== Final Channel Allocation ==========");
      LDBG("Total groups: " << allGroups.size());
      for (auto &[groupId, info] : allGroups) {
        LDBG("  Group " << groupId << ":");
        LDBG("    Position: " << info.position);
        LDBG("    DMAOps count: " << info.dmaOps.size());
        LDBG("    WaitOps count: " << info.waitOps.size());
        LDBG("    Channel base: " << info.channelBase);
      }
      LDBG("=================================================");

      // 阶段 3：为循环外的 groups 分配 channels
      allocateOutOfLoopChannels();

      // 阶段 4：转换所有操作
      transformOperations();

      // 阶段 5：插入优先级设置
      insertSetPrirOp(funcOp);
    }
  };
}

std::unique_ptr<Pass> mlir::createOptimizeDMAPass(){
  return std::make_unique<OptimizeDMAPass>();
}
