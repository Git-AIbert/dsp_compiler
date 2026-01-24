#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <queue>

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"
#include "Dialect/Schedule/Transforms/LoopPairUtils.h"

#define DEBUG_TYPE "deduplicate-multi-buffer"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_DEDUPLICATEMULTIBUFFER
#include "Dialect/Schedule/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

// 内存层次结构信息（构建阶段）
struct MemoryHierarchy {
  DenseMap<Value, DenseSet<memref::AllocOp>> children;
  DenseMap<memref::AllocOp, scf::ForOp> loop;
};

// 合并映射（去重结果）
using MergeMap = DenseMap<memref::AllocOp, memref::AllocOp>;

struct DeduplicateMultiBufferPass
    : public impl::DeduplicateMultiBufferBase<DeduplicateMultiBufferPass> {

private:

  // ============================================================================
  // 辅助函数声明
  // ============================================================================

  /// 获取 Value 的地址空间等级 (0=Global, 1=GSM, 2=SM, 3=AM)
  unsigned getAddressSpace(Value v);

  /// 追踪 Value 到根源（函数参数或 alloc）
  Value traceToRoot(Value v);

  // ============================================================================
  // 核心函数
  // ============================================================================

  /// 检查两个 sibling alloc 是否可以安全合并
  bool canMergeAllocs(const MemoryHierarchy &hierarchy, memref::AllocOp alloc1, memref::AllocOp alloc2);

  /// 处理单个循环中的DMA操作，构建内存层次结构
  void analyzeDMAInLoop(scf::ForOp loop, MemoryHierarchy &hierarchy);

  /// BFS去重，返回合并映射
  MergeMap deduplicateBFS(MemoryHierarchy &hierarchy);

  /// 应用合并映射
  void applyMerge(const MergeMap &mergeMap);

  /// 打印内存层次结构
  void dumpMemoryHierarchy(const MemoryHierarchy &hierarchy, const std::string &stage);

  /// 打印合并映射
  void dumpMergeMap(const MergeMap &mergeMap, const std::string &stage);

  /// 完整处理一对拆分循环：构建 → 去重 → 合并
  void processSplitLoopPair(scf::ForOp loop1, scf::ForOp loop2);

  /// 从内到外递归查找并处理拆分的循环对
  void findAndProcessSplitLoops(Operation *root);

public:
  void runOnOperation() override;
};

// ============================================================================
// 辅助函数实现
// ============================================================================

unsigned DeduplicateMultiBufferPass::getAddressSpace(Value v) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType) return 0;

  auto addrSpace = memrefType.getMemorySpace();
  if (!addrSpace) return 0;  // Global memory

  // MTDSP address space: 0=Global, 1=Workgroup(gsm), 2=Scalar(sm), 3=Vector(am)
  // 尝试解析为 MTDSP AddressSpaceAttr (枚举属性)
  if (auto enumAttr = dyn_cast<mlir::mtdsp::AddressSpaceAttr>(addrSpace)) {
    return static_cast<unsigned>(enumAttr.getValue());
  }

  // 兜底：尝试解析为 IntegerAttr
  if (auto intAttr = dyn_cast<IntegerAttr>(addrSpace)) {
    return intAttr.getInt();
  }

  return 0;
}

Value DeduplicateMultiBufferPass::traceToRoot(Value v) {
  while (v) {
    // 如果是函数参数，这就是根节点
    if (isa<BlockArgument>(v)) {
      return v;
    }

    auto defOp = v.getDefiningOp();
    if (!defOp) return v;

    // 如果是 alloc，这就是根节点
    if (isa<memref::AllocOp>(defOp)) {
      return v;
    }

    // 继续追踪 subview、cast 等操作
    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      v = subview.getSource();
      continue;
    }

    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      v = cast.getSource();
      continue;
    }

    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      v = reinterpret.getSource();
      continue;
    }

    // 无法继续追踪
    return v;
  }
  return v;
}


// ============================================================================
// 函数实现
// ============================================================================

void DeduplicateMultiBufferPass::analyzeDMAInLoop(scf::ForOp loop,
                                                   MemoryHierarchy &hierarchy) {
  LDBG("");
  LDBG(">>> Entering analyzeDMAInLoop");

  loop.walk([&](Operation *op) {
    // 检查是否为 DMA 操作（简单检查操作名）
    if (!op->getName().getStringRef().contains("dma")) {
      return;
    }

    LDBG("  Found DMA: " << op->getName() << " @ " << op->getLoc());

    // 跳过带有 prolog_dma 属性的 DMA
    if (op->hasAttr("prolog_dma")) {
      LDBG("    -> Skipped (has prolog_dma)");
      return;
    }

    // 获取源和目标操作数
    if (op->getNumOperands() < 2) return;
    Value src = op->getOperand(0);
    Value dst = op->getOperand(1);

    // 追踪到根节点
    Value srcRoot = traceToRoot(src);
    Value dstRoot = traceToRoot(dst);

    // 获取地址空间
    unsigned srcSpace = getAddressSpace(srcRoot);
    unsigned dstSpace = getAddressSpace(dstRoot);

    LDBG("    SrcRoot: " << srcRoot << " [Space=" << srcSpace << "]");
    LDBG("    DstRoot: " << dstRoot << " [Space=" << dstSpace << "]");

    // 地址空间：0=Global, 1=GSM, 2=SM, 3=AM
    // 从低层级到高层级：Global -> GSM -> SM -> AM
    // 地址空间小的是父节点，大的是子节点
    Value parent = (srcSpace < dstSpace) ? srcRoot : dstRoot;
    Value child = (srcSpace < dstSpace) ? dstRoot : srcRoot;

    LDBG("    Parent->Child: " << parent << " -> " << child);

    // 如果子节点是 alloc，建立父子关系
    if (auto childAlloc = child.getDefiningOp<memref::AllocOp>()) {
      hierarchy.children[parent].insert(childAlloc);

      // 记录 allocInfo：使用该 alloc 的最内层循环
      if (!hierarchy.loop.count(childAlloc)) {
        hierarchy.loop[childAlloc] = loop;
      }
    }
  });

  LDBG("<<< Exiting analyzeDMAInLoop");
  LDBG("");
}

bool DeduplicateMultiBufferPass::canMergeAllocs(const MemoryHierarchy &hierarchy,
                                                 memref::AllocOp alloc1,
                                                 memref::AllocOp alloc2) {
  // 检查类型是否完全相同
  if (alloc1.getType() != alloc2.getType()) {
    LDBG("Type mismatch");
    return false;
  }

  // 检查是否都有 loop
  auto it1 = hierarchy.loop.find(alloc1);
  auto it2 = hierarchy.loop.find(alloc2);
  if (it1 == hierarchy.loop.end() || it2 == hierarchy.loop.end()) {
    LDBG("Missing loop");
    return false;
  }

  // 检查是否位于不同的归约循环（兄弟循环）
  scf::ForOp loop1 = it1->second;
  scf::ForOp loop2 = it2->second;

  if (loop1 == loop2) {
    LDBG("Same loop - cannot merge");
    return false;  // 在同一个循环中，不能融合
  }

  LDBG("✓ Can merge!");
  return true;
}

MergeMap DeduplicateMultiBufferPass::deduplicateBFS(MemoryHierarchy &hierarchy) {
  LDBG("");
  LDBG(">>> Entering deduplicateBFS");

  MergeMap mergeMap;

  // BFS 队列，存储待处理的父节点
  std::queue<Value> queue;

  // 找到所有根节点（没有父节点的节点）
  DenseSet<Value> allParents;
  DenseSet<Value> allChildren;

  for (auto &[parent, childSet] : hierarchy.children) {
    allParents.insert(parent);
    for (auto child : childSet) {
      allChildren.insert(child.getResult());
    }
  }

  // 根节点 = 出现在 parent 中但不出现在 children 中的节点
  LDBG("Finding root nodes:");
  for (Value parent : allParents) {
    if (!allChildren.contains(parent)) {
      LDBG("  Root: " << parent.getType());
      queue.push(parent);
    }
  }

  // BFS 遍历
  int iteration = 0;
  while (!queue.empty()) {
    Value parent = queue.front();
    queue.pop();

    LDBG("");
    LDBG("[BFS Iteration " << iteration++ << "]");
    LDBG("  Parent: " << parent.getType());

    // 获取当前父节点的所有子节点
    if (!hierarchy.children.count(parent)) {
      LDBG("  No children, skipping");
      continue;
    }

    auto &childSet = hierarchy.children[parent];
    SmallVector<memref::AllocOp> childList(childSet.begin(), childSet.end());

    LDBG("  Children count: " << childList.size());

    // 尝试合并子节点：遍历所有子节点对
    for (size_t i = 0; i < childList.size(); ++i) {
      for (size_t j = i + 1; j < childList.size(); ++j) {
        memref::AllocOp child1 = childList[i];
        memref::AllocOp child2 = childList[j];

        LDBG("  Checking pair [" << i << "," << j << "]:");
        LDBG("    Child1: " << child1.getType());
        LDBG("    Child2: " << child2.getType());

        // 检查是否可以融合
        if (canMergeAllocs(hierarchy, child1, child2)) {
          LDBG("    ✓✓✓ MERGING: child2 -> child1");
          // 记录融合关系：child2 -> child1
          mergeMap[child2] = child1;

          // 将 child2 的子节点移动到 child1
          if (hierarchy.children.count(child2.getResult())) {
            auto &child2Children = hierarchy.children[child2.getResult()];
            LDBG("      Moving " << child2Children.size() << " grandchildren");
            for (auto grandChild : child2Children) {
              hierarchy.children[child1.getResult()].insert(grandChild);
            }
            hierarchy.children.erase(child2.getResult());
          }

          // 从 childSet 中删除 child2
          childSet.erase(child2);

          // 更新 childList（删除 child2）
          childList.erase(childList.begin() + j);
          --j;  // 调整索引
        }
      }
    }

    // 将剩余的子节点加入队列
    LDBG("  Adding " << childSet.size() << " children to queue");
    for (auto child : childSet) {
      queue.push(child.getResult());
    }
  }

  LDBG("Total merges: " << mergeMap.size());
  LDBG("<<< Exiting deduplicateBFS");
  LDBG("");

  return mergeMap;
}

void DeduplicateMultiBufferPass::applyMerge(const MergeMap &mergeMap) {
  LDBG(">>> Entering applyMerge");
  LDBG("");
  LDBG("Merging " << mergeMap.size() << " allocations");

  // 应用合并映射
  for (const auto &entry : mergeMap) {
    memref::AllocOp oldAlloc = entry.first;
    memref::AllocOp newAlloc = entry.second;

    LDBG("  Replacing " << oldAlloc << " with " << newAlloc);
    // 替换所有使用
    oldAlloc.replaceAllUsesWith(newAlloc.getResult());

    // 删除旧的 alloc
    oldAlloc.erase();
  }

  LDBG("<<< Exiting applyMerge");
  LDBG("");
}

void DeduplicateMultiBufferPass::dumpMemoryHierarchy(const MemoryHierarchy &hierarchy,
                                                      const std::string &stage) {
  LDBG("");
  LDBG("========================================");
  LDBG("Memory Hierarchy Dump [" << stage << "]");
  LDBG("========================================");
  LDBG("");

  // 1. 打印 children 关系
  LDBG("--- Children Relationships ---");
  for (auto &[parent, childSet] : hierarchy.children) {
    LLVM_DEBUG({
      DBGS() << "Parent: ";
      // 打印父节点信息
      if (auto blockArg = dyn_cast<BlockArgument>(parent)) {
        llvm::dbgs() << "BlockArg#" << blockArg.getArgNumber()
                     << " (type: " << parent.getType() << ")";
      } else if (auto allocOp = parent.getDefiningOp<memref::AllocOp>()) {
        llvm::dbgs() << "Alloc@" << allocOp.getLoc()
                     << " (type: " << allocOp.getType() << ")";
      } else {
        llvm::dbgs() << "Value (type: " << parent.getType() << ")";
      }
      llvm::dbgs() << " [AddressSpace=" << getAddressSpace(parent) << "]\n";
    });

    // 打印子节点信息
    LDBG("  Children (" << childSet.size() << "):");
    for (auto child : childSet) {
      LDBG("    - Alloc@" << child.getLoc()
                   << " (type: " << child.getType() << ")"
                   << " [AddressSpace=" << getAddressSpace(child.getResult()) << "]");
    }
    LDBG("");
  }

  // 2. 打印 loop
  LDBG("--- loop ---");
  for (const auto &entry : hierarchy.loop) {
    memref::AllocOp alloc = entry.first;
    scf::ForOp loop = entry.second;
    LDBG("Alloc@" << alloc.getLoc() << ":");
    LLVM_DEBUG({
      DBGS() << "  Loop: ";
      if (loop) {
        llvm::dbgs() << "ForOp@" << loop.getLoc()
                     << " [" << loop.getLowerBound()
                     << " to " << loop.getUpperBound()
                     << " step " << loop.getStep() << "]\n";
      } else {
        llvm::dbgs() << "nullptr\n";
      }
    });
    LDBG("");
  }

  LDBG("");
  LDBG("========================================");
  LDBG("");
}

void DeduplicateMultiBufferPass::dumpMergeMap(const MergeMap &mergeMap,
                                               const std::string &stage) {
  LDBG("");
  LDBG("========================================");
  LDBG("Merge Map Dump [" << stage << "]");
  LDBG("========================================");
  LDBG("");

  if (mergeMap.empty()) {
    LDBG("(empty)");
  } else {
    for (const auto &entry : mergeMap) {
      memref::AllocOp oldAlloc = entry.first;
      memref::AllocOp newAlloc = entry.second;
      LDBG("Merge: Alloc@" << oldAlloc.getLoc()
                   << " -> Alloc@" << newAlloc.getLoc());
    }
  }

  LDBG("");
  LDBG("========================================");
  LDBG("");
}

void DeduplicateMultiBufferPass::processSplitLoopPair(scf::ForOp loop1,
                                                               scf::ForOp loop2) {
  LDBG("");
  LDBG(">>> processSplitLoopPair");

  // 步骤1: 构建内存层次结构
  MemoryHierarchy hierarchy;
  analyzeDMAInLoop(loop1, hierarchy);
  analyzeDMAInLoop(loop2, hierarchy);
  LLVM_DEBUG(dumpMemoryHierarchy(hierarchy, "After Build"));

  // 步骤2: BFS去重，得到合并映射
  MergeMap mergeMap = deduplicateBFS(hierarchy);
  LLVM_DEBUG(dumpMergeMap(mergeMap, "After BFS"));

  // 步骤3: 应用合并
  applyMerge(mergeMap);

  LDBG("<<< Exiting processSplitLoopPair");
  LDBG("");
}

void DeduplicateMultiBufferPass::findAndProcessSplitLoops(Operation *root) {
  LDBG("");

  // Use the common implementation with custom debug logger
  auto debugLogger = [](const char* msg) {
    LDBG(msg);
  };

  auto isSplitPairFn = [](scf::ForOp loop1, scf::ForOp loop2) {
    return loop1.getUpperBound() == loop2.getLowerBound() &&
           loop1.getStep() == loop2.getStep();
  };

  auto processPairFn = [this](scf::ForOp loop1, scf::ForOp loop2) {
    processSplitLoopPair(loop1, loop2);
  };

  schedule::findAndProcessSplitLoopsImpl(root, isSplitPairFn, processPairFn, debugLogger);

  LDBG("");
}

// ============================================================================
// Pass 入口
// ============================================================================

void DeduplicateMultiBufferPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  LDBG("");
  LDBG("===============================================");
  LDBG("  DeduplicateMultiBuffer Pass Started");
  LDBG("===============================================");

  // 从内到外递归查找并处理拆分的循环对
  // 每对循环会独立进行：构建 → 去重 → 合并
  findAndProcessSplitLoops(funcOp);

  LDBG("===============================================");
  LDBG("  DeduplicateMultiBuffer Pass Completed");
  LDBG("===============================================");
  LDBG("");
}

}  // namespace

// ============================================================================
// Pass 注册
// ============================================================================

std::unique_ptr<Pass> mlir::createDeduplicateMultiBufferPass() {
  return std::make_unique<DeduplicateMultiBufferPass>();
}
