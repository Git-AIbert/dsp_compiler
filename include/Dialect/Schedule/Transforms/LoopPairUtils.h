//===- LoopPairUtils.h - Utilities for processing split loop pairs --------===//
//
// Common utilities for finding and processing split loop pairs across
// multiple transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef MTDSP_DIALECT_SCHEDULE_TRANSFORMS_LOOPPAIRUTILS_H
#define MTDSP_DIALECT_SCHEDULE_TRANSFORMS_LOOPPAIRUTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace schedule {

/// Generic function to find and process split loop pairs recursively
///
/// Template parameters:
///   DebugLogger: A callable that takes (const char* message) for debug output
///
/// Function parameters:
///   root: The root operation to start searching from
///   isSplitPairFn: Function to check if two loops form a split pair
///   processPairFn: Function to process a found split pair
///   debugLog: Logger for debug output
template<typename DebugLogger>
void findAndProcessSplitLoopsImpl(
    Operation *root,
    llvm::function_ref<bool(scf::ForOp, scf::ForOp)> isSplitPairFn,
    llvm::function_ref<void(scf::ForOp, scf::ForOp)> processPairFn,
    DebugLogger debugLog) {

  debugLog("");
  debugLog(">>> Entering findAndProcessSplitLoops");

  // Collect all for loops by directly iterating through regions and blocks
  SmallVector<scf::ForOp> forOps;
  for (Region &region : root->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          forOps.push_back(forOp);
        }
      }
    }
  }

  // Log the number of loops found
  {
    std::string msg = "Found " + std::to_string(forOps.size()) + " loops at current level";
    debugLog(msg.c_str());
  }

  // 记录已经作为配对处理过的循环
  DenseSet<scf::ForOp> processedInPair;

  for (size_t i = 0; i < forOps.size(); ++i) {
    scf::ForOp loop1 = forOps[i];

    // 跳过已经作为配对处理过的循环
    if (processedInPair.contains(loop1)) continue;

    bool foundPair = false;
    if(i + 1 < forOps.size()){
      scf::ForOp loop2 = forOps[i+1];
      
      if (isSplitPairFn(loop1, loop2)) {
        // 先递归处理两个循环内部
        findAndProcessSplitLoopsImpl(loop1, isSplitPairFn, processPairFn, debugLog);
        findAndProcessSplitLoopsImpl(loop2, isSplitPairFn, processPairFn, debugLog);

        // 然后处理这一对循环
        processPairFn(loop1, loop2);

        // 标记这两个循环已处理
        processedInPair.insert(loop1);
        processedInPair.insert(loop2);

        foundPair = true;
      }
    }

    if (!foundPair) {
      findAndProcessSplitLoopsImpl(loop1, isSplitPairFn, processPairFn, debugLog);
    }
  }

  debugLog("<<< Exiting findAndProcessSplitLoops");
  debugLog("");
}

}  // namespace schedule
}  // namespace mlir

#endif  // MTDSP_DIALECT_SCHEDULE_TRANSFORMS_LOOPPAIRUTILS_H