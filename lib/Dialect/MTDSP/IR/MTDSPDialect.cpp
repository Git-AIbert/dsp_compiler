#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"

using namespace mlir;
using namespace mlir::mtdsp;

#include "Dialect/MTDSP/IR/MTDSPDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MTDSPDialect
//===----------------------------------------------------------------------===//

// 实现初始化函数
void MTDSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/MTDSP/IR/MTDSPOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/MTDSP/IR/MTDSPAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MTDSP Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GroupSizeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//

struct DMAOpCanonicalizePattern : public OpRewritePattern<DMAOp> {
  using OpRewritePattern<DMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DMAOp op,
                               PatternRewriter &rewriter) const override {
    // 获取原始操作数
    Value src = op.getSrc();
    Value dst = op.getDst();

    // 检查是否需要类型转换
    auto srcType = src.getType().cast<MemRefType>();
    auto dstType = dst.getType().cast<MemRefType>();

    // 如果源或目标是cast操作的结果
    if (auto srcCast = src.getDefiningOp<memref::CastOp>())
      src = srcCast.getSource();
    if (auto dstCast = dst.getDefiningOp<memref::CastOp>())
      dst = dstCast.getSource();
    
    // 如果操作数没有改变，返回failure
    if (src == op.getSrc() && dst == op.getDst())
      return failure();
    
    // 创建新的DMA操作
    auto newOp = rewriter.create<mtdsp::DMAOp>(op.getLoc(), src, dst);
    
    // 替换原操作
    rewriter.replaceOp(op, newOp.getResult());
    
    return success();
  }
};

void DMAOp::getCanonicalizationPatterns(RewritePatternSet& results, 
                                        MLIRContext* context){
  results.add<DMAOpCanonicalizePattern>(context);
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatmulR6C96Op
//===----------------------------------------------------------------------===//

#include "Dialect/MTDSP/IR/MTDSPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MTDSP/IR/MTDSPAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/MTDSP/IR/MTDSPOps.cpp.inc"