#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

void DMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // DMA reads from source
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  // DMA writes to destination
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// DMAOptOp
//===----------------------------------------------------------------------===//

struct DMAOptOpCanonicalizePattern : public OpRewritePattern<DMAOptOp> {
  using OpRewritePattern<DMAOptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DMAOptOp op,
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
    auto newOp = rewriter.create<mtdsp::DMAOptOp>(op.getLoc(), src, dst, op.getChannel());
    
    // 替换原操作
    rewriter.replaceOp(op, newOp.getResult());
    
    return success();
  }
};

void DMAOptOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context){
  results.add<DMAOptOpCanonicalizePattern>(context);
}

void DMAOptOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // DMA reads from source
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  // DMA writes to destination
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

void WaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Wait operation has memory effects (synchronization)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// WaitP2POp
//===----------------------------------------------------------------------===//

void WaitP2POp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Wait P2P operation has memory effects (synchronization)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// SetPrirOp
//===----------------------------------------------------------------------===//

void SetPrirOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // SetPrir configures DMA channels (has side effects)
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// GroupBarrierOp
//===----------------------------------------------------------------------===//

void GroupBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Barrier has memory effects (synchronization)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Alloc allocates memory
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       llvm::cast<OpResult>(getMemref()),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

void DeallocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Dealloc frees memory
  effects.emplace_back(MemoryEffects::Free::get(), &getMemrefMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MatmulR6C96Op
//===----------------------------------------------------------------------===//

void MatmulR6C96Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Matmul reads from lhs and rhs
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  // Matmul writes to dst
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MatmulR6C128Op
//===----------------------------------------------------------------------===//

void MatmulR6C128Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Matmul reads from lhs and rhs
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  // Matmul writes to dst
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MatmulR12C128Op
//===----------------------------------------------------------------------===//

void MatmulR12C128Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Matmul reads from lhs and rhs
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  // Matmul writes to dst
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MatmulMicroKernelOp
//===----------------------------------------------------------------------===//

void MatmulMicroKernelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Matmul reads from lhs and rhs
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  // Matmul writes to dst
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AddMicroKernelOp
//===----------------------------------------------------------------------===//

void AddMicroKernelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Add reads from lhs and rhs
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  // Add writes to dst
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

#include "Dialect/MTDSP/IR/MTDSPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MTDSP/IR/MTDSPAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/MTDSP/IR/MTDSPOps.cpp.inc"