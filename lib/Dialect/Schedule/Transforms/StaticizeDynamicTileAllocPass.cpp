//===- StaticizeDynamicTileAllocPass.cpp ---------------------------------===//
//
// Replace dynamic tile memref.alloc operations with a statically-sized max
// tile allocation plus a dynamic subview. This normalizes the memref IR to the
// same shape used by static tensor.empty buffers before multi-buffering.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include "Dialect/Schedule/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_STATICIZEDYNAMICTILEALLOC
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static std::optional<int64_t> getAffineMinConstantUpperBound(Value value) {
  auto minOp = value.getDefiningOp<affine::AffineMinOp>();
  if (!minOp)
    return std::nullopt;

  std::optional<int64_t> bound;
  for (AffineExpr expr : minOp.getAffineMap().getResults()) {
    auto constantExpr = dyn_cast<AffineConstantExpr>(expr);
    if (!constantExpr)
      continue;

    int64_t constant = constantExpr.getValue();
    if (!bound || constant < *bound)
      bound = constant;
  }

  return bound;
}

static FailureOr<SmallVector<int64_t>>
computeStaticShape(MemRefType type, ValueRange dynamicSizes) {
  SmallVector<int64_t> staticShape(type.getShape().begin(),
                                  type.getShape().end());

  unsigned dynamicIndex = 0;
  for (auto [dim, size] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(size))
      continue;

    if (dynamicIndex >= dynamicSizes.size())
      return failure();

    std::optional<int64_t> bound =
        getAffineMinConstantUpperBound(dynamicSizes[dynamicIndex++]);
    if (!bound || *bound <= 0)
      return failure();

    staticShape[dim] = *bound;
  }

  if (dynamicIndex != dynamicSizes.size())
    return failure();

  return staticShape;
}

static Value createDynamicSubview(OpBuilder &builder, memref::AllocOp oldAlloc,
                                  memref::AllocOp newAlloc) {
  MemRefType oldType = oldAlloc.getType();
  MemRefType newType = newAlloc.getType();
  Location loc = oldAlloc.getLoc();

  SmallVector<OpFoldResult> offsets(oldType.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(oldType.getRank(), builder.getIndexAttr(1));

  unsigned dynamicIndex = 0;
  for (auto [dim, size] : llvm::enumerate(oldType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      sizes.push_back(oldAlloc.getDynamicSizes()[dynamicIndex++]);
    } else {
      sizes.push_back(builder.getIndexAttr(size));
    }
  }

  auto resultType = cast<MemRefType>(
      memref::SubViewOp::inferRankReducedResultType(
          oldType.getShape(), newType, offsets, sizes, strides));

  auto subview = builder.create<memref::SubViewOp>(
      loc, resultType, newAlloc.getResult(), offsets, sizes, strides);
  return builder.create<memref::CastOp>(loc, oldType, subview.getResult());
}

struct StaticizeDynamicTileAllocPass
    : public impl::StaticizeDynamicTileAllocBase<
          StaticizeDynamicTileAllocPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<memref::AllocOp> allocs;
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (!allocOp.getDynamicSizes().empty())
        allocs.push_back(allocOp);
    });

    for (memref::AllocOp allocOp : allocs) {
      MemRefType oldType = allocOp.getType();
      FailureOr<SmallVector<int64_t>> maybeStaticShape =
          computeStaticShape(oldType, allocOp.getDynamicSizes());
      if (failed(maybeStaticShape))
        continue;

      MemRefType newType = MemRefType::get(
          *maybeStaticShape, oldType.getElementType(), oldType.getLayout(),
          oldType.getMemorySpace());

      OpBuilder builder(allocOp);
      auto newAlloc = builder.create<memref::AllocOp>(
          allocOp.getLoc(), newType, /*dynamicSizes=*/ValueRange{},
          /*symbolOperands=*/ValueRange{}, allocOp.getAlignmentAttr());

      Value replacement = createDynamicSubview(builder, allocOp, newAlloc);

      for (OpOperand &use :
           llvm::make_early_inc_range(allocOp->getUses())) {
        if (auto deallocOp = dyn_cast<memref::DeallocOp>(use.getOwner())) {
          use.set(newAlloc.getResult());
          continue;
        }
        use.set(replacement);
      }

      allocOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createStaticizeDynamicTileAllocPass() {
  return std::make_unique<StaticizeDynamicTileAllocPass>();
}
