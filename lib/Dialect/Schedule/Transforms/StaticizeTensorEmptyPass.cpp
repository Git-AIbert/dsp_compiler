#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Casting.h"

#include "Dialect/Schedule/Transforms/Passes.h"

using namespace mlir;

namespace {
// 计算新的静态形状
SmallVector<int64_t> computeNewStaticShapes(ArrayRef<int64_t> staticShapes,
                                          ValueRange dynamicSizes) {
  SmallVector<int64_t> newStaticShapes;
  int dynamicDimIndex = 0;
  
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] != ShapedType::kDynamic) {
      newStaticShapes.push_back(staticShapes[i]);
      continue;
    }

    // 获取动态维度的定义操作
    Value dynamicSize = dynamicSizes[dynamicDimIndex++];
    Operation *defOp = dynamicSize.getDefiningOp();

    // 检查是否由affine.min定义
    auto affineMinOp = dyn_cast<affine::AffineMinOp>(defOp);
    assert(affineMinOp && "Dynamic size must be defined by affine.min");

    // 从affine.min中获取静态值
    AffineMap map = affineMinOp.getAffineMap();
    bool foundConstant = false;
    for (auto expr : map.getResults()) {
      if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
        newStaticShapes.push_back(constExpr.getValue());
        foundConstant = true;
        break;
      }
    }
    assert(foundConstant && "affine.min must contain at least one constant expression");
  }
  
  return newStaticShapes;
}

// 创建新的静态EmptyOp
tensor::EmptyOp createNewEmptyOp(OpBuilder &builder, 
                                ArrayRef<int64_t> newStaticShapes,
                                tensor::EmptyOp originalOp) {
  auto newEmptyOp = builder.create<tensor::EmptyOp>(
    originalOp.getLoc(),
    newStaticShapes, 
    originalOp.getType().getElementType()
  );

  // 复制所有属性
  for (auto namedAttr : originalOp->getAttrs()) {
    newEmptyOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  return newEmptyOp;
}

// 创建ExtractSliceOp
tensor::ExtractSliceOp createExtractSliceOp(OpBuilder &builder,
                                           Value source,
                                           tensor::EmptyOp originalOp) {
  // 创建offsets和strides
  SmallVector<OpFoldResult> offsets(originalOp.getType().getRank(), 
    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(originalOp.getType().getRank(), 
    builder.getIndexAttr(1));

  // 设置sizes
  SmallVector<OpFoldResult> sizes;
  ArrayRef<int64_t> staticShapes = originalOp.getType().getShape();
  ValueRange dynamicSizes = originalOp.getDynamicSizes();
  int dynamicDimIndex = 0;
  
  for (int i = 0; i < originalOp.getType().getRank(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic)
      sizes.push_back(dynamicSizes[dynamicDimIndex++]);
    else
      sizes.push_back(builder.getIndexAttr(staticShapes[i]));
  }

  return builder.create<tensor::ExtractSliceOp>(
    originalOp.getLoc(),
    originalOp.getType(),
    source,
    offsets,
    sizes,
    strides
  );
}

struct StaticizeTensorEmptyPass 
    : public PassWrapper<StaticizeTensorEmptyPass, OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    funcOp.walk([&](tensor::EmptyOp emptyOp) {
      // 获取动态维度的值
      ValueRange dynamicSizes = emptyOp.getDynamicSizes();
      if (dynamicSizes.empty())
        return;

      // 获取tensor形状信息
      ArrayRef<int64_t> staticShapes = emptyOp.getType().getShape();
      
      // 计算新的静态形状
      auto newStaticShapes = computeNewStaticShapes(staticShapes, dynamicSizes);

      // 创建新的静态EmptyOp
      builder.setInsertionPoint(emptyOp);
      auto newEmptyOp = createNewEmptyOp(builder, 
                                        newStaticShapes,
                                        emptyOp);

      // 创建extract_slice操作
      auto extractSliceOp = createExtractSliceOp(builder,
                                                newEmptyOp.getResult(),
                                                emptyOp);

      // 替换原有使用并删除原EmptyOp
      emptyOp.replaceAllUsesWith(extractSliceOp.getResult());
      emptyOp.erase();
    });
  }
};
}

std::unique_ptr<Pass> mlir::createStaticizeTensorEmptyPass() {
  return std::make_unique<StaticizeTensorEmptyPass>();
}