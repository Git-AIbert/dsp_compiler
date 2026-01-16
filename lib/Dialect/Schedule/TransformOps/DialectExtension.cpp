#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

#include "Dialect/Schedule/IR/ScheduleDialect.h"
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"
#include "Dialect/MTDSP/IR/MTDSPDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class ScheduleTransformDialectExtension
    : public transform::TransformDialectExtension<
          ScheduleTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<schedule::ScheduleDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<func::FuncDialect>();
    declareDependentDialect<mtdsp::MTDSPDialect>();

    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<index::IndexDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::schedule::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ScheduleTransformDialectExtension>();
}