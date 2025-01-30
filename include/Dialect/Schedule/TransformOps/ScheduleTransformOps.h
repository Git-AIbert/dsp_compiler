#ifndef MLIR_DIALECT_SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS_H
#define MLIR_DIALECT_SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// Schedule Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h.inc"

namespace mlir {
namespace schedule {
void registerTransformDialectExtension(DialectRegistry &registry);

tensor::EmptyOp createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                           SmallPtrSet<Operation *, 4> &newOps,
                                           Location loc, Attribute memorySpace);

linalg::CopyOp createCacheRead(OpBuilder &rewriter, Value operand,
                               Location loc, Attribute memorySpace);

FailureOr<linalg::CopyOp> createCacheWrite(OpBuilder &rewriter, OpResult result,
                                           Value cacheWriteTo, Attribute memorySpace);
} // namespace schedule
} // namespace mlir



#endif // MLIR_DIALECT_SCHEDULE_TRANSFORMOPS_SCHEDULETRANSFORMOPS_H