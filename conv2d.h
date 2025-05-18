#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;

func::FuncOp createConv2DFunction(OpBuilder &builder, ModuleOp module);

LogicalResult createAndApplyTransform2(ModuleOp module);

