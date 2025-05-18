#ifndef MLIR_DIALECT_LAYOUTTRANS_IR_LAYOUTTRANSDIALECT_H
#define MLIR_DIALECT_LAYOUTTRANS_IR_LAYOUTTRANSDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "Dialect/LayoutTrans/IR/LayoutTransDialect.h.inc"

namespace mlir {
namespace layout_trans {
// 匹配实现中使用的命名空间
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace layouttrans
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/LayoutTrans/IR/LayoutTransOps.h.inc"

#endif // MLIR_DIALECT_LAYOUTTRANS_IR_LAYOUTTRANSDIALECT_H