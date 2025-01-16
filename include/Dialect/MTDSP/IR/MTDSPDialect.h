#ifndef MLIR_DIALECT_MTDSP_IR_MTDSPDIALECT_H
#define MLIR_DIALECT_MTDSP_IR_MTDSPDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

#include "Dialect/MTDSP/IR/MTDSPEnums.h.inc"

#include "Dialect/MTDSP/IR/MTDSPDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MTDSP/IR/MTDSPAttributes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/MTDSP/IR/MTDSPOps.h.inc"

#endif // MLIR_DIALECT_MTDSP_IR_MTDSPDIALECT_H