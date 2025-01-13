#ifndef MLIR_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H
#define MLIR_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

#include "Dialect/Schedule/IR/ScheduleDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Schedule/IR/ScheduleOps.h.inc"

#endif // MLIR_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H