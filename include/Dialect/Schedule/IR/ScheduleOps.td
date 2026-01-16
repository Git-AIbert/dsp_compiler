#ifndef SCHEDULE_OPS
#define SCHEDULE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

include "Dialect/Schedule/IR/ScheduleBase.td"

//===----------------------------------------------------------------------===//
// Schedule Operations
//===----------------------------------------------------------------------===//

class Schedule_Op<string mnemonic, list<Trait> traits = []> :
    Op<Schedule_Dialect, mnemonic, traits>;


#endif // SCHEDULE_OPS