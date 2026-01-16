#ifndef SCHEDULE_BASE
#define SCHEDULE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Schedule Dialect
//===----------------------------------------------------------------------===//

def Schedule_Dialect : Dialect {
  let name = "schedule";
  let cppNamespace = "::mlir::schedule";
}

#endif // SCHEDULE_BASE