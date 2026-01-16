#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Schedule/IR/ScheduleDialect.h"

using namespace mlir;
using namespace mlir::schedule;

#include "Dialect/Schedule/IR/ScheduleDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ScheduleDialect
//===----------------------------------------------------------------------===//

// 实现初始化函数
void ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Schedule/IR/ScheduleOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Schedule Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Schedule/IR/ScheduleOps.cpp.inc"