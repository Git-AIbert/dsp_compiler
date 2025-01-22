#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"

using namespace mlir;
using namespace mlir::mtdsp;

#include "Dialect/MTDSP/IR/MTDSPDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MTDSPDialect
//===----------------------------------------------------------------------===//

// 实现初始化函数
void MTDSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/MTDSP/IR/MTDSPOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/MTDSP/IR/MTDSPAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MTDSP Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GroupSizeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatmulR6C96Op
//===----------------------------------------------------------------------===//

#include "Dialect/MTDSP/IR/MTDSPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MTDSP/IR/MTDSPAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/MTDSP/IR/MTDSPOps.cpp.inc"