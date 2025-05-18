#ifndef LAYOUTTRANS_BASE
#define LAYOUTTRANS_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// LayoutTrans Dialect
//===----------------------------------------------------------------------===//

def LayoutTrans_Dialect : Dialect {
  let name = "layout_trans";
  let cppNamespace = "::mlir::layout_trans";
}

#endif // LAYOUTTRANS_BASE