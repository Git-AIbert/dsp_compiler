#ifndef MTIR_CONVERSION_PASSES
#define MTIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LinalgToMTDSP
//===----------------------------------------------------------------------===//

def ConvertLinalgToMTDSP : Pass<"convert-linalg-to-mtdsp", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "MTDSP dialect";
  let constructor = "mtir::createConvertLinalgToMTDSPPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MemRefToMTDSP
//===----------------------------------------------------------------------===//

def ConvertMemRefToMTDSP : Pass<"convert-memref-to-mtdsp", "ModuleOp"> {
  let summary = "Convert the operations from the memref dialect into the "
                "MTDSP dialect";
  let constructor = "mtir::createConvertMemRefToMTDSPPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MTDSPToLLVM
//===----------------------------------------------------------------------===//

def ConvertMTDSPToLLVM : Pass<"convert-mtdsp-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the MTDSP dialect into the "
                "llvm dialect";
  let constructor = "mtir::createConvertMTDSPToLLVMPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

#endif // MTIR_CONVERSION_PASSES