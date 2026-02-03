#ifndef CASCADE_CONVERSION_PASSES
#define CASCADE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LinalgToMTDSP
//===----------------------------------------------------------------------===//

def ConvertLinalgToMTDSP : Pass<"convert-linalg-to-mtdsp", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "MTDSP dialect";
  let constructor = "cascade::createConvertLinalgToMTDSPPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MemRefToMTDSP
//===----------------------------------------------------------------------===//

def ConvertMemRefToMTDSP : Pass<"convert-memref-to-mtdsp", "ModuleOp"> {
  let summary = "Convert the operations from the memref dialect into the "
                "MTDSP dialect";
  let constructor = "cascade::createConvertMemRefToMTDSPPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MTDSPToLLVM
//===----------------------------------------------------------------------===//

def ConvertMTDSPToLLVM : Pass<"convert-mtdsp-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the MTDSP dialect into the "
                "llvm dialect";
  let constructor = "cascade::createConvertMTDSPToLLVMPass()";
  let dependentDialects = ["mtdsp::MTDSPDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

#endif // CASCADE_CONVERSION_PASSES