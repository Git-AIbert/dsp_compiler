add_mlir_conversion_library(ConvertToMTDSP
  ConvertToMTDSPPass.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  MTDSPDialect
  )