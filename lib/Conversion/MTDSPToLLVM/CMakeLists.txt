add_mlir_conversion_library(MTDSPToLLVM
  MTDSPToLLVMPass.cpp
  RemoveAddressSpacePass.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )