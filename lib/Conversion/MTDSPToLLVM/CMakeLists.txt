add_mlir_conversion_library(MLIRMTDSPToLLVM
  MTDSPToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/MTDSPToLLVM

  DEPENDS
  MTIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MTDSPDialect
  MLIRTransformUtils
  MLIRPass
  )
