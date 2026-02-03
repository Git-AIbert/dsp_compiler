add_mlir_conversion_library(MLIRMemRefToMTDSP
  MemRefToMTDSP.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/MemRefToMTDSP

  DEPENDS
  CASCADEConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MTDSPDialect
  MLIRTransformUtils
  MLIRPass
  )
