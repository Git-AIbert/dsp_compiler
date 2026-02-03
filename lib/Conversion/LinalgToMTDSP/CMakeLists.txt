add_mlir_conversion_library(MLIRLinalgToMTDSP
  LinalgToMTDSP.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/LinalgToMTDSP

  DEPENDS
  MTIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MTDSPDialect
  MLIRTransformUtils
  MLIRPass
  )
