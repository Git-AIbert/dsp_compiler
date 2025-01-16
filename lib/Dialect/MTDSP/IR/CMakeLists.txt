add_mlir_dialect_library(MTDSPDialect
  MTDSPDialect.cpp

  DEPENDS
  MTDSPDialectIncGen
  MTDSPOpsIncGen
  MTDSPEnumsIncGen
  MTDSPAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferIntRangeInterface
)