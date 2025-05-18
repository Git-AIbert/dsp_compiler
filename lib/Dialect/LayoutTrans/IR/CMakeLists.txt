add_mlir_dialect_library(LayoutTransDialect
  LayoutTransDialect.cpp

  DEPENDS
  LayoutTransDialectIncGen
  LayoutTransOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferIntRangeInterface
)