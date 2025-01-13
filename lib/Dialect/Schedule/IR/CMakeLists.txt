add_mlir_dialect_library(ScheduleDialect
  ScheduleDialect.cpp

  DEPENDS
  ScheduleDialectIncGen
  ScheduleOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferIntRangeInterface
)