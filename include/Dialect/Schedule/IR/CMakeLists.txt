set(LLVM_TARGET_DEFINITIONS ScheduleBase.td)
mlir_tablegen(ScheduleDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ScheduleDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ScheduleDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ScheduleOps.td)
mlir_tablegen(ScheduleOps.h.inc -gen-op-decls)
mlir_tablegen(ScheduleOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ScheduleOpsIncGen)