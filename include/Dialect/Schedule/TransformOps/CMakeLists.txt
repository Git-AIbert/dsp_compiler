set(LLVM_TARGET_DEFINITIONS ScheduleTransformOps.td)
mlir_tablegen(ScheduleTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ScheduleTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRScheduleTransformOpsIncGen)