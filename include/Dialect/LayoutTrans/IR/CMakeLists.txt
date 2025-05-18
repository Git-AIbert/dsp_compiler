set(LLVM_TARGET_DEFINITIONS LayoutTransBase.td)
mlir_tablegen(LayoutTransDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LayoutTransDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(LayoutTransDialectIncGen)

set(LLVM_TARGET_DEFINITIONS LayoutTransOps.td)
mlir_tablegen(LayoutTransOps.h.inc -gen-op-decls)
mlir_tablegen(LayoutTransOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LayoutTransOpsIncGen)