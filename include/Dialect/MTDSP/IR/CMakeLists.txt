set(LLVM_TARGET_DEFINITIONS MTDSPBase.td)
mlir_tablegen(MTDSPDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MTDSPDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MTDSPDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MTDSPOps.td)
mlir_tablegen(MTDSPOps.h.inc -gen-op-decls)
mlir_tablegen(MTDSPOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MTDSPOpsIncGen)

set(LLVM_TARGET_DEFINITIONS MTDSPOps.td)
mlir_tablegen(MTDSPEnums.h.inc -gen-enum-decls)
mlir_tablegen(MTDSPEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MTDSPEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS MTDSPOps.td)
mlir_tablegen(MTDSPAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=mtdsp)
mlir_tablegen(MTDSPAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mtdsp)
add_public_tablegen_target(MTDSPAttributesIncGen)