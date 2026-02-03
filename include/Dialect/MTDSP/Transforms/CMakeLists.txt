set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MTDSP)
add_public_tablegen_target(MLIRMTDSPPassIncGen)

add_mlir_doc(Passes MTDSPPasses ./ -gen-pass-doc)
