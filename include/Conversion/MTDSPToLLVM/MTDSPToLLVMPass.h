#ifndef MTIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H
#define MTIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mtir {

#define GEN_PASS_DECL_CONVERTMTDSPTOLLVM
#include "Conversion/Passes.h.inc"

/// Creates a pass to convert the MTDSP dialect to the LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertMTDSPToLLVMPass();

} // namespace mtir

#endif // MTIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H