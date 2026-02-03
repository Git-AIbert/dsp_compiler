#ifndef MTIR_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H
#define MTIR_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mtir {

#define GEN_PASS_DECL_CONVERTMEMREFTOMTDSP
#include "Conversion/Passes.h.inc"

/// Creates a pass to convert the memref dialect to the MTDSP dialect.
std::unique_ptr<mlir::Pass> createConvertMemRefToMTDSPPass();

} // namespace mtir

#endif // MTIR_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H