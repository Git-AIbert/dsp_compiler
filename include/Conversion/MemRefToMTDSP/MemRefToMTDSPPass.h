#ifndef CASCADE_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H
#define CASCADE_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace cascade {

#define GEN_PASS_DECL_CONVERTMEMREFTOMTDSP
#include "Conversion/Passes.h.inc"

/// Creates a pass to convert the memref dialect to the MTDSP dialect.
std::unique_ptr<mlir::Pass> createConvertMemRefToMTDSPPass();

} // namespace cascade

#endif // CASCADE_CONVERSION_MEMREFTOMTDSP_MEMREFTOMTDSPPASS_H