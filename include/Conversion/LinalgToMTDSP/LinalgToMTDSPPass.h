#ifndef MTIR_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H
#define MTIR_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mtir {

#define GEN_PASS_DECL_CONVERTLINALGTOMTDSP
#include "Conversion/Passes.h.inc"

/// Creates a pass to convert the linalg dialect to the MTDSP dialect.
std::unique_ptr<mlir::Pass> createConvertLinalgToMTDSPPass();

} // namespace mtir

#endif // MTIR_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H