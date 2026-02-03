#ifndef CASCADE_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H
#define CASCADE_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace cascade {

#define GEN_PASS_DECL_CONVERTLINALGTOMTDSP
#include "Conversion/Passes.h.inc"

/// Creates a pass to convert the linalg dialect to the MTDSP dialect.
std::unique_ptr<mlir::Pass> createConvertLinalgToMTDSPPass();

} // namespace cascade

#endif // CASCADE_CONVERSION_LINALGTOMTDSP_LINALGTOMTDSPPASS_H