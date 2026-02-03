#ifndef MTIR_CONVERSION_PASSES_H
#define MTIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "Conversion/LinalgToMTDSP/LinalgToMTDSPPass.h"
#include "Conversion/MemRefToMTDSP/MemRefToMTDSPPass.h"
#include "Conversion/MTDSPToLLVM/MTDSPToLLVMPass.h"

namespace mtir {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace mtir

#endif // MTIR_CONVERSION_PASSES_H