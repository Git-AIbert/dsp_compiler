//===- InitAllPasses.h - MLIR Passes Registration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_INITALLPASSES_H
#define MTIR_INITALLPASSES_H

#include "Conversion/Passes.h"
#include "Dialect/MTDSP/Transforms/Passes.h"
#include "Dialect/Schedule/Transforms/Passes.h"

namespace mtir {

// This function may be called to register the hivm-specific MLIR passes with
// the global registry.
inline void registerAllPasses() {
  // Conversion passes
  mtir::registerConversionPasses();
  // Dialect-specific transform passes
  mlir::registerMTDSPPasses();
  mlir::registerSchedulePasses();
}

} // namespace mtir

#endif // MTIR_INITALLPASSES_H
