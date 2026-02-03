//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all cascade
// dialect extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef CASCADE_INITALLEXTENSIONS_H
#define CASCADE_INITALLEXTENSIONS_H

#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace cascade {

inline void registerAllExtensions(mlir::DialectRegistry &registry) {
  // Register all transform dialect extensions.
  mlir::schedule::registerTransformDialectExtension(registry);
}

} // namespace cascade

#endif // CASCADE_INITALLEXTENSIONS_H