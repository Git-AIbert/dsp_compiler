//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all
// mtivm-specific dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_INITALLDIALECTS_H
#define MTIR_INITALLDIALECTS_H

#include "Dialect/Schedule/IR/ScheduleDialect.h"
#include "Dialect/MTDSP/IR/MTDSPDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mtir {

/// Add all the mtir-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::schedule::ScheduleDialect>();
  registry.insert<mlir::mtdsp::MTDSPDialect>();
}

/// Append all the mtir-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mtir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mtir

#endif // MTIR_INITALLDIALECTS_H
