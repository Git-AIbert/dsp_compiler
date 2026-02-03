//===-- Passes.h - MTDSP dialect transformation passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTDSP_TRANSFORM_PASSES_H
#define MTDSP_TRANSFORM_PASSES_H

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "Dialect/MTDSP/Transforms/Passes.h.inc"

/// Create a pass that promotes mtdsp.alloc operations without memory space
/// attribute or with Global memory space to function arguments.
std::unique_ptr<mlir::Pass> createPromoteAllocsToArgumentsPass();

/// Create a pass that removes address space attributes from all MemRef types.
std::unique_ptr<mlir::Pass> createRemoveMemRefAddressSpacePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/MTDSP/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MTDSP_TRANSFORM_PASSES_H
