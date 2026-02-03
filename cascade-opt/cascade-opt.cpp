//===- cascade-opt.cpp - Cascade Optimizer Driver -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for cascade-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "InitAllDialects.h"
#include "InitAllExtensions.h"
#include "InitAllPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

// Test dialect extensions
namespace test {
void registerTestTilingInterfaceTransformDialectExtension(mlir::DialectRegistry &);
} // namespace test

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  cascade::registerAllDialects(registry);
  // Register dialect extensions.
  mlir::registerAllExtensions(registry);
  cascade::registerAllExtensions(registry);
  // Register test dialect extensions.
  test::registerTestTilingInterfaceTransformDialectExtension(registry);
  // Register passes.
  mlir::registerAllPasses();
  cascade::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MT3000 MLIR optimizer driver\n", registry));
}
