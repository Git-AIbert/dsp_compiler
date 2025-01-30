#ifndef MLIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H_
#define MLIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H_

#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<::mlir::Pass> createMTDSPToLLVMConversionPass();

std::unique_ptr<::mlir::Pass> createRemoveAddressSpacePass();
}

#endif // MLIR_CONVERSION_MTDSPTOLLVM_MTDSPTOLLVMPASS_H_