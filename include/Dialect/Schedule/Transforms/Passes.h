#ifndef SCHEDULE_TRANSOFORM_PASSES_H
#define SCHEDULE_TRANSOFORM_PASSES_H

#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<::mlir::Pass> createStaticizeTensorEmptyPass();

std::unique_ptr<::mlir::Pass> createMultiBufferPass();

std::unique_ptr<::mlir::Pass> createParallelPass();

std::unique_ptr<::mlir::Pass> createUnrollPass();
}

#endif // SCHEDULE_TRANSOFORM_PASSES_H