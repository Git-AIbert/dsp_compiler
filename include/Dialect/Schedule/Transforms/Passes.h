#ifndef SCHEDULE_TRANSOFORM_PASSES_H
#define SCHEDULE_TRANSOFORM_PASSES_H

#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<::mlir::Pass> createStaticizeTensorEmptyPass();
}

#endif // SCHEDULE_TRANSOFORM_PASSES_H