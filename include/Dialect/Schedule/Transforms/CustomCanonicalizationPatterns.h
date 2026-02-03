//===- CustomCanonicalizationPatterns.h - Custom Canonicalization -------===//
//
// This header provides reusable patterns for canonicalization that preserve
// single-iteration loops. These patterns are used by both:
// 1. The CustomCanonicalize pass
// 2. The ApplyCustomCanonicalizationPatternsOp Transform operation
//
//===----------------------------------------------------------------------===//

#ifndef MTDSP_DIALECT_SCHEDULE_TRANSFORMS_CUSTOMCANONICALIZATIONPATTERNS_H
#define MTDSP_DIALECT_SCHEDULE_TRANSFORMS_CUSTOMCANONICALIZATIONPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {

class MLIRContext;

/// Populate patterns that perform canonicalization while preserving
/// single-iteration loops. This function collects canonicalization patterns
/// from all dialects and operations, but excludes the SimplifyTrivialLoops
/// pattern for scf.for and adds custom ForOp patterns instead.
///
/// The custom patterns include:
/// - ForOpIterArgsFolder: Folds away unnecessary ForOp iter arguments
/// - ForOpTensorCastFolder: Optimizes tensor.cast operations in loops
/// - SimplifyTrivialLoopsExceptSingleIteration: Removes 0-iteration and
///   empty loops but preserves single-iteration loops
///
/// This function is used by both:
/// 1. The CustomCanonicalize pass (--custom-canonicalize)
/// 2. The ApplyCustomCanonicalizationPatternsOp Transform operation
///    (transform.apply_patterns.custom_canonicalization)
void populateCustomCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *ctx);

} // namespace mlir

#endif // MTDSP_DIALECT_SCHEDULE_TRANSFORMS_CUSTOMCANONICALIZATIONPATTERNS_H
