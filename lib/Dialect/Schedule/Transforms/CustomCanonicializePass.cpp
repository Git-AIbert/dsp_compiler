//===- CustomCanonicializePass.cpp - Custom Canonicalization Pass --------===//
//
// This pass implements canonicalization while preserving single-iteration
// loops by excluding the SimplifyTrivialLoops pattern.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Schedule/Transforms/Passes.h"
#include "Dialect/Schedule/Transforms/CustomCanonicalizationPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CUSTOMCANONICALIZE
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct CustomCanonicalize
    : public impl::CustomCanonicalizeBase<CustomCanonicalize> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // Populate custom canonicalization patterns that preserve single-iteration loops
    populateCustomCanonicalizationPatterns(patterns, &getContext());

    // Create frozen pattern set
    auto frozenPatterns = FrozenRewritePatternSet(std::move(patterns));

    // Apply patterns with greedy rewrite
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Normal;
    config.maxIterations = 10;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns,
                                             config))) {
      // Canonicalization is best-effort, so we don't signal failure
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCustomCanonicializePass() {
  return std::make_unique<CustomCanonicalize>();
}
