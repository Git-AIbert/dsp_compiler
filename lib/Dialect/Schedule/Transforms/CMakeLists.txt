add_mlir_dialect_library(ScheduleTransforms
    StaticizeTensorEmptyPass.cpp
    MultiBufferPass.cpp
    OptimizeDMAPass.cpp
    ParallelPass.cpp
    UnrollPass.cpp
    OneShotBufferizeWithMemorySpacePass.cpp
    CustomCanonicalizationPatterns.cpp
    CustomCanonicializePass.cpp
    DeduplicateMultiBufferPass.cpp
    ChainSplitReductionPipelinesPass.cpp
    BufferLoopSinkingPass.cpp
    RemoveFunctionReturnsPass.cpp

    DEPENDS
    MLIRSchedulePassIncGen
)