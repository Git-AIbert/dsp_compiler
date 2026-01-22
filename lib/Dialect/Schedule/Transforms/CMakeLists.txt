add_mlir_dialect_library(ScheduleTransforms
    StaticizeTensorEmptyPass.cpp
    MultiBufferPass.cpp
    ParallelPass.cpp
    UnrollPass.cpp
    OneShotBufferizeWithMemorySpacePass.cpp
    CustomCanonicalizationPatterns.cpp
    CustomCanonicializePass.cpp

    DEPENDS
    MLIRSchedulePassIncGen
)