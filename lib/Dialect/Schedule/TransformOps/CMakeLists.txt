add_mlir_dialect_library(ScheduleTransformOps
    ConsumerFusion.cpp
    DialectExtension.cpp
    FusionValidator.cpp
    ScheduleTransformOps.cpp

    ADDITIONAL_HEADER_DIRS
    include/mlir/Dialect/Schedule/TransformOps

    DEPENDS
    MLIRScheduleTransformOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTransformDialect
    ScheduleDialect
)