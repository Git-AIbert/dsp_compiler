add_mlir_dialect_library(ScheduleTransformOps
    DialectExtension.cpp
    ScheduleTransformOps.cpp

    ADDITIONAL_HEADER_DIRS
    include/mlir/Dialect/Schedule/TransformOps

    DEPENDS
    MLIRScheduleTransformOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransformDialect
    ScheduleDialect
)