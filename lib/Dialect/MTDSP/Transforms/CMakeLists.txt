add_mlir_dialect_library(MTDSPTransforms
    PromoteAllocsToArguments.cpp
    RemoveMemRefAddressSpace.cpp

    DEPENDS
    MLIRMTDSPPassIncGen
    MTDSPAttributesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTransformUtils
)
