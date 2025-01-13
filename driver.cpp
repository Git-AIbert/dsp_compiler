#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <getopt.h>

#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "Dialect/Schedule/TransformOps/ScheduleTransformOps.h"

// #include "mlir/Support/RaggedArray.h"

#define LOC builder.getUnknownLoc()

using namespace mlir;

func::FuncOp createMatMulFunction(OpBuilder &builder, ModuleOp module) {
    // Create the types we need
    const int64_t dim = 1024;
    auto f32Type = builder.getF32Type();
    auto tensorType = RankedTensorType::get({dim, dim}, f32Type);
    
    // Create function type (tensor, tensor, tensor) -> tensor
    auto functionType = builder.getFunctionType(
        {tensorType, tensorType, tensorType}, // Input types
        {tensorType}                          // Result types
    );
    
    // Create the function
    builder.setInsertionPointToEnd(module.getBody());
    auto funcOp = builder.create<func::FuncOp>(
        LOC,                // Location
        "matmul",                     // Function name
        functionType                  // Function type
    );
    
    // Create the entry block and get the function arguments
    auto* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create linalg.matmul operation
    auto args = entryBlock->getArguments();
    Value lhs = args[0];
    Value rhs = args[1];
    Value output = args[2];
    
    auto matmulOp = builder.create<linalg::MatmulOp>(
        LOC,
        TypeRange{tensorType},        // Result types
        ValueRange{lhs, rhs},         // Input operands
        ValueRange{output}            // Output operands
    );
    
    // Create return operation
    builder.create<func::ReturnOp>(
        LOC,
        matmulOp.getResult(0)
    );
    
    return funcOp;
}

LogicalResult createAndApplyTransform(ModuleOp module) {
    MLIRContext* context = module->getContext();
    OpBuilder builder(context);
    
    // 1. 创建转换模块
    ModuleOp transformModule = ModuleOp::create(LOC);
    builder.setInsertionPointToEnd(transformModule.getBody());

    // 2. 创建序列操作
    auto sequenceOp = builder.create<transform::SequenceOp>(
        LOC,                                     // location
        TypeRange{},                                      // result types
        transform::FailurePropagationMode::Propagate,     // failure mode
        builder.getType<transform::AnyOpType>(),          // block argument type
        [](OpBuilder &b, Location nested, Value rootH) {} // body builder function
    );

    // 3. 插入变换操作
    auto *sequenceBody = sequenceOp.getBodyBlock();
    Value arg0 = sequenceBody->getArgument(0);
    builder.setInsertionPointToEnd(sequenceBody);

    SmallVector<StringRef, 1> opNames = {"linalg.matmul"};
    auto matmulOpHandle = builder.create<transform::MatchOp>(
        LOC,
        arg0,              // target
        opNames            // operation names to match
    );

    SmallVector<int64_t, 3> tileSizes = {256, 256, 256};  
    SmallVector<int64_t, 3> interchange = {1, 2, 0};  // 交换前两个循环的顺序
    auto tileUsingForOp = builder.create<transform::TileUsingForOp>(
        LOC, 
        matmulOpHandle,  // target
        tileSizes,     // static tile sizes
        interchange    // 指定循环交换顺序
    );
    Value tiledLinalgHandles = tileUsingForOp.getTiledLinalgOp();  // 分块后的操作
    ValueRange loopHandles = tileUsingForOp.getLoops();            // 生成的循环

    auto matmulAHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledLinalgHandles, 0);

    auto copyAHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        matmulAHandle,
        builder.getStringAttr("GSM"));

    tileSizes = {0, 64, 64};  
    interchange = {2, 1, 0};
    auto tileUsingForOp2 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledLinalgHandles,  // target
        tileSizes,     // static tile sizes
        interchange    // 指定循环交换顺序
    );
    Value tiledLinalgHandles2 = tileUsingForOp2.getTiledLinalgOp();  // 分块后的操作
    ValueRange loopHandles2 = tileUsingForOp2.getLoops();            // 生成的循环

    auto matmulBHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledLinalgHandles2, 1);

    auto copyBHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        matmulBHandle,
        builder.getStringAttr("AM"));

    tileSizes = {64};  
    auto tileUsingForOp3 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledLinalgHandles2,  // target
        tileSizes     // static tile sizes
    );
    Value tiledLinalgHandles3 = tileUsingForOp3.getTiledLinalgOp();  // 分块后的操作
    ValueRange loopHandles3 = tileUsingForOp3.getLoops();            // 生成的循环

    auto matmulCHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledLinalgHandles3, 2);

    auto readCHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        matmulCHandle,
        builder.getStringAttr("AM"));

    auto matmulResultHandle = builder.create<transform::GetResultOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledLinalgHandles3, 0);

    auto writeCHandle = builder.create<transform::CacheWriteOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        matmulResultHandle,
        builder.getStringAttr("DDR"), 
        matmulCHandle);

    tileSizes = {32};  
    auto tileUsingForOp4 = builder.create<transform::TileUsingForOp>(
        LOC, 
        tiledLinalgHandles3,  // target
        tileSizes     // static tile sizes
    );
    Value tiledLinalgHandles4 = tileUsingForOp4.getTiledLinalgOp();  // 分块后的操作
    ValueRange loopHandles4 = tileUsingForOp4.getLoops();            // 生成的循环

    auto matmulAAHandle = builder.create<transform::GetOperandOp>(
        LOC,
        builder.getType<transform::AnyValueType>(),
        tiledLinalgHandles4, 0);

    auto copyAAHandle = builder.create<transform::CacheReadOp>(
        LOC,
        builder.getType<transform::AnyOpType>(),
        matmulAAHandle,
        builder.getStringAttr("SM"));

    // 匹配所有函数操作
    auto funcOp = builder.create<transform::MatchOp>(
        LOC,
        arg0,
        SmallVector<StringRef, 1>{"func.func"}
    );

    // 应用 canonicalization 模式
    builder.create<transform::ApplyPatternsOp>(
        LOC,
        funcOp.getResult(),  // target
        [&](OpBuilder &b, Location loc) { // 即便是空，也会触发模式重写，其中包含了死代码消除的功能
            // 在 patterns 区块中添加 canonicalization
            // b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        }
    );

    // // 应用 CSE
    // builder.create<transform::ApplyCommonSubexpressionEliminationOp>(
    //     LOC,
    //     funcOp.getResult()
    // );

    // // 匹配所有循环操作
    // auto loopsOp = builder.create<transform::MatchOp>(
    //     LOC,
    //     builder.getType<transform::AnyOpType>(), // result type
    //     arg0,   // target
    //     builder.getArrayAttr({}),  // ops (empty array attribute)
    //     transform::MatchInterfaceEnumAttr::get(builder.getContext(), 
    //         transform::MatchInterfaceEnum::LoopLikeInterface),  // interface
    //     nullptr,  // op_attrs
    //     nullptr,  // filter_result_type
    //     nullptr   // filter_operand_types
    // );

    // // 应用循环不变量外提
    // auto licmOp = builder.create<transform::ApplyLoopInvariantCodeMotionOp>(
    //     LOC,
    //     loopsOp.getResult()
    // );

    builder.create<transform::YieldOp>(LOC);

    llvm::outs() << transformModule << "\n";

    // 4. 应用转换
    transform::TransformOptions options;
    if (failed(transform::applyTransforms(
        module,      // payload root
        sequenceOp,    // transform operation
        {},         // extra mapping
        options                  // options
    ))) {
        llvm::errs() << "Transform application failed\n";
        return failure();
    }

    return success();
}

LogicalResult applyOptimizationPasses(ModuleOp module, MLIRContext &context) {
    PassManager pm(&context);

    auto dumpAfterPass = [&](const std::string &passName, ModuleOp module) {
        llvm::outs() << "\n=== After " << passName << " ===\n";
        module->dump();
    };

    // 1. One-Shot Bufferize
    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocsFromLoops = false;
    options.allowUnknownOps = false;
    options.bufferizeFunctionBoundaries = true;
    options.dumpAliasSets = false;
    options.testAnalysisOnly = false;
    options.printConflicts = false;
    options.checkParallelRegions = true;
    // 设置函数参数的类型转换器
    options.functionArgTypeConverterFn = [](TensorType tensorType, 
                                          Attribute memorySpace,
                                          func::FuncOp funcOp,
                                          const bufferization::BufferizationOptions &options) -> BaseMemRefType {
        // 计算 strides
        SmallVector<int64_t> strides;
        ArrayRef<int64_t> shape = tensorType.getShape();
        int64_t stride = 1;
        
        // 从最低维到最高维计算 stride
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides.insert(strides.begin(), stride);
            if (shape[i] != ShapedType::kDynamic)
                stride *= shape[i];
        }

        // 创建带有计算出的 strides 的 layout
        auto layout = StridedLayoutAttr::get(tensorType.getContext(),
                                           /*offset=*/0,
                                           strides);
        return MemRefType::get(tensorType.getShape(),
                             tensorType.getElementType(),
                             layout,
                             memorySpace);
    };

    // 设置内存复制函数
    options.memCpyFn = [](OpBuilder &b, Location loc, Value from, Value to) {
        b.create<memref::CopyOp>(loc, from, to);
        return success();
    };

    // 添加 One-Shot Bufferize pass
    pm.addPass(bufferization::createOneShotBufferizePass(options));
    if (failed(pm.run(module))) {
        llvm::errs() << "Failed to run bufferization passes\n";
        return failure();
    }
    dumpAfterPass("One-Shot Bufferize", module);
    pm.clear();

    pm.addPass(createCSEPass());
    // if (failed(pm.run(module))) {
    //     llvm::errs() << "Failed to run CSE pass\n";
    //     return failure();
    // }
    // dumpAfterPass("CSE", module);
    // pm.clear();
    
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(module))) {
        llvm::errs() << "Failed to run Canonicalize pass\n";
        return failure();
    }
    dumpAfterPass("Canonicalize", module);
    pm.clear();

    // // 2. Buffer 优化
    // // Buffer Hoisting
    // pm.addPass(bufferization::createBufferHoistingPass());
    // dumpAfterPass("Buffer Hoisting", module);
    
    // // Loop Hoisting
    // pm.addPass(bufferization::createBufferLoopHoistingPass());
    // dumpAfterPass("Buffer Loop Hoisting", module);
    
    // // Buffer 结果转换为输出参数
    // pm.addNestedPass<func::FuncOp>(bufferization::createBufferResultsToOutParamsPass());
    // dumpAfterPass("Buffer Results to Out Params", module);

    return success();
}

int main(int argc, char* argv[]) {
    // 1.解析参数

    // 2.词法和语法分析

    // 3.创建 MLIR 上下文并加载方言
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::schedule::registerTransformDialectExtension(registry);
    mlir::registerAllPasses();

    MLIRContext context;
    context.appendDialectRegistry(registry);
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<linalg::LinalgDialect>();
    context.loadDialect<transform::TransformDialect>();
    context.loadDialect<memref::MemRefDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    context.loadDialect<vector::VectorDialect>();
    context.loadDialect<cf::ControlFlowDialect>();
    context.loadDialect<affine::AffineDialect>();
    context.loadDialect<LLVM::LLVMDialect>();
    context.loadDialect<pdl_interp::PDLInterpDialect>();
    context.loadDialect<pdl::PDLDialect>();
    // context.loadDialect<mtdsp::MTDSPDialect>();

    context.getDiagEngine().registerHandler([](Diagnostic &diag) {
        llvm::errs() << "[DEBUG] " << diag.str() << "\n";
        return success();
    });

    // 4.生成 MLIR
    OpBuilder builder(&context);
    ModuleOp module = ModuleOp::create(LOC);

    // 创建矩阵乘法函数
    createMatMulFunction(builder, module);
    llvm::outs() << "\n=== Original MLIR ===\n";
    module->dump();

    // 进行变换
    createAndApplyTransform(module);
    llvm::outs() << "\n=== After Transform ===\n";
    module->dump();

    // 使用pass进行优化
    if (failed(applyOptimizationPasses(module, context))) {
        llvm::errs() << "Failed to Optimize module\n";
        return -1;
    }

    return 0;
}