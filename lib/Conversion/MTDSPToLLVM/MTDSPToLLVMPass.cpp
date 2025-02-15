#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Conversion/MTDSPToLLVM/MTDSPToLLVMPass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MTDSPToFunc RewritePatterns: ThreadIdOp
//===----------------------------------------------------------------------===//

class ThreadIdOpLowering : public OpConversionPattern<mtdsp::ThreadIdOp> {
  using OpConversionPattern<mtdsp::ThreadIdOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 检查函数是否已经声明
    if (!module.lookupSymbol("get_thread_id")) {
        // 声明函数类型
        auto functionType = FunctionType::get(op.getContext(), 
                                            /*inputs=*/{}, 
                                            /*results=*/{rewriter.getI32Type()});
        
        // 在模块开始处创建函数声明
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<func::FuncOp>(
            module->getLoc(),
            "get_thread_id",        // 函数名
            functionType            // 函数类型
        ).setPrivate();             // 设置为私有
    }
    
    auto callOp = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),     // results
        "get_thread_id",           // callee
        ValueRange());             // empty args
    
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MTDSPToFunc RewritePatterns: GroupSizeOp
//===----------------------------------------------------------------------===//

class GroupSizeOpLowering : public OpConversionPattern<mtdsp::GroupSizeOp> {
  using OpConversionPattern<mtdsp::GroupSizeOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::GroupSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 检查函数是否已经声明
    if (!module.lookupSymbol("get_group_size")) {
        // 声明函数类型
        auto resultType = op.getType();
        auto functionType = FunctionType::get(op.getContext(), 
                                            /*inputs=*/{}, 
                                            /*results=*/{rewriter.getI32Type()});
        
        // 在模块开始处创建函数声明
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<func::FuncOp>(
            module->getLoc(),
            "get_group_size",       // 函数名
            functionType            // 函数类型
        ).setPrivate();             // 设置为私有
    }
    
    auto callOp = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),     // results
        "get_group_size",          // callee 
        ValueRange());             // empty args
    
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MTDSPToLLVM RewritePatterns: AllocOp
//===----------------------------------------------------------------------===//

class AllocOpLowering : public OpConversionPattern<mtdsp::AllocOp> {
  using OpConversionPattern<mtdsp::AllocOp>::OpConversionPattern;
  static int nameCounter;
public:
  LogicalResult
  matchAndRewrite(mtdsp::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 判断分配的memref的内存空间索引
    // 如果是默认或global，返回失败
    mlir::Attribute attr = op.getType().getMemorySpace();
    if(!attr) return failure();
    mtdsp::AddressSpaceAttr addrSpaceAttr = llvm::cast<mtdsp::AddressSpaceAttr>(attr);
    
    // 申请簇上共享空间
    MemRefType memrefTy = op.getType();
    ArrayRef<int64_t> shape = memrefTy.getShape();
    Type elemType = memrefTy.getElementType();

    // 在模块开始处创建 GlobalOp
    int64_t numElements = 1;
    for (int64_t dim : shape) {
        // 如果维度是动态的(-1)，这里需要特殊处理
        if (dim == ShapedType::kDynamic)
            continue;  // 或者其他处理逻辑
        numElements *= dim;
    }

    auto zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    Value addr;
    switch (addrSpaceAttr.getValue()) {
      case mtdsp::AddressSpace::Global:
        return failure();
      case mtdsp::AddressSpace::Workgroup:{
        // 申请簇上共享内存
        // 在模块开始处创建 GlobalOp
        LLVM::LLVMArrayType arrayType = LLVM::LLVMArrayType::get(elemType, numElements);
        std::string globalName = "gsm_" + std::to_string(nameCounter++);
        LLVM::GlobalOp global;
        {
          auto module = op->getParentOfType<ModuleOp>();
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(module.getBody());
          global = rewriter.create<LLVM::GlobalOp>(
              loc,
              arrayType,
              /*isConstant=*/false,
              LLVM::Linkage::Internal,
              globalName,
              /*initValue=*/Attribute(),
              /*alignment=*/0,
              /*addrSpace=*/0);
        }
    
        // 创建 AddressOfOp
        auto addressOf = rewriter.create<LLVM::AddressOfOp>(
            loc, 
            LLVM::LLVMPointerType::get(arrayType.getContext(), 0),
            global.getSymName());

        // 将数组类型转换为指针类型
        auto ptrType = LLVM::LLVMPointerType::get(op->getContext(), 0);
        SmallVector<NamedAttribute, 1> attributes;
        auto elemTypeAttr = TypeAttr::get(arrayType.getElementType());
        attributes.push_back(rewriter.getNamedAttr("elem_type", elemTypeAttr));
        addr = rewriter.create<LLVM::GEPOp>(
          loc,
          ptrType,                // resultType
          arrayType,              // elementType
          addressOf.getResult(),  // basePtr
          ValueRange{zero, zero}, // indices
          /*inbounds=*/true       // inbounds flag
        );
          break;
      }
      case mtdsp::AddressSpace::Scalar:{
        // 申请私有标量内存
        // 检查函数是否已经声明
        if (!module.lookupSymbol("scalar_malloc")) {
            // 声明函数类型
            auto functionType = FunctionType::get(op.getContext(), 
                                                /*inputs=*/{rewriter.getI32Type()}, 
                                                /*results=*/{rewriter.getType<LLVM::LLVMPointerType>()});
            
            // 在模块开始处创建函数声明
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            rewriter.create<func::FuncOp>(
                module->getLoc(),
                "scalar_malloc",        // 函数名
                functionType            // 函数类型
            ).setPrivate();             // 设置为私有
        }
        
        DataLayout layout(module);
        unsigned elementSize = layout.getTypeSize(elemType);
        auto totalBytes = rewriter.create<LLVM::ConstantOp>(
            loc, 
            rewriter.getI32Type(), 
            numElements * elementSize);
        auto callOp = rewriter.create<func::CallOp>(
            loc,
            rewriter.getType<LLVM::LLVMPointerType>(),  // results
            "scalar_malloc",                            // callee 
            ValueRange{totalBytes});                    // bytes
        addr = callOp->getResult(0);
        break;
      }
      case mtdsp::AddressSpace::Vector:{
        // 申请私有向量内存
        // 检查函数是否已经声明
        if (!module.lookupSymbol("vector_malloc")) {
            // 声明函数类型
            auto functionType = FunctionType::get(op.getContext(), 
                                                /*inputs=*/{rewriter.getI32Type()}, 
                                                /*results=*/{rewriter.getType<LLVM::LLVMPointerType>()});
            
            // 在模块开始处创建函数声明
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            rewriter.create<func::FuncOp>(
                module->getLoc(),
                "vector_malloc",        // 函数名
                functionType            // 函数类型
            ).setPrivate();             // 设置为私有
        }
        
        DataLayout layout(module);
        unsigned elementSize = layout.getTypeSize(elemType);
        auto totalBytes = rewriter.create<LLVM::ConstantOp>(
            loc, 
            rewriter.getI32Type(), 
            numElements * elementSize);
        auto callOp = rewriter.create<func::CallOp>(
            loc,
            rewriter.getType<LLVM::LLVMPointerType>(),  // results
            "vector_malloc",                            // callee 
            ValueRange{totalBytes});                    // bytes
        addr = callOp->getResult(0);
        break;
      }
    }
    
    // 创建memref结构
    auto structType = getTypeConverter()->convertType(memrefTy);
    auto desc = MemRefDescriptor::undef(rewriter, loc, structType);
    desc.setAllocatedPtr(rewriter, loc, addr);
    desc.setAlignedPtr(rewriter, loc, addr);
    desc.setOffset(rewriter, loc, zero);
    for (unsigned i = 0; i < shape.size(); ++i) {
      auto dimSize = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), shape[i]);
      desc.setSize(rewriter, loc, i, dimSize);
      
      int64_t stride = 1;
      for (unsigned j = i + 1; j < shape.size(); ++j) {
        if (shape[j] != ShapedType::kDynamic)
          stride *= shape[j];
      }
      auto strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), stride);
      desc.setStride(rewriter, loc, i, strideVal);
    }
    
    // 转换回memref类型
    auto memRef = rewriter.create<UnrealizedConversionCastOp>(
        loc, memrefTy, ValueRange{desc});
        
    rewriter.replaceOp(op, memRef.getResult(0));
    return success();
  }
};

// Initialize the static counter
int AllocOpLowering::nameCounter = 0;

//===----------------------------------------------------------------------===//
// MTDSPToFunc RewritePatterns: DeallocOp
//===----------------------------------------------------------------------===//

class DeallocOpLowering : public OpConversionPattern<mtdsp::DeallocOp> {
  using OpConversionPattern<mtdsp::DeallocOp>::OpConversionPattern;
private:
  // 辅助函数：创建内存释放函数声明
  void ensureFreeFunction(ConversionPatternRewriter &rewriter, 
                         ModuleOp module,
                         StringRef funcName) const {
    if (!module.lookupSymbol(funcName)) {
      auto functionType = FunctionType::get(getContext(),
                                          /*inputs=*/{rewriter.getType<LLVM::LLVMPointerType>()},
                                          /*results=*/{rewriter.getI32Type()});
      
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<func::FuncOp>(
          module->getLoc(),
          funcName,           // 函数名
          functionType        // 函数类型
      ).setPrivate();        // 设置为私有
    }
  }

  // 辅助函数：创建内存释放调用
  void createFreeCall(ConversionPatternRewriter &rewriter,
                     Location loc,
                     Value memref,
                     StringRef funcName) const {
    auto memRefType = memref.getType().cast<MemRefType>();
    
    // 将 memref 转换为 LLVM descriptor 格式
    auto structType = getTypeConverter()->convertType(memRefType);
    auto desc = rewriter.create<UnrealizedConversionCastOp>(
        loc, structType, ValueRange{memref}).getResult(0);
    
    // 从 descriptor 提取 aligned pointer (索引为1的字段)
    auto alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc,
        LLVM::LLVMPointerType::get(getContext()),
        desc,
        ArrayRef<int64_t>{1});
    
    // 调用释放函数
    rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),    
        funcName,                 
        ValueRange{alignedPtr});  
  }

public:
  LogicalResult
  matchAndRewrite(mtdsp::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto memref = op.getMemref();
    auto memRefType = memref.getType().cast<MemRefType>();
    mlir::Attribute attr = memRefType.getMemorySpace();
    if(!attr) return failure();
    mtdsp::AddressSpaceAttr addrSpaceAttr = llvm::cast<mtdsp::AddressSpaceAttr>(attr);

    switch (addrSpaceAttr.getValue()) {
      case mtdsp::AddressSpace::Global:
        return failure();
      case mtdsp::AddressSpace::Workgroup:{
        rewriter.eraseOp(op);
        return success();
      }
      case mtdsp::AddressSpace::Scalar:{
        ensureFreeFunction(rewriter, module, "scalar_free");
        createFreeCall(rewriter, loc, memref, "scalar_free");
        rewriter.eraseOp(op);
        return success();
      }
      case mtdsp::AddressSpace::Vector:
        ensureFreeFunction(rewriter, module, "vector_free");
        createFreeCall(rewriter, loc, memref, "vector_free");
        rewriter.eraseOp(op);
        return success();
    }
  }
};

//===----------------------------------------------------------------------===//
// Common Utilities for DMA Operations
//===----------------------------------------------------------------------===//

// 声明 DMA 相关函数的通用函数
void declareDMAFunction(ConversionPatternRewriter &rewriter, ModuleOp module,
                       StringRef funcName, bool hasChannelParam = false) {
  if (module.lookupSymbol(funcName))
    return;
    
  // 基本输入类型
  SmallVector<Type, 11> inputTypes = {
      rewriter.getType<LLVM::LLVMPointerType>(),  // void* src
      rewriter.getI64Type(),                      // unsigned long src_row_num
      rewriter.getI32Type(),                      // unsigned int src_row_size
      rewriter.getI32Type(),                      // int src_row_step
      rewriter.getType<LLVM::LLVMPointerType>(),  // void* dst
      rewriter.getI64Type(),                      // unsigned long dst_row_num
      rewriter.getI32Type(),                      // unsigned int dst_row_size
      rewriter.getI32Type(),                      // int dst_row_step
      rewriter.getI1Type(),                       // bool row_syn
      rewriter.getI32Type()                       // unsigned int synmask/p2pmask
  };
  
  // 如果需要channel参数，添加到输入类型列表
  if (hasChannelParam)
    inputTypes.push_back(rewriter.getI32Type());  // int ch
    
  auto functionType = FunctionType::get(module.getContext(), 
                                      inputTypes, 
                                      {rewriter.getI32Type()}); // 返回 unsigned int
                                      
  // 在模块开始处创建函数声明
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<func::FuncOp>(
      module->getLoc(),
      funcName,
      functionType
  ).setPrivate();
}

// 提取 DMA 传输参数的通用函数
LogicalResult extractDMATransferParams(
    ConversionPatternRewriter &rewriter, Location loc,
    Value memrefDescriptor, SmallVectorImpl<Value> &callOperands) {
  // 获取aligned ptr和offset
  Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, memrefDescriptor, ArrayRef<int64_t>{1});
  Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, memrefDescriptor, ArrayRef<int64_t>{2});
  // 计算偏移后的地址
  Value offsetPtr = rewriter.create<LLVM::GEPOp>(loc,
    rewriter.getType<LLVM::LLVMPointerType>(),  // 结果类型是指针
    rewriter.getF32Type(),                      // 元素类型(float) TODO
    alignedPtr,                                 // 基地址
    offset,                                     // 偏移量
    /*inbounds=*/true);
  
  // 直接获取shape中的维度 (一次性获取)
  Value numRows64 = rewriter.create<LLVM::ExtractValueOp>(loc, memrefDescriptor, ArrayRef<int64_t>{3, 0});
  Value numCols64 = rewriter.create<LLVM::ExtractValueOp>(loc, memrefDescriptor, ArrayRef<int64_t>{3, 1});
  Value stride064 = rewriter.create<LLVM::ExtractValueOp>(loc, memrefDescriptor, ArrayRef<int64_t>{4, 0});
  
  // 转换为i32
  Value numCols = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), numCols64);
  Value stride0 = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), stride064);
  
  // 计算每个元素的字节大小 (假设是float类型，4字节)
  Value elemSize = rewriter.create<LLVM::ConstantOp>(loc, 
      rewriter.getI32Type(), rewriter.getI32IntegerAttr(4));
  
  // 计算每行的字节数
  Value rowSize = rewriter.create<LLVM::MulOp>(loc, numCols, elemSize);
  
  // 计算完整stride的字节数
  Value fullStride = rewriter.create<LLVM::MulOp>(loc, stride0, elemSize);
  
  // 计算行间偏移 = fullStride - rowSize
  Value rowStep = rewriter.create<LLVM::SubOp>(loc, fullStride, rowSize);
  
  // 添加所有参数
  callOperands.push_back(offsetPtr);       // 偏移后的地址
  callOperands.push_back(numRows64);       // 行数（i64）
  callOperands.push_back(rowSize);         // 每行字节数（i32）
  callOperands.push_back(rowStep);         // 行间偏移（i32）

  // // 验证可以放在最后，使用新创建的Values进行验证
  // const uint32_t maxSize = 8 * 1024 * 1024; // 8M
  
  // // 如果numRows是常量，验证行数
  // if (auto constRows = numRows.getDefiningOp<LLVM::ConstantOp>()) {
  //   if (constRows.getValue().cast<IntegerAttr>().getInt() > maxSize) {
  //     return emitError(loc, "row number cannot exceed 8M");
  //   }
  // }
  
  // // 如果rowSize是常量，验证每行字节数
  // if (auto constRowSize = rowSize.getDefiningOp<LLVM::ConstantOp>()) {
  //   if (constRowSize.getValue().cast<IntegerAttr>().getInt() > maxSize) {
  //     return emitError(loc, "row size in bytes cannot exceed 8M");
  //   }
  // }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MTDSPToLLVM RewritePatterns: DMAOp
//===----------------------------------------------------------------------===//

class DMAOpLowering : public OpConversionPattern<mtdsp::DMAOp> {
  using OpConversionPattern<mtdsp::DMAOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::DMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 1. 声明函数
    declareDMAFunction(rewriter, module, "dma_p2p");

    // 2. 计算参数
    SmallVector<Value, 10> callOperands;

    // 获取src相关参数
    if (failed(extractDMATransferParams(rewriter, loc, adaptor.getSrc(), callOperands))) {
      return failure();
    }
    
    // 获取dst相关参数
    if (failed(extractDMATransferParams(rewriter, loc, adaptor.getDst(), callOperands))) {
      return failure();
    }
    
    // 添加row_syn和synmask参数
    callOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        loc, 0, rewriter.getI1Type()));  // row_syn = false
    callOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        loc, 0, rewriter.getI32Type())); // synmask = 0
    
    // 3. 创建函数调用
    auto callOp = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),
        "dma_p2p",
        callOperands);
    
    // 4. 将原始op的结果替换为函数调用的结果
    rewriter.replaceOp(op, callOp.getResult(0));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// DMAOptOp Lowering
//===----------------------------------------------------------------------===//

class DMAOptOpLowering : public OpConversionPattern<mtdsp::DMAOptOp> {
  using OpConversionPattern<mtdsp::DMAOptOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::DMAOptOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 1. 声明函数
    declareDMAFunction(rewriter, module, "dma_p2p_opt", /*hasChannelParam=*/true);

    // 2. 计算参数
    SmallVector<Value, 11> callOperands;

    // 获取src相关参数
    if (failed(extractDMATransferParams(rewriter, loc, adaptor.getSrc(), callOperands))) {
      return failure();
    }
    
    // 获取dst相关参数
    if (failed(extractDMATransferParams(rewriter, loc, adaptor.getDst(), callOperands))) {
      return failure();
    }
    
    // 添加row_syn和p2pmask参数
    callOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        loc, 0, rewriter.getI1Type()));  // row_syn = false
    callOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        loc, 0, rewriter.getI32Type())); // p2pmask = 0
        
    // 添加channel参数
    callOperands.push_back(adaptor.getChannel());
    
    // 3. 创建函数调用
    auto callOp = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),
        "dma_p2p_opt",
        callOperands);
    
    // 4. 将原始op的结果替换为函数调用的结果
    rewriter.replaceOp(op, callOp.getResult(0));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// MTDSPToLLVM RewritePatterns: WaitOp
//===----------------------------------------------------------------------===//

class WaitOpLowering : public OpConversionPattern<mtdsp::WaitOp> {
  using OpConversionPattern<mtdsp::WaitOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 检查函数是否已经声明
    if (!module.lookupSymbol("dma_wait")) {
        // 声明函数类型
        auto channelType = op.getChannel().getType();
        auto functionType = FunctionType::get(op.getContext(), 
                                            /*inputs=*/{channelType}, 
                                            /*results=*/{}); // void返回类型
        
        // 在模块开始处创建函数声明
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<func::FuncOp>(
            module->getLoc(),
            "dma_wait",            // 函数名
            functionType           // 函数类型
        ).setPrivate();           // 设置为私有
    }
    
    // 创建对dma_wait的调用
    rewriter.create<func::CallOp>(
        loc,
        TypeRange{},              // 无返回值
        "dma_wait",              // callee
        ValueRange{adaptor.getChannel()}); // 传入channel参数
    
    // 由于原操作没有返回值,直接擦除原操作即可
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MTDSPToLLVM RewritePatterns: MatmulR6C96Op
//===----------------------------------------------------------------------===//

class MatmulR6C96OpLowering : public OpConversionPattern<mtdsp::MatmulR6C96Op> {
  using OpConversionPattern<mtdsp::MatmulR6C96Op>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(mtdsp::MatmulR6C96Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // 1. 检查函数是否已经声明
    if (!module.lookupSymbol("micro_kernel_asm_r6c96")) {
      // 声明函数类型
      SmallVector<Type, 6> inputTypes = {
          rewriter.getType<LLVM::LLVMPointerType>(),      // float* src_a
          rewriter.getType<LLVM::LLVMPointerType>(),      // lvector float* src_b 
          rewriter.getType<LLVM::LLVMPointerType>(),      // lvector float* dst_c
          rewriter.getI64Type(),                          // const long K_data
          rewriter.getI64Type(),                          // const long K_buffer
          rewriter.getI64Type()                           // const long N_buffer
      };
      auto functionType = FunctionType::get(op.getContext(), 
                                          inputTypes, 
                                          /*results=*/{}); // void返回类型
      
      // 在模块开始处创建函数声明
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<func::FuncOp>(
          module->getLoc(),
          "micro_kernel_asm_r6c96",  // 函数名
          functionType               // 函数类型
      ).setPrivate();               // 设置为私有
    }

    // 2. 提取所有参数
    SmallVector<Value, 6> callOperands;
    if (failed(extractMatmulParams(rewriter, loc, adaptor, callOperands))) {
      return failure();
    }

    // 3. 创建函数调用
    rewriter.create<func::CallOp>(
      loc,
      TypeRange{},              // 无返回值
      "micro_kernel_asm_r6c96", // callee
      callOperands);            // 参数列表

    // 4. 由于原操作没有返回值,直接擦除原操作即可
    rewriter.eraseOp(op);
    return success();
  }

private:
  // 提取 MatmulR6C96Op 所需的所有参数
  LogicalResult extractMatmulParams(
      ConversionPatternRewriter &rewriter, Location loc,
      OpAdaptor adaptor, SmallVectorImpl<Value> &callOperands) const {

    // 1. 提取 lhs (src_a) 相关参数
    Value lhsDesc = adaptor.getLhs();
    Value lhsAlignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, lhsDesc, ArrayRef<int64_t>{1});
    Value lhsOffset = rewriter.create<LLVM::ExtractValueOp>(loc, lhsDesc, ArrayRef<int64_t>{2});
    Value lhsPtr = rewriter.create<LLVM::GEPOp>(loc,
      rewriter.getType<LLVM::LLVMPointerType>(),
      rewriter.getF32Type(),
      lhsAlignedPtr,
      lhsOffset,
      /*inbounds=*/true);
    
    // 获取 K_data (lhs 的内层维度大小)
    Value kData = rewriter.create<LLVM::ExtractValueOp>(loc, lhsDesc, ArrayRef<int64_t>{3, 1});
    // 获取 K_buffer (lhs 的外层维度的 stride)
    Value kBuffer = rewriter.create<LLVM::ExtractValueOp>(loc, lhsDesc, ArrayRef<int64_t>{4, 0});

    // 2. 提取 rhs (src_b) 相关参数
    Value rhsDesc = adaptor.getRhs();
    Value rhsAlignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, rhsDesc, ArrayRef<int64_t>{1});
    Value rhsOffset = rewriter.create<LLVM::ExtractValueOp>(loc, rhsDesc, ArrayRef<int64_t>{2});
    Value rhsPtr = rewriter.create<LLVM::GEPOp>(loc,
      rewriter.getType<LLVM::LLVMPointerType>(),
      rewriter.getF32Type(),
      rhsAlignedPtr,
      rhsOffset,
      /*inbounds=*/true);
    
    // 获取 N_buffer (rhs 的外层维度的 stride)
    Value nBuffer = rewriter.create<LLVM::ExtractValueOp>(loc, rhsDesc, ArrayRef<int64_t>{4, 0});

    // 3. 提取 dst (dst_c) 相关参数
    Value dstDesc = adaptor.getDst();
    Value dstAlignedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, dstDesc, ArrayRef<int64_t>{1});
    Value dstOffset = rewriter.create<LLVM::ExtractValueOp>(loc, dstDesc, ArrayRef<int64_t>{2});
    Value dstPtr = rewriter.create<LLVM::GEPOp>(loc,
      rewriter.getType<LLVM::LLVMPointerType>(),
      rewriter.getF32Type(),
      dstAlignedPtr,
      dstOffset,
      /*inbounds=*/true);

    // 4. 按顺序添加所有参数
    callOperands.push_back(lhsPtr);    // src_a
    callOperands.push_back(rhsPtr);    // src_b
    callOperands.push_back(dstPtr);    // dst_c
    callOperands.push_back(kData);     // K_data
    callOperands.push_back(kBuffer);   // K_buffer
    callOperands.push_back(nBuffer);   // N_buffer

    return success();
  }
};
namespace {
struct MTDSPToLLVMConversionPass : public PassWrapper<MTDSPToLLVMConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MTDSPToLLVMConversionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final;
};
}

void MTDSPToLLVMConversionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         linalg::LinalgDialect, affine::AffineDialect,
                         arith::ArithDialect, memref::MemRefDialect, 
                         vector::VectorDialect, scf::SCFDialect,
                         LLVM::LLVMDialect>();
  target.addIllegalDialect<mtdsp::MTDSPDialect>();

  RewritePatternSet patterns(context);
  LowerToLLVMOptions options(&getContext());
  // options.useBarePtrCallConv = true;
  LLVMTypeConverter typeConverter(context, options);
  typeConverter.addTypeAttributeConversion(
    [](BaseMemRefType type, mtdsp::AddressSpaceAttr memorySpaceAttr) {
      mtdsp::AddressSpace memorySpace = memorySpaceAttr.getValue();
      unsigned addressSpace;
      switch (memorySpace) {
        // case mtdsp::AddressSpace::Global:
        //   addressSpace = 1;
        //   break;
        // case mtdsp::AddressSpace::Workgroup:
        //   addressSpace = 3;
        //   break;
        // case mtdsp::AddressSpace::Private:
        //   addressSpace = 5;
        //   break;
        default:
          addressSpace = 0;
      }
      return IntegerAttr::get(IntegerType::get(memorySpaceAttr.getContext(), 64), 
                              addressSpace);
    });
  patterns.add<
        ThreadIdOpLowering, 
        GroupSizeOpLowering,
        AllocOpLowering,
        DeallocOpLowering,
        DMAOpLowering,
        DMAOptOpLowering,
        WaitOpLowering,
        MatmulR6C96OpLowering
    >(
      typeConverter, 
      context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createMTDSPToLLVMConversionPass() {
  return std::make_unique<MTDSPToLLVMConversionPass>();
};