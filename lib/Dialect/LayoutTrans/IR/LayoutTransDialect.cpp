#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/LayoutTrans/IR/LayoutTransDialect.h"

using namespace mlir;
using namespace mlir::layout_trans;
using namespace mlir::bufferization;

#include "Dialect/LayoutTrans/IR/LayoutTransDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LayoutTransDialect
//===----------------------------------------------------------------------===//

// 实现初始化函数
void LayoutTransDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/LayoutTrans/IR/LayoutTransOps.cpp.inc"
      >();
}

// 通用模板实现
template <typename OpTy>
struct LayoutTransOpInterface
    : public DstBufferizableOpInterfaceExternalModel<LayoutTransOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                             const AnalysisState &state) const {
    // 使用DestinationStyleOpInterface的接口方法来确定是否是输入操作数
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    return !dstOp.isDpsInit(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // 使用DestinationStyleOpInterface的接口方法来确定是否是输出操作数
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    return dstOp.isDpsInit(&opOperand);
  }

  // 元素访问分析方法
  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                    ArrayRef<OpOperand *> opOperands) const {
    // 对于布局转换操作，通常不是元素级访问
    // 因为它们重排了元素，所以返回false更安全
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                         const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    
    auto layoutOp = cast<OpTy>(op);
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    
    // 获取所有输入操作数的缓冲区
    SmallVector<Value> newInputBuffers;
    // 遍历操作的所有输入操作数
    for (OpOperand *opOperand : dstOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
      if (failed(buffer))
        return op->emitError() << "无法获取输入操作数的缓冲区";
      newInputBuffers.push_back(*buffer);
    }
    
    // 获取所有输出操作数的缓冲区
    SmallVector<Value> newOutputBuffers;
    // 遍历操作的所有结果
    for (OpResult opResult : op->getOpResults()) {
      // 获取与该结果对应的输出操作数，在目标样式操作中，每个结果都对应一个初始化操作数
      OpOperand *opOperand = dstOp.getDpsInitOperand(opResult.getResultNumber());
      FailureOr<Value> resultBuffer = getBuffer(rewriter, opOperand->get(), options);
      if (failed(resultBuffer))
        return op->emitError() << "无法获取输出操作数的缓冲区";
      newOutputBuffers.push_back(*resultBuffer);
    }
    
    // 合并输入和输出操作数
    SmallVector<Value> newOperands = newInputBuffers;
    newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());
    
    // 创建新操作
    OperationState state(op->getLoc(), op->getName(), newOperands, TypeRange{},
                        op->getAttrs());
    
    // 创建并插入新操作
    Operation *newOp = Operation::create(state);
    rewriter.insert(newOp);
    
    // 替换原操作的结果
    replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);
    
    return success();
  }
};

// 辅助结构，迭代注册所有操作
template <typename... Ops>
struct LayoutTransOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    // 使用折叠表达式注册所有操作
    (Ops::template attachInterface<LayoutTransOpInterface<Ops>>(*ctx), ...);
  }
};

// 注册函数
void layout_trans::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, layout_trans::LayoutTransDialect *dialect) {
    // 注册所有需要的操作
    LayoutTransOpInterfaceHelper<
      layout_trans::Img2ColOp,
      layout_trans::Col2ImgOp
    >::registerOpInterface(ctx);

    // 等价于以下
    // layout_trans::Img2ColOp::attachInterface<LayoutTransOpInterface<layout_trans::Img2ColOp>>(*ctx);
    // layout_trans::Col2ImgOp::attachInterface<LayoutTransOpInterface<layout_trans::Col2ImgOp>>(*ctx);
  });
}

//===----------------------------------------------------------------------===//
// LayoutTrans Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Img2ColOp
//===----------------------------------------------------------------------===//

void Img2ColOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // 为输入添加读取效果
  effects.emplace_back(
      MemoryEffects::Read::get(), 
      &getOperation()->getOpOperand(0),  // 输入操作数
      /*stage=*/0,
      /*effectOnFullRegion=*/true, 
      SideEffects::DefaultResource::get());
  
  // 为输出添加写入效果
  effects.emplace_back(
      MemoryEffects::Write::get(), 
      &getOperation()->getOpOperand(1),  // 输出操作数
      /*stage=*/0,
      /*effectOnFullRegion=*/true, 
      SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Col2ImgOp
//===----------------------------------------------------------------------===//

void Col2ImgOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // 为输入添加读取效果
  effects.emplace_back(
      MemoryEffects::Read::get(), 
      &getOperation()->getOpOperand(0),  // 输入操作数
      /*stage=*/0,
      /*effectOnFullRegion=*/true, 
      SideEffects::DefaultResource::get());
  
  // 为输出添加写入效果
  effects.emplace_back(
      MemoryEffects::Write::get(), 
      &getOperation()->getOpOperand(1),  // 输出操作数
      /*stage=*/0,
      /*effectOnFullRegion=*/true, 
      SideEffects::DefaultResource::get());
}

#define GET_OP_CLASSES
#include "Dialect/LayoutTrans/IR/LayoutTransOps.cpp.inc"