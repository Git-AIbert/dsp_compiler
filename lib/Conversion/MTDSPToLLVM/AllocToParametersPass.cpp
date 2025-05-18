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

class AllocToParametersPass 
    : public PassWrapper<AllocToParametersPass, OperationPass<func::FuncOp>> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* context = &getContext();
    
    // 步骤1：收集所有没有地址空间属性的mtdsp.alloc操作
    SmallVector<mtdsp::AllocOp, 4> allocsToReplace;
    funcOp.walk([&](mtdsp::AllocOp allocOp) {
      MemRefType memrefTy = allocOp.getType();
      mlir::Attribute attr = memrefTy.getMemorySpace();
      if (!attr) {
        // 这个alloc没有内存空间属性，需要被替换
        allocsToReplace.push_back(allocOp);
      }
    });
    
    if (allocsToReplace.empty())
      return;
    
    // 步骤2：创建带有额外参数的新函数类型
    FunctionType oldFuncType = funcOp.getFunctionType();
    SmallVector<Type, 8> newInputTypes(oldFuncType.getInputs().begin(), 
                                      oldFuncType.getInputs().end());
    
    // 使用映射来存储alloc操作与新参数之间的关系
    llvm::DenseMap<Operation*, unsigned> allocToParamIndex;
    unsigned baseParamIndex = oldFuncType.getInputs().size();
    
    for (auto allocOp : allocsToReplace) {
      newInputTypes.push_back(allocOp.getType());
      allocToParamIndex[allocOp] = baseParamIndex++;
    }
    
    FunctionType newFuncType = FunctionType::get(context, 
                                                newInputTypes, 
                                                oldFuncType.getResults());
    
    // 步骤3：更新函数签名
    funcOp.setType(newFuncType);
    
    // 步骤4：向入口块添加新的块参数
    Block& entryBlock = funcOp.getBody().front();
    for (auto allocOp : allocsToReplace) {
      entryBlock.addArgument(allocOp.getType(), allocOp.getLoc());
    }
    
    // 步骤5：用相应的函数参数替换每个alloc的使用
    for (auto allocOp : allocsToReplace) {
      unsigned paramIndex = allocToParamIndex[allocOp];
      Value newArg = entryBlock.getArgument(paramIndex);
      allocOp.getResult().replaceAllUsesWith(newArg);
      allocOp.erase();
    }
    
    // 步骤6：更新对该函数的所有调用点（如果在同一模块内）
    auto module = funcOp->getParentOfType<ModuleOp>();
    if (module) {
      SmallVector<func::CallOp, 4> callsToUpdate;
      module.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == funcOp.getName()) {
          callsToUpdate.push_back(callOp);
        }
      });
      
      // 这需要在每个调用点添加主机端分配逻辑，这超出了此Pass的范围
      if (!callsToUpdate.empty()) {
        funcOp.emitWarning() << "发现 " << callsToUpdate.size() 
                             << " 处调用点需要更新以匹配新的函数签名";
      }
    }
  }
};

// 向系统注册这个Pass
std::unique_ptr<mlir::Pass> mlir::createAllocToParametersPass() {
  return std::make_unique<AllocToParametersPass>();
}