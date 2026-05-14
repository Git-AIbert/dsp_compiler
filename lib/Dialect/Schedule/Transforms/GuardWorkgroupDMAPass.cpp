#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"

#include "Dialect/MTDSP/IR/MTDSPDialect.h"
#include "Dialect/Schedule/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_GUARDWORKGROUPDMA
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct GuardWorkgroupDMAPass
    : public impl::GuardWorkgroupDMABase<GuardWorkgroupDMAPass> {
  unsigned getAddressSpace(Value value) {
    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType)
      return static_cast<unsigned>(mtdsp::AddressSpace::Global);

    Attribute memorySpace = memrefType.getMemorySpace();
    if (!memorySpace)
      return static_cast<unsigned>(mtdsp::AddressSpace::Global);

    if (auto addressSpace =
            dyn_cast<mtdsp::AddressSpaceAttr>(memorySpace)) {
      return static_cast<unsigned>(addressSpace.getValue());
    }

    if (auto integerSpace = dyn_cast<IntegerAttr>(memorySpace))
      return integerSpace.getInt();

    return static_cast<unsigned>(mtdsp::AddressSpace::Global);
  }

  bool isGlobalToWorkgroup(Value src, Value dst) {
    return getAddressSpace(src) ==
               static_cast<unsigned>(mtdsp::AddressSpace::Global) &&
           getAddressSpace(dst) ==
               static_cast<unsigned>(mtdsp::AddressSpace::Workgroup);
  }

  Value getOrCreateThreadId(func::FuncOp funcOp) {
    Value threadId;
    funcOp.walk([&](mtdsp::ThreadIdOp threadIdOp) {
      threadId = threadIdOp.getResult();
      return WalkResult::interrupt();
    });
    if (threadId)
      return threadId;

    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    return builder.create<mtdsp::ThreadIdOp>(funcOp.getLoc()).getResult();
  }

  Value createTidIsZero(func::FuncOp funcOp, Value threadId) {
    OpBuilder builder(funcOp);
    if (Operation *defOp = threadId.getDefiningOp())
      builder.setInsertionPointAfter(defOp);
    else
      builder.setInsertionPointToStart(&funcOp.getBody().front());

    Location loc = funcOp.getLoc();
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    return builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, threadId, zero);
  }

  Value createZeroI32(func::FuncOp funcOp) {
    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    return builder.create<arith::ConstantIntOp>(funcOp.getLoc(), 0, 32);
  }

  void guardDMA(mtdsp::DMAOptOp dmaOp, Value isTidZero, Value zeroI32) {
    OpBuilder builder(dmaOp);
    Location loc = dmaOp.getLoc();

    auto ifOp = builder.create<scf::IfOp>(
        loc, TypeRange{dmaOp.getResult().getType()}, isTidZero,
        /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto guardedDMA = builder.create<mtdsp::DMAOptOp>(
        loc, dmaOp.getSrc(), dmaOp.getDst(), dmaOp.getChannel());
    guardedDMA->setAttrs(dmaOp->getAttrs());
    builder.create<scf::YieldOp>(loc, guardedDMA.getResult());

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, zeroI32);

    dmaOp.getResult().replaceAllUsesWith(ifOp.getResult(0));
    dmaOp.erase();
  }

  void guardWait(mtdsp::WaitP2POp waitOp, Value isTidZero) {
    OpBuilder builder(waitOp);
    Location loc = waitOp.getLoc();
    Value channel = waitOp.getChannel();
    NamedAttrList attrs(waitOp->getAttrs());

    builder.create<scf::IfOp>(
        loc, isTidZero,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          auto guardedWait =
              thenBuilder.create<mtdsp::WaitP2POp>(thenLoc, channel);
          guardedWait->setAttrs(attrs);
          thenBuilder.create<scf::YieldOp>(thenLoc);
        },
        nullptr);

    waitOp.erase();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<mtdsp::DMAOptOp> dmasToGuard;
    DenseSet<int64_t> guardedGroups;

    funcOp.walk([&](mtdsp::DMAOptOp dmaOp) {
      if (!isGlobalToWorkgroup(dmaOp.getSrc(), dmaOp.getDst()))
        return;

      dmasToGuard.push_back(dmaOp);
      if (auto groupAttr = dmaOp->getAttrOfType<IntegerAttr>("group"))
        guardedGroups.insert(groupAttr.getInt());
    });

    if (dmasToGuard.empty())
      return;

    Value threadId = getOrCreateThreadId(funcOp);
    Value isTidZero = createTidIsZero(funcOp, threadId);
    Value zeroI32 = createZeroI32(funcOp);

    for (mtdsp::DMAOptOp dmaOp : dmasToGuard)
      guardDMA(dmaOp, isTidZero, zeroI32);

    SmallVector<mtdsp::WaitP2POp> waitsToGuard;
    funcOp.walk([&](mtdsp::WaitP2POp waitOp) {
      auto groupAttr = waitOp->getAttrOfType<IntegerAttr>("group");
      if (groupAttr && guardedGroups.contains(groupAttr.getInt()))
        waitsToGuard.push_back(waitOp);
    });

    for (mtdsp::WaitP2POp waitOp : waitsToGuard)
      guardWait(waitOp, isTidZero);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createGuardWorkgroupDMAPass() {
  return std::make_unique<GuardWorkgroupDMAPass>();
}
