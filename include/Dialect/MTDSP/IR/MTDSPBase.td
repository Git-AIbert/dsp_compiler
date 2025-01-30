#ifndef MTDSP_BASE
#define MTDSP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// MTDSP Dialect
//===----------------------------------------------------------------------===//

def MTDSP_Dialect : Dialect {
  let name = "mtdsp";
  let cppNamespace = "::mlir::mtdsp";
  // let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Returns the numeric value used to identify the global memory address
    /// space.
    static AddressSpace getGlobalAddressSpace() { return AddressSpace::Global; }

    /// Returns the numeric value used to identify the workgroup memory address
    /// space.
    static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }

    /// Returns the numeric value used to identify the private memory address
    /// space.
    static AddressSpace getScalarAddressSpace() { return AddressSpace::Scalar; }
    static AddressSpace getVectorAddressSpace() { return AddressSpace::Vector; }
  }];

  // let dependentDialects = ["arith::ArithDialect"];
  let useDefaultAttributePrinterParser = 1;
  // let useDefaultTypePrinterParser = 1;
  // let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// MTDSP Enums
//===----------------------------------------------------------------------===//
// I32EnumAttrCase (基础枚举值)
//        ↓
// AddressSpaceGlobal/Workgroup/Private (具体枚举值)
//        ↓
// MTDSP_I32Enum (自定义枚举类型基类)
//        ↓
// AddressSpaceEnum (具体枚举类型)
//        ↓
// MTDSP_I32EnumAttr (自定义属性基类)
//        ↓
// AddressSpaceAttr (最终属性定义)

def AddressSpaceGlobal : I32EnumAttrCase<"Global", 0, "global">;
def AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 1, "gsm">;
def AddressSpaceScalar : I32EnumAttrCase<"Scalar", 2, "sm">;
def AddressSpaceVector : I32EnumAttrCase<"Vector", 3, "am">;

class MTDSP_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtdsp";
}

def AddressSpaceEnum : MTDSP_I32Enum<
  "AddressSpace", "MTDSP address space", [
    AddressSpaceGlobal,
    AddressSpaceWorkgroup,
    AddressSpaceScalar,
    AddressSpaceVector
  ]>;

class MTDSP_I32EnumAttr<string mnemonic, MTDSP_I32Enum enumInfo> :
    EnumAttr<MTDSP_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def AddressSpaceAttr :
  MTDSP_I32EnumAttr<"address_space", AddressSpaceEnum>;


#endif // MTDSP_BASE