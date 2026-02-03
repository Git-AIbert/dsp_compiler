# Cascade

**A Multi-Level MLIR Compiler for DSP Processors**

## 项目介绍

Cascade是一个基于MLIR的DSP编译器，采用**计算调度分离**和**级联式编译架构**设计。编译器将算法定义（计算）与性能优化（调度）解耦，通过自定义的Schedule方言实现灵活的调度策略表达，再经由多层次的渐进式转换，将高级张量计算（Linalg）逐步lowering到MTDSP硬件指令。

级联式编译架构体现在三个层面：

1. **编译Pass级联** - 30+个编译pass顺序执行，每个pass的输出是下一个pass的输入
2. **内存层次级联** - 数据在多级内存间流动：DDR → GSM → AM → SM，匹配DSP的层次化存储架构
3. **调度粒度级联** - 分块策略从粗到细逐层细化（如576→96→12→6），实现精细化的性能优化

## 主要特性

- **计算调度分离（Decoupling Computation and Scheduling）**
  - **计算（What）**：使用Linalg等标准方言描述算法逻辑
  - **调度（How）**：使用Schedule方言独立描述优化策略
  - 同一算法可应用不同调度策略，无需修改计算定义
  - 类似Halide的设计理念，提供更好的可组合性和可维护性

- **级联式编译架构**
  - 完整编译流程：Linalg → Transform → Bufferization → MTDSP → LLVM
  - Schedule方言驱动的声明式调度变换
  - 30+个编译pass顺序级联执行

- **多级内存层次优化**
  - 自动生成DDR、GSM、AM、SM四级内存间的数据搬运
  - 多缓冲优化（Multi-buffering）
  - DMA传输优化

- **自定义MLIR方言**
  - **MTDSP方言**：DSP特定的操作和指令
  - **Schedule方言**：调度和变换操作（分块、缓存、并行等）

## 🔍 仓库结构
Cascade编译器关键目录如下所示：
```
├── cascade-opt/         // 编译器工具
│   └── cascade-opt.cpp  // 编译器主入口
├── include/             // 头文件
│   ├── Dialect/         // 自定义方言定义
│   │   ├── MTDSP/       // MTDSP方言：DSP硬件指令抽象
│   │   │   ├── IR/      // 方言定义（操作、类型、属性）
│   │   │   └── Transforms/  // MTDSP内部优化变换
│   │   └── Schedule/    // Schedule方言：调度策略表达
│   │       ├── IR/      // 调度操作定义
│   │       ├── TransformOps/  // Transform接口实现
│   │       └── Transforms/    // 调度变换实现
│   └── Conversion/      // 方言转换passes
│       ├── LinalgToMTDSP/     // Linalg → MTDSP
│       ├── MemRefToMTDSP/     // MemRef → MTDSP内存操作
│       └── MTDSPToLLVM/       // MTDSP → LLVM IR
├── lib/                 // 源文件（与include结构对应）
├── examples/            // 端到端示例
│   ├── matmul/          // 矩阵乘法示例
│   ├── matmul_add/      // 矩阵乘加示例
│   └── matmul_add_fuse/ // 矩阵乘加融合示例
├── docs/                // 文档目录
├── CMakeLists.txt
└── README.md
```

## 构建指南

**前置要求：**
- CMake 3.20+
- C++17编译器
- LLVM/MLIR 19

**配置LLVM路径**

在[CMakeLists.txt](CMakeLists.txt#L10-L11)中配置你的LLVM/MLIR安装路径：

```cmake
set(LLVM_DIR "/path/to/your/llvm/lib/cmake/llvm")
set(MLIR_DIR "/path/to/your/llvm/lib/cmake/mlir")
```

**编译项目**

```bash
mkdir build && cd build
cmake ..
make
```

## 使用示例

### 计算调度分离示例

**1. 定义计算（使用Linalg）**
```mlir
// 矩阵乘法的计算定义 - 描述"做什么"
func.func @matmul(%A: tensor<2304x1024xf32>, %B: tensor<1024x1536xf32>)
    -> tensor<2304x1536xf32> {
  %C = linalg.matmul ins(%A, %B: tensor<2304x1024xf32>, tensor<1024x1536xf32>)
                     outs(%C_init: tensor<2304x1536xf32>) -> tensor<2304x1536xf32>
  return %C : tensor<2304x1536xf32>
}
```

**2. 定义调度（使用Schedule方言）**
```mlir
// 调度策略 - 描述"怎么做"（分块、缓存、内存层次）
transform.structured.tile_using_for %matmul tile_sizes [576]      // M维度分块
transform.structured.tile_using_for %matmul tile_sizes [0, 0, 512] // K维度分块
transform.structured.cache_read %A {memory_space = #mtdsp.address_space<gsm>} // 缓存到GSM
// ... 更多调度策略见 examples/matmul/tile.mlir
```

### 完整编译流程

查看[examples/matmul/test.sh](examples/matmul/test.sh)了解完整的编译流程：

```bash
cd examples/matmul
./test.sh
```

该脚本演示了从MLIR输入到LLVM IR输出的完整编译过程，包括：
- 应用Transform/Schedule调度策略
- 张量缓冲化到多级内存空间
- 多缓冲和DMA优化
- 转换到MTDSP方言
- 最终lowering到LLVM IR
