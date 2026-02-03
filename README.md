# Cascade

**A Multi-Level MLIR Compiler for DSP Processors**

## 项目介绍

Cascade是一个基于MLIR的DSP编译器，采用**计算调度分离**和**级联式编译架构**设计。编译器将算法定义（计算）与性能优化（调度）解耦：使用Linalg等标准方言描述算法逻辑，使用基于Transform方言扩展的Schedule方言独立描述优化策略。同一算法可应用不同调度策略而无需修改计算定义，类似Halide的设计理念。

完整编译流程为：**Linalg → Transform → Bufferization → MTDSP → LLVM**。编译器通过30+个Pass的顺序级联执行，将高级张量计算逐步lowering到MTDSP硬件指令，同时自动生成多级内存（DDR/GSM/AM/SM）间的数据搬运、多缓冲和DMA优化。

级联式编译架构体现在三个层面：

1. **编译Pass级联** - 30+个编译pass顺序执行，每个pass的输出是下一个pass的输入
2. **内存层次级联** - 数据在多级内存间流动：DDR → GSM → AM → SM，匹配DSP的层次化存储架构
3. **调度粒度级联** - 分块策略从粗到细逐层细化（如576→96→12→6），实现精细化的性能优化

编译器定义了两个自定义MLIR方言：**MTDSP方言**提供DSP硬件指令抽象，**Schedule方言**扩展Transform方言提供声明式调度操作。

## Schedule方言架构设计

Schedule方言采用了MLIR Transform方言的扩展机制。方言本身（[Schedule/IR/ScheduleOps.td](include/Dialect/Schedule/IR/ScheduleOps.td)）不包含任何操作定义，仅作为命名空间存在。所有的调度变换操作都定义在[Schedule/TransformOps/ScheduleTransformOps.td](include/Dialect/Schedule/TransformOps/ScheduleTransformOps.td)中，且这些操作属于Transform方言（使用`Op<Transform_Dialect, ...>`定义）。

这种设计遵循MLIR的标准范式：正如`mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td`为Linalg操作定义变换操作（如`transform.structured.tile_using_for`），我们的Schedule方言扩展Transform方言以提供面向张量计算的调度操作（如`transform.structured.cache_read`和`transform.mark_parallel`）。虽然这些操作在设计上是平台无关的，但它们的实现针对DSP编译的需求进行了定制。这些操作可以与MLIR内置的Transform操作无缝组合，共享统一的Handle传递语义和变换应用框架。

这种架构的优势在于复用Transform方言的基础设施（Handle系统、接口定义、应用框架），同时允许我们定义特定的变换操作。用户可以在单个变换序列中自由混合使用我们的自定义操作和标准MLIR变换操作，形成灵活的调度策略表达。

## 编译流程与调度实现

编译器的Transform调度操作分为两类，它们在编译流程的不同阶段发挥作用：

**立刻改变代码结构的变换操作**

这类操作在Transform应用阶段立刻修改IR结构，效果直接可见：

- `tile_using_for` - 生成嵌套的循环结构（`scf.for`）
- `fuse_eltwise_consumer` - 将element-wise消费者操作融合到生产者循环内
- `fuse_elementwise_generic` - 融合两个连续的element-wise generic操作
- `cache_read`/`cache_write` - 插入显式的数据拷贝操作（`linalg.copy`），此时仍为张量形式

**仅添加属性标记的操作**

这类操作在Transform阶段只在IR上添加属性标记，真正的实现发生在Bufferization之后的专门Pass中：

- `mark_parallel` → 添加并行标记 → `-parallel` Pass实现并行化
- `mark_unroll` → 添加展开标记 → `-unroll` Pass实现循环展开
- `multi_buffer`属性 → `-multi-buffer` Pass生成多缓冲和DMA操作

**完整编译流程**

1. **Transform应用阶段** - 在张量层面进行调度声明
   - 执行结构变换：分块、融合、插入缓存拷贝
   - 添加属性标记：parallel、unroll、multi_buffer等

2. **Bufferization阶段** - 从张量语义转换到内存语义
   - 张量转换为memref，同时保留内存空间属性（DDR/GSM/AM/SM）
   - 之前插入的`linalg.copy`变为实际的内存拷贝

3. **调度实现阶段** - 根据属性标记生成实际代码
   - Multi-buffer Pass：根据multi_buffer属性生成多缓冲和DMA传输
   - Optimize-DMA Pass：DMA通道分配、点对点同步优化
   - Parallel/Unroll Pass：实现并行化和循环展开
   - Buffer优化Pass：提升分配、合并冗余缓冲等

4. **方言转换阶段** - 逐步lowering到目标代码
   - Linalg/MemRef → MTDSP硬件指令抽象
   - MTDSP → LLVM IR
   - 最终生成可执行代码

这种分阶段的设计允许在高层（张量）进行调度决策，在低层（内存）进行实现优化，实现了关注点分离。

## 仓库结构
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
│   │       ├── IR/      // 方言定义（仅命名空间，不含操作）
│   │       ├── TransformOps/  // Transform方言扩展操作定义
│   │       └── Transforms/    // 编译Pass定义（多缓冲、DMA优化等）
│   └── Conversion/      // 方言转换passes
│       ├── LinalgToMTDSP/     // Linalg → MTDSP
│       ├── MemRefToMTDSP/     // MemRef → MTDSP
│       └── MTDSPToLLVM/       // MTDSP → LLVM
├── lib/                 // 源文件（与include结构对应）
├── examples/            // 端到端示例
│   ├── matmul/          // 矩阵乘法示例
│   ├── matmul_add/      // 矩阵乘加示例
│   ├── matmul_add_fuse/ // 矩阵乘加融合示例
│   └── ...              // 其他示例
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

查看[examples/matmul/compile.sh](examples/matmul/compile.sh)了解完整的编译流程：

```bash
cd examples/matmul
./compile.sh
```

该脚本演示了从MLIR输入到LLVM IR输出的完整编译过程，包括：
- 应用调度变换
- 张量缓冲化到多级内存空间
- 多缓冲和DMA优化
- 转换到MTDSP方言
- 最终lowering到LLVM IR
