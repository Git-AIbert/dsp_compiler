#!/usr/bin/env python3
"""
MLIR 矩阵乘法 IR 生成器
用法: python generate_mlir.py M N K [output_file]
"""

import sys

def generate_matmul_ir(M, N, K, output_file="matmul.mlir"):
    ir_template = f"""module {{
  func.func @matmul(%arg0: tensor<{M}x{K}xf32>, %arg1: tensor<{K}x{N}xf32>, %arg2: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%arg2 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
    return %0 : tensor<{M}x{N}xf32>
  }}
}}"""
    
    with open(output_file, 'w') as f:
        f.write(ir_template)
    
    print(f"Generated {output_file} with dimensions: A({M}x{K}) * B({K}x{N}) = C({M}x{N})")

if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("Usage: python generate_mlir.py M K N [output_file]")
    #     print("Example: python generate_mlir.py 640 1920 2048")
    #     sys.exit(1)
    
    # M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    # output_file = sys.argv[4] if len(sys.argv) > 4 else "matmul.mlir"

    M, K, N = 1920, 2048, 1920
    output_file = "demos/matmul.mlir"
    
    generate_matmul_ir(M, N, K, output_file)