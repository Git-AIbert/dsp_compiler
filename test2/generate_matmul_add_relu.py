#!/usr/bin/env python3
"""
Generate an MLIR module for a fused matmul + add + relu kernel.

Usage:
    python3 generate_matmul_add_relu.py M K N [output_file]
"""

from __future__ import annotations

import sys
from pathlib import Path


def generate_matmul_add_relu_ir(
    m: int, k: int, n: int, output_file: Path = Path("input.mlir")
) -> None:
    ir = f"""module {{
  func.func @matmul_add_relu(%A: tensor<{m}x{k}xf32>,
                        %B: tensor<{k}x{n}xf32>,
                        %D: tensor<{m}x{n}xf32>,
                        %C: tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
    %matmul = linalg.matmul
        ins(%A, %B : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%C : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>

    %add = linalg.add
        ins(%matmul, %D : tensor<{m}x{n}xf32>, tensor<{m}x{n}xf32>)
        outs(%matmul : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>

    %c0 = arith.constant 0.0 : f32
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      op_label = "relu"
    }} ins(%add : tensor<{m}x{n}xf32>)
      outs(%add : tensor<{m}x{n}xf32>) {{
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    }} -> tensor<{m}x{n}xf32>

    return %result : tensor<{m}x{n}xf32>
  }}
}}
"""

    output_file.write_text(ir, encoding="utf-8")
    print(
        f"Generated {output_file} with dimensions: "
        f"A({m}x{k}) * B({k}x{n}) + D({m}x{n}) = C({m}x{n})"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("Usage: python3 generate_matmul_add_relu.py M K N [output_file]")
        print("Example: python3 generate_matmul_add_relu.py 2304 2048 1024")
        return 1

    m, k, n = (int(argv[1]), int(argv[2]), int(argv[3]))
    output_file = Path(argv[4]) if len(argv) > 4 else Path("input.mlir")
    generate_matmul_add_relu_ir(m, k, n, output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
