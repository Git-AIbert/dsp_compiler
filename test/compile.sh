#!/bin/bash
set -e

M=${1:-3600}
K=${2:-2048}
N=${3:-128}

echo "Using matrix dimensions: M=$M, K=$K, N=$N"

python3 generate_linalg.py "$M" "$K" "$N"
python3 generate_transform.py schedule.mlir
rm -f kernel.ll
sh run_compiler.sh
