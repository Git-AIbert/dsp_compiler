#!/bin/bash
set -e

M=${1:-1152}
K=${2:-1024}
N=${3:-1024}

echo "Using matrix dimensions: M=$M, K=$K, N=$N"

python3 generate_linalg.py "$M" "$K" "$N"
python3 generate_transform.py schedule.mlir
rm -f kernel.ll
sh run_compiler.sh
