#!/bin/bash
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_FILE="${SCRIPT_DIR}/input.mlir"
TRANSFORM_FILE="${SCRIPT_DIR}/transform.mlir"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
OUTPUT_MLIR="${SCRIPT_DIR}/output.mlir"
OUTPUT_TMP_LL="${SCRIPT_DIR}/output_tmp.ll"
OUTPUT_LL="${SCRIPT_DIR}/kernel.ll"

LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-/opt/llvm-19}"
# LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-${HOME}/albert/opt/llvm-19}"
CASCADE_OPT="${CASCADE_OPT:-${WORKSPACE_DIR}/build/bin/cascade-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-${LLVM_INSTALL_PREFIX}/bin/mlir-translate}"

if [ ! -x "$CASCADE_OPT" ]; then
    echo "❌ 错误: 找不到 cascade-opt: $CASCADE_OPT"
    echo "   请先在项目根目录执行: cmake -S . -B build-native && cmake --build build-native --target cascade-opt"
    exit 1
fi

if [ ! -x "$MLIR_TRANSLATE" ]; then
    echo "❌ 错误: 找不到 mlir-translate: $MLIR_TRANSLATE"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 找不到输入文件: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$TRANSFORM_FILE" ]; then
    echo "❌ 错误: 找不到 transform 文件: $TRANSFORM_FILE"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
rm -f "$OUTPUT_MLIR" "$OUTPUT_TMP_LL" "$OUTPUT_LL"
cd "$WORKSPACE_DIR"

if (
    "$CASCADE_OPT" \
        "$INPUT_FILE" \
        -transform-preload-library="transform-library-paths=${TRANSFORM_FILE}" \
        -transform-interpreter \
        -custom-canonicalize -cse -custom-canonicalize \
        -staticize-tensor-empty \
        -one-shot-bufferize-with-memory-space='bufferize-function-boundaries=1' \
        -custom-canonicalize -cse -custom-canonicalize \
        -expand-realloc \
        -custom-canonicalize \
        -ownership-based-buffer-deallocation \
        -custom-canonicalize \
        -buffer-deallocation-simplification \
        -bufferization-lower-deallocations \
        -custom-canonicalize -cse -custom-canonicalize \
        -buffer-hoisting \
        -buffer-loop-hoisting \
        -buffer-loop-sinking \
        -remove-function-returns \
        -custom-canonicalize -cse -custom-canonicalize \
        -parallel \
        -custom-canonicalize -cse -custom-canonicalize \
        -multi-buffer \
        -optimize-dma \
        -custom-canonicalize -cse -custom-canonicalize \
        -deduplicate-multi-buffer \
        -custom-canonicalize -cse -custom-canonicalize \
        -chain-split-reduction-pipelines \
        -guard-workgroup-dma \
        -custom-canonicalize -cse -custom-canonicalize \
        -unroll \
        -canonicalize -cse -canonicalize \
        -convert-linalg-to-mtdsp \
        -convert-memref-to-mtdsp \
        -canonicalize -cse -canonicalize \
        -promote-allocs-to-arguments \
        -canonicalize -cse -canonicalize \
        -convert-mtdsp-to-llvm \
        -remove-memref-address-space \
        -canonicalize -cse -canonicalize \
        -expand-strided-metadata \
        -convert-linalg-to-loops \
        -lower-affine \
        -convert-scf-to-cf \
        -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
        -finalize-memref-to-llvm \
        -convert-cf-to-llvm \
        -convert-arith-to-llvm \
        -convert-index-to-llvm \
        -reconcile-unrealized-casts \
        -canonicalize -cse -canonicalize \
        > "$OUTPUT_MLIR" || exit $?

    if [ ! -s "$OUTPUT_MLIR" ]; then
        echo "error: output MLIR was not generated or is empty" >&2
        exit 1
    fi

    "$MLIR_TRANSLATE" -mlir-to-llvmir "$OUTPUT_MLIR" > "$OUTPUT_TMP_LL" || exit $?

    if [ ! -s "$OUTPUT_TMP_LL" ]; then
        echo "error: temporary LLVM IR was not generated or is empty" >&2
        exit 1
    fi

    sed '
/^define .*{$/ {
    s/{$/ section ".global" {/
}
/^@[^ ]* = .*global / {
    s/global /global /
    / section /! s/$/, section ".gsm"/
}
' "$OUTPUT_TMP_LL" > "$OUTPUT_LL"

    if [ ! -s "$OUTPUT_LL" ]; then
        echo "error: LLVM IR was not generated or is empty" >&2
        exit 1
    fi

    rm -f "$OUTPUT_TMP_LL"
) > "$OUTPUT_FILE" 2>&1; then
    :
else
    status=$?
    echo "错误: compiler 执行失败，退出码: $status" >&2
    echo "日志文件: $OUTPUT_FILE" >&2
    echo "前 40 行日志:" >&2
    sed -n '1,40p' "$OUTPUT_FILE" >&2
    exit "$status"
fi

echo "✅ compiler 执行成功"
echo "📄 日志文件已生成: $OUTPUT_FILE"
echo "📄 MLIR 文件已生成: $OUTPUT_MLIR"
echo "📄 LLVM IR 文件已生成: $OUTPUT_LL"
