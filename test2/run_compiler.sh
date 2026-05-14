#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_FILE="${SCRIPT_DIR}/input.mlir"
TRANSFORM_FILE="${SCRIPT_DIR}/tile_fuse.mlir"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
OUTPUT_MLIR="${SCRIPT_DIR}/output.mlir"
OUTPUT_TMP_LL="${SCRIPT_DIR}/output_tmp.ll"
OUTPUT_LL="${SCRIPT_DIR}/kernel.ll"

LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-${HOME}/albert/opt/llvm-19}"
CASCADE_OPT="${CASCADE_OPT:-${WORKSPACE_DIR}/build/bin/cascade-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-${LLVM_INSTALL_PREFIX}/bin/mlir-translate}"

if [ ! -x "$CASCADE_OPT" ]; then
    echo "error: cannot find cascade-opt: $CASCADE_OPT" >&2
    exit 1
fi

if [ ! -x "$MLIR_TRANSLATE" ]; then
    echo "error: cannot find mlir-translate: $MLIR_TRANSLATE" >&2
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "error: cannot find input file: $INPUT_FILE" >&2
    exit 1
fi

if [ ! -f "$TRANSFORM_FILE" ]; then
    echo "error: cannot find transform file: $TRANSFORM_FILE" >&2
    exit 1
fi

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
        -staticize-dynamic-tile-alloc \
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
        -convert-mtdsp-to-llvm='matmul-micro-kernel-fn=micro_kernel_asm_r6c128' \
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
    echo "error: compiler failed with status $status" >&2
    echo "log file: $OUTPUT_FILE" >&2
    echo "first 40 log lines:" >&2
    sed -n '1,40p' "$OUTPUT_FILE" >&2
    exit "$status"
fi

echo "compiler succeeded"
echo "log: $OUTPUT_FILE"
echo "mlir: $OUTPUT_MLIR"
echo "llvm ir: $OUTPUT_LL"
