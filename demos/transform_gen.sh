#!/bin/bash

../build/bin/mtir-opt \
    "$1" \
    --transform-preload-library="transform-library-paths=$2" \
    --transform-interpreter \
    --canonicalize \
    --cse \
    --staticize-tensor-empty \
    --one-shot-bufferize-with-memory-space='bufferize-function-boundaries=1' \
    --canonicalize \
    --cse \
    --buffer-hoisting \
    --buffer-loop-hoisting \
    --canonicalize \
    --cse \
    --parallel \
    --canonicalize \
    --cse \
    --multi-buffer \
    --canonicalize \
    --cse \
    --unroll \
    --canonicalize \
    --cse \
    --convert-linalg-to-mtdsp \
    --convert-memref-to-mtdsp \
    --canonicalize \
    --cse \
    --promote-allocs-to-arguments \
    --canonicalize \
    --cse \
    --convert-mtdsp-to-llvm \
    --remove-memref-address-space \
    --canonicalize \
    --cse \
    --expand-strided-metadata \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
    --finalize-memref-to-llvm \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --reconcile-unrealized-casts \
    --canonicalize \
    --cse \
  > output.mlir

# 翻译MLIR到LLVM IR
mlir-translate --mlir-to-llvmir output.mlir > output_tmp.ll

# 为函数添加.global section属性，为全局变量添加.gsm section属性
sed '
/^define .*{$/ {
    s/{$/ section ".global" {/
}
/^@[^ ]* = .*global / {
    s/global /global /
    / section /! s/$/, section ".gsm"/
}
' output_tmp.ll > output.ll

rm output_tmp.ll