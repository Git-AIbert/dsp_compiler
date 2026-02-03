../../build/bin/cascade-opt \
    matmul_add.mlir \
    -transform-preload-library="transform-library-paths=tile.mlir" \
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
  > output.mlir

# 翻译MLIR到LLVM IR
mlir-translate -mlir-to-llvmir output.mlir > output_tmp.ll

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