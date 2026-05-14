#!/bin/bash
../build/bin/cascade-opt input.mlir \
  -transform-preload-library="transform-library-paths=transform.mlir" \
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
  # -unroll \
  # -canonicalize -cse -canonicalize \
  # -convert-linalg-to-mtdsp \
  # -convert-memref-to-mtdsp \
  # -canonicalize -cse -canonicalize \
  # -promote-allocs-to-arguments \
  # -canonicalize -cse -canonicalize \
  # -convert-mtdsp-to-llvm \
  # -remove-memref-address-space \
  # -canonicalize -cse -canonicalize \
  # -expand-strided-metadata \
  # -convert-linalg-to-loops \
  # -lower-affine \
  # -convert-scf-to-cf \
  # -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
  # -finalize-memref-to-llvm \
  # -convert-cf-to-llvm \
  # -convert-arith-to-llvm \
  # -convert-index-to-llvm \
  # -reconcile-unrealized-casts \
  # -canonicalize -cse -canonicalize \
  # > output.mlir 2> output.txt
