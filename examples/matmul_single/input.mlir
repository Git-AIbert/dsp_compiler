module {
  func.func @matmul(%arg0: tensor<1152x1024xf32>, %arg1: tensor<1024x128xf32>, %arg2: tensor<1152x128xf32>) -> tensor<1152x128xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1152x1024xf32>, tensor<1024x128xf32>) outs(%arg2 : tensor<1152x128xf32>) -> tensor<1152x128xf32>
    return %0 : tensor<1152x128xf32>
  }
}