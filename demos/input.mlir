module {
  func.func @matmul(%arg0: tensor<1920x2048xf32>, %arg1: tensor<2048x1920xf32>, %arg2: tensor<1920x1920xf32>) -> tensor<1920x1920xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1920x2048xf32>, tensor<2048x1920xf32>) outs(%arg2 : tensor<1920x1920xf32>) -> tensor<1920x1920xf32>
    return %0 : tensor<1920x1920xf32>
  }
}