module {
  func.func @matmul(%A: tensor<2048x2304xf32>, 
                    %B: tensor<2048x1024xf32>,
                    %C: tensor<2304x1024xf32>) -> tensor<2304x1024xf32> {
    %init = tensor.empty() : tensor<2304x2048xf32>
    
    %A_t = linalg.transpose
      ins(%A : tensor<2048x2304xf32>)
      outs(%init : tensor<2304x2048xf32>)
      permutation = [1, 0]

    %matmul = linalg.matmul 
        ins(%A_t, %B : tensor<2304x2048xf32>, tensor<2048x1024xf32>) 
        outs(%C : tensor<2304x1024xf32>) -> tensor<2304x1024xf32>
    
    return %matmul : tensor<2304x1024xf32>
  }
}