module {
  func.func @matmul_add(%A: tensor<2304x2048xf32>, 
                        %B: tensor<2048x1024xf32>,
                        %D: tensor<2304x1024xf32>,
                        %C: tensor<2304x1024xf32>) -> tensor<2304x1024xf32> {
    %matmul = linalg.matmul 
        ins(%A, %B : tensor<2304x2048xf32>, tensor<2048x1024xf32>) 
        outs(%C : tensor<2304x1024xf32>) -> tensor<2304x1024xf32>
    %result = linalg.add 
        ins(%matmul, %D : tensor<2304x1024xf32>, tensor<2304x1024xf32>)
        outs(%matmul : tensor<2304x1024xf32>) -> tensor<2304x1024xf32>
    
    return %result : tensor<2304x1024xf32>
  }
}