module {
  func.func @matmul_add_relu(%A: tensor<2304x2048xf32>, 
                        %B: tensor<2048x1024xf32>,
                        %D: tensor<2304x1024xf32>,
                        %C: tensor<2304x1024xf32>) -> tensor<2304x1024xf32> {
    %matmul = linalg.matmul 
        ins(%A, %B : tensor<2304x2048xf32>, tensor<2048x1024xf32>) 
        outs(%C : tensor<2304x1024xf32>) -> tensor<2304x1024xf32>
    
    %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      op_label = "add"
    } ins(%matmul, %D : tensor<2304x1024xf32>, tensor<2304x1024xf32>)
      outs(%matmul : tensor<2304x1024xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %sum = arith.addf %in1, %in2 : f32
      linalg.yield %sum : f32
    } -> tensor<2304x1024xf32>

    %c0 = arith.constant 0.0 : f32
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      op_label = "relu"
    } ins(%add : tensor<2304x1024xf32>)
      outs(%add : tensor<2304x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<2304x1024xf32>

    return %result : tensor<2304x1024xf32>
  }
}