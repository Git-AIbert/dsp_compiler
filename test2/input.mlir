module {
  func.func @matmul_add_relu(%A: tensor<512x768xf32>,
                        %B: tensor<768x3072xf32>,
                        %D: tensor<512x3072xf32>,
                        %C: tensor<512x3072xf32>) -> tensor<512x3072xf32> {
    %matmul = linalg.matmul
        ins(%A, %B : tensor<512x768xf32>, tensor<768x3072xf32>)
        outs(%C : tensor<512x3072xf32>) -> tensor<512x3072xf32>

    %add = linalg.add
        ins(%matmul, %D : tensor<512x3072xf32>, tensor<512x3072xf32>)
        outs(%matmul : tensor<512x3072xf32>) -> tensor<512x3072xf32>

    %c0 = arith.constant 0.0 : f32
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      op_label = "relu"
    } ins(%add : tensor<512x3072xf32>)
      outs(%add : tensor<512x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<512x3072xf32>

    return %result : tensor<512x3072xf32>
  }
}
