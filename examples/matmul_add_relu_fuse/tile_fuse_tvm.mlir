module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
  %add = transform.structured.match ops{["linalg.add"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op
  %relu = transform.structured.match
      attributes {op_label = "relu"} in %arg0
      : (!transform.any_op) -> !transform.any_op

  // M 24
  %matmul_m, %for_m = transform.structured.tile_using_for %matmul
    tile_sizes [24]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // N 256
  %matmul_m_n, %for_n0 = transform.structured.tile_using_for %matmul_m
    tile_sizes [0, 256]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  transform.mark_parallel %for_n num_threads = 8 : (!transform.any_op) -> !transform.any_op

  // C -> C_am, C_am -> C
  %C_ddr = transform.get_operand %matmul_m_n[2] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %C_ddr {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  %C_result_am = transform.get_result %matmul_m_n[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_write %C_result_am, %C_ddr {memory_space = #mtdsp.address_space<ddr>} : (!transform.any_value, !transform.any_value) -> !transform.any_op

  // K 512
  %matmul_m_n_k, %for_k = transform.structured.tile_using_for %matmul_m_n
    tile_sizes [0, 0, 512]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // A -> A_sm, B -> B_am
  %A_ddr = transform.get_operand %matmul_m_n_k[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %A_ddr {memory_space = #mtdsp.address_space<sm>} : (!transform.any_value) -> !transform.any_op

  %B_ddr = transform.get_operand %matmul_m_n_k[1] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %B_ddr {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  // M 6
  %matmul_m_n_k_m, %for_m1 = transform.structured.tile_using_for %matmul_m_n_k
    tile_sizes [6]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // N 128
  %matmul_m_n_k_m_n, %for_n1 = transform.structured.tile_using_for %matmul_m_n_k_m
    tile_sizes [0, 128]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  transform.yield
  }
}
