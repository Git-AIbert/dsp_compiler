module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op
  %add = transform.structured.match ops{["linalg.add"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op

  // 对matmul分块
  %matmul_m, %for_m = transform.structured.tile_using_for %matmul
    tile_sizes [576]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %matmul_m_k, %for_k = transform.structured.tile_using_for %matmul_m
    tile_sizes [0, 0, 512]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %A_ddr = transform.get_operand %matmul_m_k[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %A_ddr multi_buffer = true {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op

  %matmul_m_k_n, %for_n = transform.structured.tile_using_for %matmul_m_k
      tile_sizes [0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %B_ddr = transform.get_operand %matmul_m_k_n[1] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %B_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  %matmul_m_k_n_m, %for_m2 = transform.structured.tile_using_for %matmul_m_k_n
      tile_sizes [96]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %C_ddr = transform.get_operand %matmul_m_k_n_m[2] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  %C_result_am = transform.get_result %matmul_m_k_n_m[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_write %C_result_am, %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<global>} : (!transform.any_value, !transform.any_value) -> !transform.any_op

  %matmul_m_k_n_m_m, %for_m3 = transform.structured.tile_using_for %matmul_m_k_n_m
      tile_sizes [12]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %A_gsm = transform.get_operand %matmul_m_k_n_m_m[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %A_gsm multi_buffer = true {memory_space = #mtdsp.address_space<sm>} : (!transform.any_value) -> !transform.any_op

  %matmul_m_k_n_m_m_m, %for_m4 = transform.structured.tile_using_for %matmul_m_k_n_m_m
      tile_sizes [6]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  transform.mark_unroll %for_m4 unroll_factor = 2 : (!transform.any_op) -> !transform.any_op

  // 对add进行分块 (n128, m96)
  %add_n, %for2_n = transform.structured.extended_tile_using_for %add
      tile_sizes [0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %add_n_m, %for2_m = transform.structured.extended_tile_using_for %add_n
    tile_sizes [96]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %matmul_result_ddr = transform.get_operand %add_n_m[2] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %matmul_result_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  %add_result_am = transform.get_result %add_n_m[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_write %add_result_am, %matmul_result_ddr multi_buffer = true {memory_space = #mtdsp.address_space<global>} : (!transform.any_value, !transform.any_value) -> !transform.any_op

  %D_ddr = transform.get_operand %add_n_m[1] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %D_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  transform.yield
  }
}
