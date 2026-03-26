module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op
  %transpose = transform.structured.match ops{["linalg.transpose"]} in %arg0 
      : (!transform.any_op) -> !transform.any_op

  // 对matmul分块
  %matmul_m_k, %for_m, %for_k = transform.structured.tile_using_for %matmul
    tile_sizes [576, 0, 512]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

  %transpose_m, %for_m2 = transform.structured.fuse_into_containing_op %transpose into %for_m
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %transpose_m_k, %for_k2 = transform.structured.fuse_into_containing_op %transpose_m into %for_k
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %A_ddr = transform.get_operand %transpose_m_k[0] : (!transform.any_op) -> !transform.any_value
  transform.structured.cache_read %A_ddr multi_buffer = false {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op

  %transpose_m_k_gsm = transform.structured.set_dps_init_memory_space %transpose_m_k
    init_index = 0 {memory_space = #mtdsp.address_space<gsm>}
    : (!transform.any_op) -> !transform.any_op

  // %A_ddr = transform.get_operand %transpose_m_k[1] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_read %A_ddr multi_buffer = false {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op

  // %matmul_m_k_n, %for_n = transform.structured.tile_using_for %matmul_m_k
  //     tile_sizes [0, 128]
  //     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // transform.mark_parallel %for_n num_threads = 8 : (!transform.any_op) -> !transform.any_op

  // %B_ddr = transform.get_operand %matmul_m_k_n[1] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_read %B_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  // %matmul_m_k_n_m, %for_m2 = transform.structured.tile_using_for %matmul_m_k_n
  //     tile_sizes [96]  // %A_ddr = transform.get_operand %matmul_m_k[0] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_read %A_ddr multi_buffer = true {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op

  //     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // %C_ddr = transform.get_operand %matmul_m_k_n_m[2] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_read %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op

  // %C_result_am = transform.get_result %matmul_m_k_n_m[0] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_write %C_result_am, %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<global>} : (!transform.any_value, !transform.any_value) -> !transform.any_op

  // %matmul_m_k_n_m_m, %for_m3 = transform.structured.tile_using_for %matmul_m_k_n_m
  //     tile_sizes [12]
  //     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // transform.mark_unroll %for_m3 unroll_factor = 2 : (!transform.any_op) -> !transform.any_op

  // %A_gsm = transform.get_operand %matmul_m_k_n_m_m[0] : (!transform.any_op) -> !transform.any_value
  // transform.structured.cache_read %A_gsm multi_buffer = true {memory_space = #mtdsp.address_space<sm>} : (!transform.any_value) -> !transform.any_op

  // %matmul_m_k_n_m_m_m, %for_m4 = transform.structured.tile_using_for %matmul_m_k_n_m_m
  //     tile_sizes [6]
  //     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // transform.mark_unroll %for_m4 unroll_factor = 2 : (!transform.any_op) -> !transform.any_op

  transform.yield
  }
}
