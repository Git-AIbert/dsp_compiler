module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // K 512
    %matmul_M_N_K512, %K512 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 512] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // M 576 A->A_gsm(multi_buffer)
    %matmul_M576_N_K512, %M576 = transform.structured.tile_using_for %matmul_M_N_K512 tile_sizes [576, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %A_ddr = transform.get_operand %matmul_M576_N_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %A_ddr multi_buffer = true {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op
    // N 128 B->B_am(multi_buffer)
    %matmul_M576_N128_K512, %N128 = transform.structured.tile_using_for %matmul_M576_N_K512 tile_sizes [0, 128, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %B_ddr = transform.get_operand %matmul_M576_N128_K512[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %B_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op
    // M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
    %matmul_M144_N128_K512, %M144 = transform.structured.tile_using_for %matmul_M576_N128_K512 tile_sizes [144, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %C_ddr = transform.get_operand %matmul_M144_N128_K512[2] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op
    %C_result_am = transform.get_result %matmul_M144_N128_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_write %C_result_am, %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<ddr>} : (!transform.any_value, !transform.any_value) -> !transform.any_op
    // M 12 A_gsm->A_sm(multi_buffer)
    %matmul_M12_N128_K512, %M12 = transform.structured.tile_using_for %matmul_M144_N128_K512 tile_sizes [12, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %A_gsm = transform.get_operand %matmul_M12_N128_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %A_gsm multi_buffer = true {memory_space = #mtdsp.address_space<sm>} : (!transform.any_value) -> !transform.any_op
    // M 6 unroll(2)
    %matmul_M6_N128_K512, %M6 = transform.structured.tile_using_for %matmul_M12_N128_K512 tile_sizes [6, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.mark_unroll %M6 unroll_factor = 2 : (!transform.any_op) -> !transform.any_op
  }
}