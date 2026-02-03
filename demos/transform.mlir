module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // M 960
    %matmul_M960_N_K, %M960 = transform.structured.tile_using_for %0 tile_sizes [960, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // K 512 A->A_gsm(multi_buffer)
    %matmul_M960_N_K512, %K512 = transform.structured.tile_using_for %matmul_M960_N_K tile_sizes [0, 0, 512] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %A_ddr = transform.get_operand %matmul_M960_N_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %A_ddr multi_buffer = true {memory_space = #mtdsp.address_space<gsm>} : (!transform.any_value) -> !transform.any_op
    // N 96  parallel(8) B->B_am(multi_buffer)
    %matmul_M960_N96_K512, %N96 = transform.structured.tile_using_for %matmul_M960_N_K512 tile_sizes [0, 96, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %B_ddr = transform.get_operand %matmul_M960_N96_K512[1] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %B_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op
    transform.mark_parallel %N96 num_threads = 8 : (!transform.any_op) -> !transform.any_op
    // M 240 C->C_am(multi_buffer) C_am->C(multi_buffer)
    %matmul_M240_N96_K512, %M240 = transform.structured.tile_using_for %matmul_M960_N96_K512 tile_sizes [240, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %C_ddr = transform.get_operand %matmul_M240_N96_K512[2] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<am>} : (!transform.any_value) -> !transform.any_op
    %C_result_am = transform.get_result %matmul_M240_N96_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_write %C_result_am, %C_ddr multi_buffer = true {memory_space = #mtdsp.address_space<global>} : (!transform.any_value, !transform.any_value) -> !transform.any_op
    // M 12  A_gsm->A_sm(multi_buffer)
    %matmul_M12_N96_K512, %M12 = transform.structured.tile_using_for %matmul_M240_N96_K512 tile_sizes [12, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %A_gsm = transform.get_operand %matmul_M12_N96_K512[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.cache_read %A_gsm multi_buffer = true {memory_space = #mtdsp.address_space<sm>} : (!transform.any_value) -> !transform.any_op
    // M 6   unroll(2)
    %matmul_M6_N96_K512, %M6 = transform.structured.tile_using_for %matmul_M12_N96_K512 tile_sizes [6, 0, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.mark_unroll %M6 unroll_factor = 2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}