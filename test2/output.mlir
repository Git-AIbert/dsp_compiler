module {
  llvm.func @vector_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @relu_micro_kernel_n128(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @elemwise_add_micro_kernel_v16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @micro_kernel_asm_r6c128(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @group_barrier(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_wait_p2p(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.mlir.global internal @gsm_0() {addr_space = 0 : i32} : !llvm.array<589824 x f32>
  llvm.func @vector_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @get_thread_id() -> i32 attributes {sym_visibility = "private"}
  llvm.func @set_prir(i64) attributes {sym_visibility = "private"}
  llvm.func @matmul_add_relu(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.constant(786432 : index) : i64
    %1 = llvm.mlir.constant(6144 : index) : i64
    %2 = llvm.mlir.constant(12288 : index) : i64
    %3 = llvm.mlir.constant(-1 : index) : i64
    %4 = llvm.mlir.constant(65536 : index) : i64
    %5 = llvm.mlir.constant(256 : index) : i64
    %6 = llvm.mlir.constant(1024 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(96 : index) : i64
    %9 = llvm.mlir.constant(12 : index) : i64
    %10 = llvm.mlir.constant(6 : index) : i64
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(8 : index) : i64
    %13 = llvm.mlir.constant(3 : index) : i64
    %14 = llvm.mlir.constant(4 : index) : i64
    %15 = llvm.mlir.constant(8 : i32) : i32
    %16 = llvm.mlir.constant(2 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(9 : i32) : i32
    %19 = llvm.mlir.constant(5 : index) : i64
    %20 = llvm.mlir.constant(524288 : i32) : i32
    %21 = llvm.mlir.constant(147456 : i32) : i32
    %22 = llvm.mlir.constant(3 : i64) : i64
    %23 = llvm.mlir.addressof @gsm_0 : !llvm.ptr
    %24 = llvm.mlir.constant(98304 : i32) : i32
    %25 = llvm.mlir.constant(49152 : i32) : i32
    %26 = llvm.mlir.constant(4 : i32) : i32
    %27 = llvm.mlir.constant(false) : i1
    %28 = llvm.mlir.constant(768 : index) : i64
    %29 = llvm.mlir.constant(3072 : index) : i64
    %30 = llvm.mlir.constant(512 : index) : i64
    %31 = llvm.mlir.constant(0 : index) : i64
    llvm.call @set_prir(%22) : (i64) -> ()
    %32 = llvm.call @get_thread_id() : () -> i32
    %33 = llvm.icmp "eq" %32, %17 : i32
    %34 = llvm.call @vector_malloc(%20) : (i32) -> !llvm.ptr
    %35 = llvm.call @vector_malloc(%21) : (i32) -> !llvm.ptr
    %36 = llvm.getelementptr inbounds %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<589824 x f32>
    %37 = llvm.call @vector_malloc(%24) : (i32) -> !llvm.ptr
    %38 = llvm.call @scalar_malloc(%25) : (i32) -> !llvm.ptr
    llvm.cond_br %33, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %39 = llvm.trunc %30 : i64 to i32
    %40 = llvm.trunc %28 : i64 to i32
    %41 = llvm.mul %39, %26 : i32
    %42 = llvm.mul %40, %26 : i32
    %43 = llvm.sub %42, %41 : i32
    %44 = llvm.sub %41, %41 : i32
    %45 = llvm.call @dma_p2p_opt(%arg0, %30, %41, %43, %36, %30, %41, %44, %27, %17, %15) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb3(%45 : i32)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%17 : i32)
  ^bb3(%46: i32):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.cond_br %33, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %47 = llvm.getelementptr inbounds %arg0[512] : (!llvm.ptr) -> !llvm.ptr, f32
    %48 = llvm.trunc %5 : i64 to i32
    %49 = llvm.trunc %28 : i64 to i32
    %50 = llvm.mul %48, %26 : i32
    %51 = llvm.mul %49, %26 : i32
    %52 = llvm.sub %51, %50 : i32
    %53 = llvm.getelementptr inbounds %36[294912] : (!llvm.ptr) -> !llvm.ptr, f32
    %54 = llvm.trunc %30 : i64 to i32
    %55 = llvm.mul %54, %26 : i32
    %56 = llvm.sub %55, %50 : i32
    %57 = llvm.call @dma_p2p_opt(%47, %30, %50, %52, %53, %30, %50, %56, %27, %17, %18) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.call @dma_wait_p2p(%46) : (i32) -> ()
    llvm.br ^bb7(%57 : i32)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%17 : i32)
  ^bb7(%58: i32):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    llvm.call @group_barrier(%17) : (i32) -> ()
    %59 = llvm.sext %32 : i32 to i64
    %60 = llvm.mul %59, %7 : i64
    %61 = llvm.sdiv %60, %6  : i64
    %62 = llvm.urem %61, %11  : i64
    %63 = llvm.mul %62, %4 : i64
    %64 = llvm.add %62, %10 : i64
    %65 = llvm.trunc %64 : i64 to i32
    %66 = llvm.getelementptr inbounds %arg1[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.trunc %7 : i64 to i32
    %68 = llvm.trunc %29 : i64 to i32
    %69 = llvm.mul %67, %26 : i32
    %70 = llvm.mul %68, %26 : i32
    %71 = llvm.sub %70, %69 : i32
    %72 = llvm.getelementptr inbounds %34[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.sub %69, %69 : i32
    %74 = llvm.call @dma_p2p_opt(%66, %30, %69, %71, %72, %30, %69, %73, %27, %17, %65) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb9(%60, %74 : i64, i32)
  ^bb9(%75: i64, %76: i32):  // 2 preds: ^bb8, ^bb36
    %77 = llvm.icmp "slt" %75, %29 : i64
    llvm.cond_br %77, ^bb10, ^bb37
  ^bb10:  // pred: ^bb9
    %78 = llvm.add %75, %6 : i64
    %79 = llvm.icmp "slt" %78, %29 : i64
    llvm.cond_br %79, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %80 = llvm.sdiv %78, %6  : i64
    %81 = llvm.urem %80, %11  : i64
    %82 = llvm.mul %81, %4 : i64
    %83 = llvm.add %81, %10 : i64
    %84 = llvm.trunc %83 : i64 to i32
    %85 = llvm.getelementptr inbounds %arg1[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.getelementptr inbounds %34[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.call @dma_p2p_opt(%85, %30, %69, %71, %86, %30, %69, %73, %27, %17, %84) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb13(%87 : i32)
  ^bb12:  // pred: ^bb10
    llvm.br ^bb13(%17 : i32)
  ^bb13(%88: i32):  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    llvm.call @dma_wait_p2p(%76) : (i32) -> ()
    %89 = llvm.sdiv %75, %6  : i64
    %90 = llvm.urem %89, %11  : i64
    %91 = llvm.mul %90, %4 : i64
    %92 = llvm.getelementptr inbounds %arg3[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.call @dma_p2p_opt(%92, %8, %69, %71, %35, %8, %69, %73, %27, %17, %16) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb15(%31, %17, %93 : i64, i32, i32)
  ^bb15(%94: i64, %95: i32, %96: i32):  // 2 preds: ^bb14, ^bb35
    %97 = llvm.icmp "slt" %94, %30 : i64
    llvm.cond_br %97, ^bb16, ^bb36
  ^bb16:  // pred: ^bb15
    %98 = llvm.add %94, %8 : i64
    %99 = llvm.icmp "slt" %98, %30 : i64
    llvm.cond_br %99, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %100 = llvm.sdiv %98, %8  : i64
    %101 = llvm.urem %100, %13  : i64
    %102 = llvm.mul %98, %3 : i64
    %103 = llvm.add %102, %30 : i64
    %104 = llvm.intr.smin(%103, %8)  : (i64, i64) -> i64
    %105 = llvm.mul %98, %29 : i64
    %106 = llvm.add %75, %105 : i64
    %107 = llvm.mul %101, %2 : i64
    %108 = llvm.urem %100, %11  : i64
    %109 = llvm.add %108, %11 : i64
    %110 = llvm.trunc %109 : i64 to i32
    %111 = llvm.getelementptr inbounds %arg3[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.getelementptr inbounds %35[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.call @dma_p2p_opt(%111, %104, %69, %71, %112, %104, %69, %73, %27, %17, %110) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb19(%113 : i32)
  ^bb18:  // pred: ^bb16
    llvm.br ^bb19(%17 : i32)
  ^bb19(%114: i32):  // 2 preds: ^bb17, ^bb18
    llvm.br ^bb20
  ^bb20:  // pred: ^bb19
    llvm.call @dma_wait_p2p(%96) : (i32) -> ()
    %115 = llvm.sdiv %94, %8  : i64
    %116 = llvm.urem %115, %13  : i64
    %117 = llvm.mul %94, %3 : i64
    %118 = llvm.add %117, %30 : i64
    %119 = llvm.intr.smin(%118, %8)  : (i64, i64) -> i64
    %120 = llvm.mul %94, %29 : i64
    %121 = llvm.add %75, %120 : i64
    %122 = llvm.mul %116, %2 : i64
    %123 = llvm.intr.smin(%118, %9)  : (i64, i64) -> i64
    %124 = llvm.intr.smin(%123, %8)  : (i64, i64) -> i64
    %125 = llvm.mul %94, %30 : i64
    %126 = llvm.getelementptr inbounds %36[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.trunc %30 : i64 to i32
    %128 = llvm.mul %127, %26 : i32
    %129 = llvm.sub %128, %128 : i32
    %130 = llvm.call @dma_p2p_opt(%126, %124, %128, %129, %38, %124, %128, %129, %27, %17, %17) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb21(%31, %130 : i64, i32)
  ^bb21(%131: i64, %132: i32):  // 2 preds: ^bb20, ^bb32
    %133 = llvm.icmp "slt" %131, %119 : i64
    llvm.cond_br %133, ^bb22, ^bb33
  ^bb22:  // pred: ^bb21
    %134 = llvm.add %131, %9 : i64
    %135 = llvm.icmp "slt" %134, %119 : i64
    llvm.cond_br %135, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %136 = llvm.sdiv %134, %9  : i64
    %137 = llvm.urem %136, %11  : i64
    %138 = llvm.mul %134, %3 : i64
    %139 = llvm.add %138, %119 : i64
    %140 = llvm.intr.smin(%139, %9)  : (i64, i64) -> i64
    %141 = llvm.mul %134, %30 : i64
    %142 = llvm.add %141, %125 : i64
    %143 = llvm.mul %137, %1 : i64
    %144 = llvm.trunc %137 : i64 to i32
    %145 = llvm.getelementptr inbounds %36[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.getelementptr inbounds %38[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %147 = llvm.call @dma_p2p_opt(%145, %140, %128, %129, %146, %140, %128, %129, %27, %17, %144) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb25(%147 : i32)
  ^bb24:  // pred: ^bb22
    llvm.br ^bb25(%17 : i32)
  ^bb25(%148: i32):  // 2 preds: ^bb23, ^bb24
    llvm.br ^bb26
  ^bb26:  // pred: ^bb25
    llvm.call @dma_wait_p2p(%132) : (i32) -> ()
    %149 = llvm.sdiv %131, %9  : i64
    %150 = llvm.urem %149, %11  : i64
    %151 = llvm.mul %131, %3 : i64
    %152 = llvm.add %151, %119 : i64
    %153 = llvm.intr.smin(%152, %9)  : (i64, i64) -> i64
    %154 = llvm.add %153, %19 : i64
    %155 = llvm.udiv %154, %10  : i64
    %156 = llvm.srem %155, %11  : i64
    %157 = llvm.sub %155, %156 : i64
    %158 = llvm.mul %157, %10 : i64
    llvm.br ^bb27(%31 : i64)
  ^bb27(%159: i64):  // 2 preds: ^bb26, ^bb28
    %160 = llvm.icmp "slt" %159, %158 : i64
    llvm.cond_br %160, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %161 = llvm.mul %159, %30 : i64
    %162 = llvm.mul %150, %1 : i64
    %163 = llvm.add %161, %162 : i64
    %164 = llvm.mul %159, %7 : i64
    %165 = llvm.mul %131, %7 : i64
    %166 = llvm.add %164, %165 : i64
    %167 = llvm.add %166, %122 : i64
    %168 = llvm.getelementptr inbounds %38[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %169 = llvm.getelementptr inbounds %34[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %170 = llvm.getelementptr inbounds %35[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%168, %169, %170, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %171 = llvm.add %159, %10 : i64
    %172 = llvm.mul %171, %30 : i64
    %173 = llvm.add %172, %162 : i64
    %174 = llvm.mul %171, %7 : i64
    %175 = llvm.add %174, %165 : i64
    %176 = llvm.add %175, %122 : i64
    %177 = llvm.getelementptr inbounds %38[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %178 = llvm.getelementptr inbounds %35[%176] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%177, %169, %178, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %179 = llvm.add %159, %9 : i64
    llvm.br ^bb27(%179 : i64)
  ^bb29:  // pred: ^bb27
    llvm.br ^bb30(%158 : i64)
  ^bb30(%180: i64):  // 2 preds: ^bb29, ^bb31
    %181 = llvm.icmp "slt" %180, %153 : i64
    llvm.cond_br %181, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %182 = llvm.mul %180, %30 : i64
    %183 = llvm.mul %150, %1 : i64
    %184 = llvm.add %182, %183 : i64
    %185 = llvm.mul %180, %7 : i64
    %186 = llvm.mul %131, %7 : i64
    %187 = llvm.add %185, %186 : i64
    %188 = llvm.add %187, %122 : i64
    %189 = llvm.getelementptr inbounds %38[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %190 = llvm.getelementptr inbounds %34[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %191 = llvm.getelementptr inbounds %35[%188] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%189, %190, %191, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %192 = llvm.add %180, %10 : i64
    llvm.br ^bb30(%192 : i64)
  ^bb32:  // pred: ^bb30
    llvm.br ^bb21(%134, %148 : i64, i32)
  ^bb33:  // pred: ^bb21
    %193 = llvm.urem %115, %11  : i64
    %194 = llvm.add %193, %14 : i64
    %195 = llvm.trunc %194 : i64 to i32
    %196 = llvm.getelementptr inbounds %35[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %197 = llvm.getelementptr inbounds %arg3[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %198 = llvm.call @dma_p2p_opt(%196, %119, %69, %73, %197, %119, %69, %71, %27, %17, %195) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %199 = llvm.icmp "ne" %94, %31 : i64
    llvm.cond_br %199, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    llvm.call @dma_wait_p2p(%95) : (i32) -> ()
    llvm.br ^bb35
  ^bb35:  // 2 preds: ^bb33, ^bb34
    llvm.br ^bb15(%98, %198, %114 : i64, i32, i32)
  ^bb36:  // pred: ^bb15
    llvm.call @dma_wait_p2p(%95) : (i32) -> ()
    llvm.br ^bb9(%78, %88 : i64, i32)
  ^bb37:  // pred: ^bb9
    llvm.call @group_barrier(%17) : (i32) -> ()
    llvm.cond_br %33, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    llvm.call @dma_wait_p2p(%58) : (i32) -> ()
    llvm.br ^bb39
  ^bb39:  // 2 preds: ^bb37, ^bb38
    llvm.call @group_barrier(%17) : (i32) -> ()
    %200 = llvm.add %60, %0 : i64
    %201 = llvm.add %62, %12 : i64
    %202 = llvm.trunc %201 : i64 to i32
    %203 = llvm.getelementptr inbounds %arg1[%200] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %204 = llvm.call @dma_p2p_opt(%203, %30, %69, %71, %72, %30, %69, %73, %27, %17, %202) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb40(%60, %204 : i64, i32)
  ^bb40(%205: i64, %206: i32):  // 2 preds: ^bb39, ^bb67
    %207 = llvm.icmp "slt" %205, %29 : i64
    llvm.cond_br %207, ^bb41, ^bb68
  ^bb41:  // pred: ^bb40
    %208 = llvm.add %205, %6 : i64
    %209 = llvm.icmp "slt" %208, %29 : i64
    llvm.cond_br %209, ^bb42, ^bb43
  ^bb42:  // pred: ^bb41
    %210 = llvm.sdiv %208, %6  : i64
    %211 = llvm.urem %210, %11  : i64
    %212 = llvm.mul %211, %4 : i64
    %213 = llvm.add %208, %0 : i64
    %214 = llvm.add %211, %12 : i64
    %215 = llvm.trunc %214 : i64 to i32
    %216 = llvm.getelementptr inbounds %arg1[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %217 = llvm.getelementptr inbounds %34[%212] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %218 = llvm.call @dma_p2p_opt(%216, %30, %69, %71, %217, %30, %69, %73, %27, %17, %215) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb44(%218 : i32)
  ^bb43:  // pred: ^bb41
    llvm.br ^bb44(%17 : i32)
  ^bb44(%219: i32):  // 2 preds: ^bb42, ^bb43
    llvm.br ^bb45
  ^bb45:  // pred: ^bb44
    llvm.call @dma_wait_p2p(%206) : (i32) -> ()
    %220 = llvm.sdiv %205, %6  : i64
    %221 = llvm.urem %220, %11  : i64
    %222 = llvm.mul %221, %4 : i64
    %223 = llvm.getelementptr inbounds %arg3[%205] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %224 = llvm.call @dma_p2p_opt(%223, %8, %69, %71, %35, %8, %69, %73, %27, %17, %16) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %225 = llvm.getelementptr inbounds %arg2[%205] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %226 = llvm.call @dma_p2p_opt(%225, %8, %69, %71, %37, %8, %69, %73, %27, %17, %26) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb46(%31, %17, %224, %226 : i64, i32, i32, i32)
  ^bb46(%227: i64, %228: i32, %229: i32, %230: i32):  // 2 preds: ^bb45, ^bb66
    %231 = llvm.icmp "slt" %227, %30 : i64
    llvm.cond_br %231, ^bb47, ^bb67
  ^bb47:  // pred: ^bb46
    %232 = llvm.add %227, %8 : i64
    %233 = llvm.icmp "slt" %232, %30 : i64
    llvm.cond_br %233, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    %234 = llvm.sdiv %232, %8  : i64
    %235 = llvm.urem %234, %13  : i64
    %236 = llvm.urem %234, %11  : i64
    %237 = llvm.mul %232, %3 : i64
    %238 = llvm.add %237, %30 : i64
    %239 = llvm.intr.smin(%238, %8)  : (i64, i64) -> i64
    %240 = llvm.mul %232, %29 : i64
    %241 = llvm.add %205, %240 : i64
    %242 = llvm.mul %236, %2 : i64
    %243 = llvm.mul %235, %2 : i64
    %244 = llvm.add %236, %11 : i64
    %245 = llvm.trunc %244 : i64 to i32
    %246 = llvm.getelementptr inbounds %arg3[%241] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %247 = llvm.getelementptr inbounds %35[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %248 = llvm.call @dma_p2p_opt(%246, %239, %69, %71, %247, %239, %69, %73, %27, %17, %245) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %249 = llvm.add %236, %14 : i64
    %250 = llvm.trunc %249 : i64 to i32
    %251 = llvm.getelementptr inbounds %arg2[%241] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %252 = llvm.getelementptr inbounds %37[%242] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %253 = llvm.call @dma_p2p_opt(%251, %239, %69, %71, %252, %239, %69, %73, %27, %17, %250) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb50(%248, %253 : i32, i32)
  ^bb49:  // pred: ^bb47
    llvm.br ^bb50(%17, %17 : i32, i32)
  ^bb50(%254: i32, %255: i32):  // 2 preds: ^bb48, ^bb49
    llvm.br ^bb51
  ^bb51:  // pred: ^bb50
    llvm.call @dma_wait_p2p(%229) : (i32) -> ()
    llvm.call @dma_wait_p2p(%230) : (i32) -> ()
    %256 = llvm.sdiv %227, %8  : i64
    %257 = llvm.urem %256, %13  : i64
    %258 = llvm.urem %256, %11  : i64
    %259 = llvm.mul %227, %3 : i64
    %260 = llvm.add %259, %30 : i64
    %261 = llvm.intr.smin(%260, %8)  : (i64, i64) -> i64
    %262 = llvm.mul %227, %29 : i64
    %263 = llvm.add %205, %262 : i64
    %264 = llvm.mul %258, %2 : i64
    %265 = llvm.mul %257, %2 : i64
    %266 = llvm.intr.smin(%260, %9)  : (i64, i64) -> i64
    %267 = llvm.intr.smin(%266, %8)  : (i64, i64) -> i64
    %268 = llvm.mul %227, %30 : i64
    %269 = llvm.getelementptr inbounds %36[%268] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %270 = llvm.trunc %30 : i64 to i32
    %271 = llvm.mul %270, %26 : i32
    %272 = llvm.sub %271, %271 : i32
    %273 = llvm.call @dma_p2p_opt(%269, %267, %271, %272, %38, %267, %271, %272, %27, %17, %17) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb52(%31, %273 : i64, i32)
  ^bb52(%274: i64, %275: i32):  // 2 preds: ^bb51, ^bb63
    %276 = llvm.icmp "slt" %274, %261 : i64
    llvm.cond_br %276, ^bb53, ^bb64
  ^bb53:  // pred: ^bb52
    %277 = llvm.add %274, %9 : i64
    %278 = llvm.icmp "slt" %277, %261 : i64
    llvm.cond_br %278, ^bb54, ^bb55
  ^bb54:  // pred: ^bb53
    %279 = llvm.sdiv %277, %9  : i64
    %280 = llvm.urem %279, %11  : i64
    %281 = llvm.mul %277, %3 : i64
    %282 = llvm.add %281, %261 : i64
    %283 = llvm.intr.smin(%282, %9)  : (i64, i64) -> i64
    %284 = llvm.mul %277, %30 : i64
    %285 = llvm.add %284, %268 : i64
    %286 = llvm.mul %280, %1 : i64
    %287 = llvm.trunc %280 : i64 to i32
    %288 = llvm.getelementptr inbounds %36[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %289 = llvm.getelementptr inbounds %38[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %290 = llvm.call @dma_p2p_opt(%288, %283, %271, %272, %289, %283, %271, %272, %27, %17, %287) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb56(%290 : i32)
  ^bb55:  // pred: ^bb53
    llvm.br ^bb56(%17 : i32)
  ^bb56(%291: i32):  // 2 preds: ^bb54, ^bb55
    llvm.br ^bb57
  ^bb57:  // pred: ^bb56
    llvm.call @dma_wait_p2p(%275) : (i32) -> ()
    %292 = llvm.sdiv %274, %9  : i64
    %293 = llvm.urem %292, %11  : i64
    %294 = llvm.mul %274, %3 : i64
    %295 = llvm.add %294, %261 : i64
    %296 = llvm.intr.smin(%295, %9)  : (i64, i64) -> i64
    %297 = llvm.add %296, %19 : i64
    %298 = llvm.udiv %297, %10  : i64
    %299 = llvm.srem %298, %11  : i64
    %300 = llvm.sub %298, %299 : i64
    %301 = llvm.mul %300, %10 : i64
    llvm.br ^bb58(%31 : i64)
  ^bb58(%302: i64):  // 2 preds: ^bb57, ^bb59
    %303 = llvm.icmp "slt" %302, %301 : i64
    llvm.cond_br %303, ^bb59, ^bb60
  ^bb59:  // pred: ^bb58
    %304 = llvm.mul %302, %30 : i64
    %305 = llvm.mul %293, %1 : i64
    %306 = llvm.add %304, %305 : i64
    %307 = llvm.mul %302, %7 : i64
    %308 = llvm.mul %274, %7 : i64
    %309 = llvm.add %307, %308 : i64
    %310 = llvm.add %309, %265 : i64
    %311 = llvm.getelementptr inbounds %38[%306] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %312 = llvm.getelementptr inbounds %34[%222] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %313 = llvm.getelementptr inbounds %35[%310] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%311, %312, %313, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %314 = llvm.add %302, %10 : i64
    %315 = llvm.mul %314, %30 : i64
    %316 = llvm.add %315, %305 : i64
    %317 = llvm.mul %314, %7 : i64
    %318 = llvm.add %317, %308 : i64
    %319 = llvm.add %318, %265 : i64
    %320 = llvm.getelementptr inbounds %38[%316] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %321 = llvm.getelementptr inbounds %35[%319] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%320, %312, %321, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %322 = llvm.add %302, %9 : i64
    llvm.br ^bb58(%322 : i64)
  ^bb60:  // pred: ^bb58
    llvm.br ^bb61(%301 : i64)
  ^bb61(%323: i64):  // 2 preds: ^bb60, ^bb62
    %324 = llvm.icmp "slt" %323, %296 : i64
    llvm.cond_br %324, ^bb62, ^bb63
  ^bb62:  // pred: ^bb61
    %325 = llvm.mul %323, %30 : i64
    %326 = llvm.mul %293, %1 : i64
    %327 = llvm.add %325, %326 : i64
    %328 = llvm.mul %323, %7 : i64
    %329 = llvm.mul %274, %7 : i64
    %330 = llvm.add %328, %329 : i64
    %331 = llvm.add %330, %265 : i64
    %332 = llvm.getelementptr inbounds %38[%327] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %333 = llvm.getelementptr inbounds %34[%222] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %334 = llvm.getelementptr inbounds %35[%331] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @micro_kernel_asm_r6c128(%332, %333, %334, %30, %30, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %335 = llvm.add %323, %10 : i64
    llvm.br ^bb61(%335 : i64)
  ^bb63:  // pred: ^bb61
    llvm.br ^bb52(%277, %291 : i64, i32)
  ^bb64:  // pred: ^bb52
    %336 = llvm.getelementptr inbounds %35[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %337 = llvm.getelementptr inbounds %37[%264] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @elemwise_add_micro_kernel_v16(%336, %337, %336, %261, %7, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.call @relu_micro_kernel_n128(%336, %261, %7) : (!llvm.ptr, i64, i64) -> ()
    %338 = llvm.add %258, %10 : i64
    %339 = llvm.trunc %338 : i64 to i32
    %340 = llvm.getelementptr inbounds %arg3[%263] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %341 = llvm.call @dma_p2p_opt(%336, %261, %69, %73, %340, %261, %69, %71, %27, %17, %339) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %342 = llvm.icmp "ne" %227, %31 : i64
    llvm.cond_br %342, ^bb65, ^bb66
  ^bb65:  // pred: ^bb64
    llvm.call @dma_wait_p2p(%228) : (i32) -> ()
    llvm.br ^bb66
  ^bb66:  // 2 preds: ^bb64, ^bb65
    llvm.br ^bb46(%232, %341, %254, %255 : i64, i32, i32, i32)
  ^bb67:  // pred: ^bb46
    llvm.call @dma_wait_p2p(%228) : (i32) -> ()
    llvm.br ^bb40(%208, %219 : i64, i32)
  ^bb68:  // pred: ^bb40
    llvm.call @group_barrier(%17) : (i32) -> ()
    %343 = llvm.call @scalar_free(%38) : (!llvm.ptr) -> i32
    %344 = llvm.call @vector_free(%37) : (!llvm.ptr) -> i32
    %345 = llvm.call @vector_free(%35) : (!llvm.ptr) -> i32
    %346 = llvm.call @vector_free(%34) : (!llvm.ptr) -> i32
    llvm.return
  }
}
