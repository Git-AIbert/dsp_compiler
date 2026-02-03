module {
  llvm.func @scalar_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @vector_free(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @matmul_micro_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @group_barrier(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_wait_p2p(i32) attributes {sym_visibility = "private"}
  llvm.func @dma_p2p_opt(!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @scalar_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @vector_malloc(i32) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.mlir.global internal @gsm_0() {addr_space = 0 : i32} : !llvm.array<983040 x f32>
  llvm.func @get_thread_id() -> i32 attributes {sym_visibility = "private"}
  llvm.func @set_prir(i64) attributes {sym_visibility = "private"}
  llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(576 : index) : i64
    %1 = llvm.mlir.constant(3072 : index) : i64
    %2 = llvm.mlir.constant(6144 : index) : i64
    %3 = llvm.mlir.constant(23040 : index) : i64
    %4 = llvm.mlir.constant(49152 : index) : i64
    %5 = llvm.mlir.constant(491520 : index) : i64
    %6 = llvm.mlir.constant(768 : i64) : i64
    %7 = llvm.mlir.constant(768 : index) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(960 : index) : i64
    %10 = llvm.mlir.constant(512 : index) : i64
    %11 = llvm.mlir.constant(96 : index) : i64
    %12 = llvm.mlir.constant(240 : index) : i64
    %13 = llvm.mlir.constant(12 : index) : i64
    %14 = llvm.mlir.constant(6 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(4 : index) : i64
    %17 = llvm.mlir.constant(3 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(8 : i32) : i32
    %20 = llvm.mlir.addressof @gsm_0 : !llvm.ptr
    %21 = llvm.mlir.constant(393216 : i32) : i32
    %22 = llvm.mlir.constant(276480 : i32) : i32
    %23 = llvm.mlir.constant(49152 : i32) : i32
    %24 = llvm.mlir.constant(4 : i32) : i32
    %25 = llvm.mlir.constant(false) : i1
    %26 = llvm.mlir.constant(2048 : index) : i64
    %27 = llvm.mlir.constant(1920 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    llvm.call @set_prir(%6) : (i64) -> ()
    %29 = llvm.call @get_thread_id() : () -> i32
    %30 = llvm.getelementptr inbounds %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<983040 x f32>
    %31 = llvm.call @vector_malloc(%21) : (i32) -> !llvm.ptr
    %32 = llvm.call @vector_malloc(%22) : (i32) -> !llvm.ptr
    %33 = llvm.call @scalar_malloc(%23) : (i32) -> !llvm.ptr
    llvm.br ^bb1(%28 : i64)
  ^bb1(%34: i64):  // 2 preds: ^bb0, ^bb32
    %35 = llvm.icmp "slt" %34, %27 : i64
    llvm.cond_br %35, ^bb2, ^bb33
  ^bb2:  // pred: ^bb1
    %36 = llvm.mul %34, %26 : i64
    %37 = llvm.getelementptr inbounds %arg0[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.trunc %10 : i64 to i32
    %39 = llvm.trunc %26 : i64 to i32
    %40 = llvm.mul %38, %24 : i32
    %41 = llvm.mul %39, %24 : i32
    %42 = llvm.sub %41, %40 : i32
    %43 = llvm.sub %40, %40 : i32
    %44 = llvm.call @dma_p2p_opt(%37, %9, %40, %42, %30, %9, %40, %43, %25, %8, %8) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb3(%28, %44 : i64, i32)
  ^bb3(%45: i64, %46: i32):  // 2 preds: ^bb2, ^bb31
    %47 = llvm.icmp "slt" %45, %26 : i64
    llvm.cond_br %47, ^bb4, ^bb32
  ^bb4:  // pred: ^bb3
    %48 = llvm.add %45, %10 : i64
    %49 = llvm.icmp "slt" %48, %26 : i64
    llvm.cond_br %49, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %50 = llvm.sdiv %48, %10  : i64
    %51 = llvm.urem %50, %15  : i64
    %52 = llvm.mul %51, %5 : i64
    %53 = llvm.add %48, %36 : i64
    %54 = llvm.trunc %51 : i64 to i32
    %55 = llvm.getelementptr inbounds %arg0[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.getelementptr inbounds %30[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %57 = llvm.call @dma_p2p_opt(%55, %9, %40, %42, %56, %9, %40, %43, %25, %8, %54) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb7(%57 : i32)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%8 : i32)
  ^bb7(%58: i32):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    llvm.call @dma_wait_p2p(%46) : (i32) -> ()
    %59 = llvm.sdiv %45, %10  : i64
    %60 = llvm.urem %59, %15  : i64
    llvm.call @group_barrier(%8) : (i32) -> ()
    %61 = llvm.sext %29 : i32 to i64
    %62 = llvm.mul %61, %11 : i64
    %63 = llvm.sdiv %62, %7  : i64
    %64 = llvm.urem %63, %15  : i64
    %65 = llvm.mul %64, %4 : i64
    %66 = llvm.mul %45, %27 : i64
    %67 = llvm.add %62, %66 : i64
    %68 = llvm.add %64, %15 : i64
    %69 = llvm.trunc %68 : i64 to i32
    %70 = llvm.getelementptr inbounds %arg1[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.trunc %11 : i64 to i32
    %72 = llvm.trunc %27 : i64 to i32
    %73 = llvm.mul %71, %24 : i32
    %74 = llvm.mul %72, %24 : i32
    %75 = llvm.sub %74, %73 : i32
    %76 = llvm.getelementptr inbounds %31[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %77 = llvm.sub %73, %73 : i32
    %78 = llvm.call @dma_p2p_opt(%70, %10, %73, %75, %76, %10, %73, %77, %25, %8, %69) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb9(%62, %78 : i64, i32)
  ^bb9(%79: i64, %80: i32):  // 2 preds: ^bb8, ^bb30
    %81 = llvm.icmp "slt" %79, %27 : i64
    llvm.cond_br %81, ^bb10, ^bb31
  ^bb10:  // pred: ^bb9
    %82 = llvm.add %79, %7 : i64
    %83 = llvm.icmp "slt" %82, %27 : i64
    llvm.cond_br %83, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %84 = llvm.sdiv %82, %7  : i64
    %85 = llvm.urem %84, %15  : i64
    %86 = llvm.mul %85, %4 : i64
    %87 = llvm.add %82, %66 : i64
    %88 = llvm.add %85, %15 : i64
    %89 = llvm.trunc %88 : i64 to i32
    %90 = llvm.getelementptr inbounds %arg1[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %91 = llvm.getelementptr inbounds %31[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.call @dma_p2p_opt(%90, %10, %73, %75, %91, %10, %73, %77, %25, %8, %89) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb13(%92 : i32)
  ^bb12:  // pred: ^bb10
    llvm.br ^bb13(%8 : i32)
  ^bb13(%93: i32):  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    llvm.call @dma_wait_p2p(%80) : (i32) -> ()
    %94 = llvm.sdiv %79, %7  : i64
    %95 = llvm.urem %94, %15  : i64
    %96 = llvm.mul %95, %4 : i64
    %97 = llvm.mul %34, %27 : i64
    %98 = llvm.add %79, %97 : i64
    %99 = llvm.getelementptr inbounds %arg2[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.call @dma_p2p_opt(%99, %12, %73, %75, %32, %12, %73, %77, %25, %8, %24) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb15(%28, %8, %100 : i64, i32, i32)
  ^bb15(%101: i64, %102: i32, %103: i32):  // 2 preds: ^bb14, ^bb29
    %104 = llvm.icmp "slt" %101, %9 : i64
    llvm.cond_br %104, ^bb16, ^bb30
  ^bb16:  // pred: ^bb15
    %105 = llvm.add %101, %12 : i64
    %106 = llvm.icmp "slt" %105, %9 : i64
    llvm.cond_br %106, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %107 = llvm.sdiv %105, %12  : i64
    %108 = llvm.urem %107, %17  : i64
    %109 = llvm.mul %108, %3 : i64
    %110 = llvm.mul %105, %27 : i64
    %111 = llvm.add %110, %79 : i64
    %112 = llvm.add %111, %97 : i64
    %113 = llvm.urem %107, %15  : i64
    %114 = llvm.add %113, %16 : i64
    %115 = llvm.trunc %114 : i64 to i32
    %116 = llvm.getelementptr inbounds %arg2[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %117 = llvm.getelementptr inbounds %32[%109] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %118 = llvm.call @dma_p2p_opt(%116, %12, %73, %75, %117, %12, %73, %77, %25, %8, %115) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb19(%118 : i32)
  ^bb18:  // pred: ^bb16
    llvm.br ^bb19(%8 : i32)
  ^bb19(%119: i32):  // 2 preds: ^bb17, ^bb18
    llvm.br ^bb20
  ^bb20:  // pred: ^bb19
    llvm.call @dma_wait_p2p(%103) : (i32) -> ()
    %120 = llvm.sdiv %101, %12  : i64
    %121 = llvm.urem %120, %17  : i64
    %122 = llvm.mul %121, %3 : i64
    %123 = llvm.mul %101, %27 : i64
    %124 = llvm.add %123, %79 : i64
    %125 = llvm.add %124, %97 : i64
    %126 = llvm.mul %101, %10 : i64
    %127 = llvm.mul %60, %5 : i64
    %128 = llvm.add %126, %127 : i64
    %129 = llvm.getelementptr inbounds %30[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.call @dma_p2p_opt(%129, %13, %40, %43, %33, %13, %40, %43, %25, %8, %19) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb21(%28, %130 : i64, i32)
  ^bb21(%131: i64, %132: i32):  // 2 preds: ^bb20, ^bb26
    %133 = llvm.icmp "slt" %131, %12 : i64
    llvm.cond_br %133, ^bb22, ^bb27
  ^bb22:  // pred: ^bb21
    %134 = llvm.add %131, %13 : i64
    %135 = llvm.icmp "slt" %134, %12 : i64
    llvm.cond_br %135, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %136 = llvm.sdiv %134, %13  : i64
    %137 = llvm.urem %136, %15  : i64
    %138 = llvm.mul %137, %2 : i64
    %139 = llvm.mul %134, %10 : i64
    %140 = llvm.add %139, %126 : i64
    %141 = llvm.add %140, %127 : i64
    %142 = llvm.add %137, %18 : i64
    %143 = llvm.trunc %142 : i64 to i32
    %144 = llvm.getelementptr inbounds %30[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.getelementptr inbounds %33[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.call @dma_p2p_opt(%144, %13, %40, %43, %145, %13, %40, %43, %25, %8, %143) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    llvm.br ^bb25(%146 : i32)
  ^bb24:  // pred: ^bb22
    llvm.br ^bb25(%8 : i32)
  ^bb25(%147: i32):  // 2 preds: ^bb23, ^bb24
    llvm.br ^bb26
  ^bb26:  // pred: ^bb25
    llvm.call @dma_wait_p2p(%132) : (i32) -> ()
    %148 = llvm.sdiv %131, %13  : i64
    %149 = llvm.urem %148, %15  : i64
    %150 = llvm.mul %149, %2 : i64
    %151 = llvm.mul %131, %11 : i64
    %152 = llvm.add %151, %122 : i64
    %153 = llvm.getelementptr inbounds %33[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %154 = llvm.getelementptr inbounds %31[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %155 = llvm.getelementptr inbounds %32[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @matmul_micro_kernel(%153, %154, %155, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
    %156 = llvm.add %150, %1 : i64
    %157 = llvm.add %152, %0 : i64
    %158 = llvm.getelementptr inbounds %33[%156] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %159 = llvm.getelementptr inbounds %32[%157] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.call @matmul_micro_kernel(%158, %154, %159, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb21(%134, %147 : i64, i32)
  ^bb27:  // pred: ^bb21
    %160 = llvm.urem %120, %15  : i64
    %161 = llvm.add %160, %14 : i64
    %162 = llvm.trunc %161 : i64 to i32
    %163 = llvm.getelementptr inbounds %32[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %164 = llvm.getelementptr inbounds %arg2[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %165 = llvm.call @dma_p2p_opt(%163, %12, %73, %77, %164, %12, %73, %75, %25, %8, %162) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, i64, i32, i32, i1, i32, i32) -> i32
    %166 = llvm.icmp "ne" %101, %28 : i64
    llvm.cond_br %166, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    llvm.call @dma_wait_p2p(%102) : (i32) -> ()
    llvm.br ^bb29
  ^bb29:  // 2 preds: ^bb27, ^bb28
    llvm.br ^bb15(%105, %165, %119 : i64, i32, i32)
  ^bb30:  // pred: ^bb15
    llvm.call @dma_wait_p2p(%102) : (i32) -> ()
    llvm.br ^bb9(%82, %93 : i64, i32)
  ^bb31:  // pred: ^bb9
    llvm.call @group_barrier(%8) : (i32) -> ()
    llvm.br ^bb3(%48, %58 : i64, i32)
  ^bb32:  // pred: ^bb3
    %167 = llvm.add %34, %9 : i64
    llvm.br ^bb1(%167 : i64)
  ^bb33:  // pred: ^bb1
    %168 = llvm.call @vector_free(%31) : (!llvm.ptr) -> i32
    %169 = llvm.call @vector_free(%32) : (!llvm.ptr) -> i32
    %170 = llvm.call @scalar_free(%33) : (!llvm.ptr) -> i32
    llvm.return %arg2 : !llvm.ptr
  }
}

