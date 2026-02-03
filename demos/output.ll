; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@gsm_0 = internal global [983040 x float] undef, section ".gsm"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @scalar_free(ptr)

declare i32 @vector_free(ptr)

declare void @matmul_micro_kernel(ptr, ptr, ptr, i64)

declare void @group_barrier(i32)

declare void @dma_wait_p2p(i32)

declare i32 @dma_p2p_opt(ptr, i64, i32, i32, ptr, i64, i32, i32, i1, i32, i32)

declare ptr @scalar_malloc(i32)

declare ptr @vector_malloc(i32)

declare i32 @get_thread_id()

declare void @set_prir(i64)

define ptr @matmul(ptr %0, ptr %1, ptr %2)  section ".global" {
  call void @set_prir(i64 768)
  %4 = call i32 @get_thread_id()
  %5 = call ptr @vector_malloc(i32 393216)
  %6 = call ptr @vector_malloc(i32 276480)
  %7 = call ptr @scalar_malloc(i32 49152)
  br label %8

8:                                                ; preds = %160, %3
  %9 = phi i64 [ %161, %160 ], [ 0, %3 ]
  %10 = icmp slt i64 %9, 1920
  br i1 %10, label %11, label %162

11:                                               ; preds = %8
  %12 = mul i64 %9, 2048
  %13 = getelementptr inbounds float, ptr %0, i64 %12
  %14 = call i32 @dma_p2p_opt(ptr %13, i64 960, i32 2048, i32 6144, ptr @gsm_0, i64 960, i32 2048, i32 0, i1 false, i32 0, i32 0)
  br label %15

15:                                               ; preds = %159, %11
  %16 = phi i64 [ %20, %159 ], [ 0, %11 ]
  %17 = phi i32 [ %33, %159 ], [ %14, %11 ]
  %18 = icmp slt i64 %16, 2048
  br i1 %18, label %19, label %160

19:                                               ; preds = %15
  %20 = add i64 %16, 512
  %21 = icmp slt i64 %20, 2048
  br i1 %21, label %22, label %31

22:                                               ; preds = %19
  %23 = sdiv i64 %20, 512
  %24 = urem i64 %23, 2
  %25 = mul i64 %24, 491520
  %26 = add i64 %20, %12
  %27 = trunc i64 %24 to i32
  %28 = getelementptr inbounds float, ptr %0, i64 %26
  %29 = getelementptr inbounds float, ptr @gsm_0, i64 %25
  %30 = call i32 @dma_p2p_opt(ptr %28, i64 960, i32 2048, i32 6144, ptr %29, i64 960, i32 2048, i32 0, i1 false, i32 0, i32 %27)
  br label %32

31:                                               ; preds = %19
  br label %32

32:                                               ; preds = %22, %31
  %33 = phi i32 [ 0, %31 ], [ %30, %22 ]
  br label %34

34:                                               ; preds = %32
  call void @dma_wait_p2p(i32 %17)
  %35 = sdiv i64 %16, 512
  %36 = urem i64 %35, 2
  call void @group_barrier(i32 0)
  %37 = sext i32 %4 to i64
  %38 = mul i64 %37, 96
  %39 = sdiv i64 %38, 768
  %40 = urem i64 %39, 2
  %41 = mul i64 %40, 49152
  %42 = mul i64 %16, 1920
  %43 = add i64 %38, %42
  %44 = add i64 %40, 2
  %45 = trunc i64 %44 to i32
  %46 = getelementptr inbounds float, ptr %1, i64 %43
  %47 = getelementptr inbounds float, ptr %5, i64 %41
  %48 = call i32 @dma_p2p_opt(ptr %46, i64 512, i32 384, i32 7296, ptr %47, i64 512, i32 384, i32 0, i1 false, i32 0, i32 %45)
  br label %49

49:                                               ; preds = %158, %34
  %50 = phi i64 [ %54, %158 ], [ %38, %34 ]
  %51 = phi i32 [ %68, %158 ], [ %48, %34 ]
  %52 = icmp slt i64 %50, 1920
  br i1 %52, label %53, label %159

53:                                               ; preds = %49
  %54 = add i64 %50, 768
  %55 = icmp slt i64 %54, 1920
  br i1 %55, label %56, label %66

56:                                               ; preds = %53
  %57 = sdiv i64 %54, 768
  %58 = urem i64 %57, 2
  %59 = mul i64 %58, 49152
  %60 = add i64 %54, %42
  %61 = add i64 %58, 2
  %62 = trunc i64 %61 to i32
  %63 = getelementptr inbounds float, ptr %1, i64 %60
  %64 = getelementptr inbounds float, ptr %5, i64 %59
  %65 = call i32 @dma_p2p_opt(ptr %63, i64 512, i32 384, i32 7296, ptr %64, i64 512, i32 384, i32 0, i1 false, i32 0, i32 %62)
  br label %67

66:                                               ; preds = %53
  br label %67

67:                                               ; preds = %56, %66
  %68 = phi i32 [ 0, %66 ], [ %65, %56 ]
  br label %69

69:                                               ; preds = %67
  call void @dma_wait_p2p(i32 %51)
  %70 = sdiv i64 %50, 768
  %71 = urem i64 %70, 2
  %72 = mul i64 %71, 49152
  %73 = mul i64 %9, 1920
  %74 = add i64 %50, %73
  %75 = getelementptr inbounds float, ptr %2, i64 %74
  %76 = call i32 @dma_p2p_opt(ptr %75, i64 240, i32 384, i32 7296, ptr %6, i64 240, i32 384, i32 0, i1 false, i32 0, i32 4)
  br label %77

77:                                               ; preds = %157, %69
  %78 = phi i64 [ %83, %157 ], [ 0, %69 ]
  %79 = phi i32 [ %154, %157 ], [ 0, %69 ]
  %80 = phi i32 [ %100, %157 ], [ %76, %69 ]
  %81 = icmp slt i64 %78, 960
  br i1 %81, label %82, label %158

82:                                               ; preds = %77
  %83 = add i64 %78, 240
  %84 = icmp slt i64 %83, 960
  br i1 %84, label %85, label %98

85:                                               ; preds = %82
  %86 = sdiv i64 %83, 240
  %87 = urem i64 %86, 3
  %88 = mul i64 %87, 23040
  %89 = mul i64 %83, 1920
  %90 = add i64 %89, %50
  %91 = add i64 %90, %73
  %92 = urem i64 %86, 2
  %93 = add i64 %92, 4
  %94 = trunc i64 %93 to i32
  %95 = getelementptr inbounds float, ptr %2, i64 %91
  %96 = getelementptr inbounds float, ptr %6, i64 %88
  %97 = call i32 @dma_p2p_opt(ptr %95, i64 240, i32 384, i32 7296, ptr %96, i64 240, i32 384, i32 0, i1 false, i32 0, i32 %94)
  br label %99

98:                                               ; preds = %82
  br label %99

99:                                               ; preds = %85, %98
  %100 = phi i32 [ 0, %98 ], [ %97, %85 ]
  br label %101

101:                                              ; preds = %99
  call void @dma_wait_p2p(i32 %80)
  %102 = sdiv i64 %78, 240
  %103 = urem i64 %102, 3
  %104 = mul i64 %103, 23040
  %105 = mul i64 %78, 1920
  %106 = add i64 %105, %50
  %107 = add i64 %106, %73
  %108 = mul i64 %78, 512
  %109 = mul i64 %36, 491520
  %110 = add i64 %108, %109
  %111 = getelementptr inbounds float, ptr @gsm_0, i64 %110
  %112 = call i32 @dma_p2p_opt(ptr %111, i64 12, i32 2048, i32 0, ptr %7, i64 12, i32 2048, i32 0, i1 false, i32 0, i32 8)
  br label %113

113:                                              ; preds = %135, %101
  %114 = phi i64 [ %118, %135 ], [ 0, %101 ]
  %115 = phi i32 [ %134, %135 ], [ %112, %101 ]
  %116 = icmp slt i64 %114, 240
  br i1 %116, label %117, label %148

117:                                              ; preds = %113
  %118 = add i64 %114, 12
  %119 = icmp slt i64 %118, 240
  br i1 %119, label %120, label %132

120:                                              ; preds = %117
  %121 = sdiv i64 %118, 12
  %122 = urem i64 %121, 2
  %123 = mul i64 %122, 6144
  %124 = mul i64 %118, 512
  %125 = add i64 %124, %108
  %126 = add i64 %125, %109
  %127 = add i64 %122, 8
  %128 = trunc i64 %127 to i32
  %129 = getelementptr inbounds float, ptr @gsm_0, i64 %126
  %130 = getelementptr inbounds float, ptr %7, i64 %123
  %131 = call i32 @dma_p2p_opt(ptr %129, i64 12, i32 2048, i32 0, ptr %130, i64 12, i32 2048, i32 0, i1 false, i32 0, i32 %128)
  br label %133

132:                                              ; preds = %117
  br label %133

133:                                              ; preds = %120, %132
  %134 = phi i32 [ 0, %132 ], [ %131, %120 ]
  br label %135

135:                                              ; preds = %133
  call void @dma_wait_p2p(i32 %115)
  %136 = sdiv i64 %114, 12
  %137 = urem i64 %136, 2
  %138 = mul i64 %137, 6144
  %139 = mul i64 %114, 96
  %140 = add i64 %139, %104
  %141 = getelementptr inbounds float, ptr %7, i64 %138
  %142 = getelementptr inbounds float, ptr %5, i64 %72
  %143 = getelementptr inbounds float, ptr %6, i64 %140
  call void @matmul_micro_kernel(ptr %141, ptr %142, ptr %143, i64 512)
  %144 = add i64 %138, 3072
  %145 = add i64 %140, 576
  %146 = getelementptr inbounds float, ptr %7, i64 %144
  %147 = getelementptr inbounds float, ptr %6, i64 %145
  call void @matmul_micro_kernel(ptr %146, ptr %142, ptr %147, i64 512)
  br label %113

148:                                              ; preds = %113
  %149 = urem i64 %102, 2
  %150 = add i64 %149, 6
  %151 = trunc i64 %150 to i32
  %152 = getelementptr inbounds float, ptr %6, i64 %104
  %153 = getelementptr inbounds float, ptr %2, i64 %107
  %154 = call i32 @dma_p2p_opt(ptr %152, i64 240, i32 384, i32 0, ptr %153, i64 240, i32 384, i32 7296, i1 false, i32 0, i32 %151)
  %155 = icmp ne i64 %78, 0
  br i1 %155, label %156, label %157

156:                                              ; preds = %148
  call void @dma_wait_p2p(i32 %79)
  br label %157

157:                                              ; preds = %156, %148
  br label %77

158:                                              ; preds = %77
  call void @dma_wait_p2p(i32 %79)
  br label %49

159:                                              ; preds = %49
  call void @group_barrier(i32 0)
  br label %15

160:                                              ; preds = %15
  %161 = add i64 %9, 960
  br label %8

162:                                              ; preds = %8
  %163 = call i32 @vector_free(ptr %5)
  %164 = call i32 @vector_free(ptr %6)
  %165 = call i32 @scalar_free(ptr %7)
  ret ptr %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
