; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@gsm_0 = internal global [589824 x float] undef, section ".gsm"

declare i32 @vector_free(ptr)

declare i32 @scalar_free(ptr)

declare void @relu_micro_kernel_n128(ptr, i64, i64)

declare void @elemwise_add_micro_kernel_v16(ptr, ptr, ptr, i64, i64, i64)

declare void @micro_kernel_asm_r6c128(ptr, ptr, ptr, i64, i64, i64)

declare void @group_barrier(i32)

declare void @dma_wait_p2p(i32)

declare i32 @dma_p2p_opt(ptr, i64, i32, i32, ptr, i64, i32, i32, i1, i32, i32)

declare ptr @scalar_malloc(i32)

declare ptr @vector_malloc(i32)

declare i32 @get_thread_id()

declare void @set_prir(i64)

define void @matmul_add_relu(ptr %0, ptr %1, ptr %2, ptr %3)  section ".global" {
  call void @set_prir(i64 3)
  %5 = call i32 @get_thread_id()
  %6 = icmp eq i32 %5, 0
  %7 = call ptr @vector_malloc(i32 524288)
  %8 = call ptr @vector_malloc(i32 147456)
  %9 = call ptr @vector_malloc(i32 98304)
  %10 = call ptr @scalar_malloc(i32 49152)
  br i1 %6, label %11, label %13

11:                                               ; preds = %4
  %12 = call i32 @dma_p2p_opt(ptr %0, i64 512, i32 2048, i32 1024, ptr @gsm_0, i64 512, i32 2048, i32 0, i1 false, i32 0, i32 8)
  br label %14

13:                                               ; preds = %4
  br label %14

14:                                               ; preds = %11, %13
  %15 = phi i32 [ 0, %13 ], [ %12, %11 ]
  br label %16

16:                                               ; preds = %14
  br i1 %6, label %17, label %20

17:                                               ; preds = %16
  %18 = getelementptr inbounds float, ptr %0, i32 512
  %19 = call i32 @dma_p2p_opt(ptr %18, i64 512, i32 1024, i32 2048, ptr getelementptr inbounds (float, ptr @gsm_0, i32 294912), i64 512, i32 1024, i32 1024, i1 false, i32 0, i32 9)
  call void @dma_wait_p2p(i32 %15)
  br label %21

20:                                               ; preds = %16
  br label %21

21:                                               ; preds = %17, %20
  %22 = phi i32 [ 0, %20 ], [ %19, %17 ]
  br label %23

23:                                               ; preds = %21
  call void @group_barrier(i32 0)
  %24 = sext i32 %5 to i64
  %25 = mul i64 %24, 128
  %26 = sdiv i64 %25, 1024
  %27 = urem i64 %26, 2
  %28 = mul i64 %27, 65536
  %29 = add i64 %27, 6
  %30 = trunc i64 %29 to i32
  %31 = getelementptr inbounds float, ptr %1, i64 %25
  %32 = getelementptr inbounds float, ptr %7, i64 %28
  %33 = call i32 @dma_p2p_opt(ptr %31, i64 512, i32 512, i32 11776, ptr %32, i64 512, i32 512, i32 0, i1 false, i32 0, i32 %30)
  br label %34

34:                                               ; preds = %183, %23
  %35 = phi i64 [ %39, %183 ], [ %25, %23 ]
  %36 = phi i32 [ %52, %183 ], [ %33, %23 ]
  %37 = icmp slt i64 %35, 3072
  br i1 %37, label %38, label %184

38:                                               ; preds = %34
  %39 = add i64 %35, 1024
  %40 = icmp slt i64 %39, 3072
  br i1 %40, label %41, label %50

41:                                               ; preds = %38
  %42 = sdiv i64 %39, 1024
  %43 = urem i64 %42, 2
  %44 = mul i64 %43, 65536
  %45 = add i64 %43, 6
  %46 = trunc i64 %45 to i32
  %47 = getelementptr inbounds float, ptr %1, i64 %39
  %48 = getelementptr inbounds float, ptr %7, i64 %44
  %49 = call i32 @dma_p2p_opt(ptr %47, i64 512, i32 512, i32 11776, ptr %48, i64 512, i32 512, i32 0, i1 false, i32 0, i32 %46)
  br label %51

50:                                               ; preds = %38
  br label %51

51:                                               ; preds = %41, %50
  %52 = phi i32 [ 0, %50 ], [ %49, %41 ]
  br label %53

53:                                               ; preds = %51
  call void @dma_wait_p2p(i32 %36)
  %54 = sdiv i64 %35, 1024
  %55 = urem i64 %54, 2
  %56 = mul i64 %55, 65536
  %57 = getelementptr inbounds float, ptr %3, i64 %35
  %58 = call i32 @dma_p2p_opt(ptr %57, i64 96, i32 512, i32 11776, ptr %8, i64 96, i32 512, i32 0, i1 false, i32 0, i32 2)
  br label %59

59:                                               ; preds = %182, %53
  %60 = phi i64 [ %65, %182 ], [ 0, %53 ]
  %61 = phi i32 [ %179, %182 ], [ 0, %53 ]
  %62 = phi i32 [ %84, %182 ], [ %58, %53 ]
  %63 = icmp slt i64 %60, 512
  br i1 %63, label %64, label %183

64:                                               ; preds = %59
  %65 = add i64 %60, 96
  %66 = icmp slt i64 %65, 512
  br i1 %66, label %67, label %82

67:                                               ; preds = %64
  %68 = sdiv i64 %65, 96
  %69 = urem i64 %68, 3
  %70 = mul i64 %65, -1
  %71 = add i64 %70, 512
  %72 = call i64 @llvm.smin.i64(i64 %71, i64 96)
  %73 = mul i64 %65, 3072
  %74 = add i64 %35, %73
  %75 = mul i64 %69, 12288
  %76 = urem i64 %68, 2
  %77 = add i64 %76, 2
  %78 = trunc i64 %77 to i32
  %79 = getelementptr inbounds float, ptr %3, i64 %74
  %80 = getelementptr inbounds float, ptr %8, i64 %75
  %81 = call i32 @dma_p2p_opt(ptr %79, i64 %72, i32 512, i32 11776, ptr %80, i64 %72, i32 512, i32 0, i1 false, i32 0, i32 %78)
  br label %83

82:                                               ; preds = %64
  br label %83

83:                                               ; preds = %67, %82
  %84 = phi i32 [ 0, %82 ], [ %81, %67 ]
  br label %85

85:                                               ; preds = %83
  call void @dma_wait_p2p(i32 %62)
  %86 = sdiv i64 %60, 96
  %87 = urem i64 %86, 3
  %88 = mul i64 %60, -1
  %89 = add i64 %88, 512
  %90 = call i64 @llvm.smin.i64(i64 %89, i64 96)
  %91 = mul i64 %60, 3072
  %92 = add i64 %35, %91
  %93 = mul i64 %87, 12288
  %94 = call i64 @llvm.smin.i64(i64 %89, i64 12)
  %95 = call i64 @llvm.smin.i64(i64 %94, i64 96)
  %96 = mul i64 %60, 512
  %97 = getelementptr inbounds float, ptr @gsm_0, i64 %96
  %98 = call i32 @dma_p2p_opt(ptr %97, i64 %95, i32 2048, i32 0, ptr %10, i64 %95, i32 2048, i32 0, i1 false, i32 0, i32 0)
  br label %99

99:                                               ; preds = %172, %85
  %100 = phi i64 [ %104, %172 ], [ 0, %85 ]
  %101 = phi i32 [ %121, %172 ], [ %98, %85 ]
  %102 = icmp slt i64 %100, %90
  br i1 %102, label %103, label %173

103:                                              ; preds = %99
  %104 = add i64 %100, 12
  %105 = icmp slt i64 %104, %90
  br i1 %105, label %106, label %119

106:                                              ; preds = %103
  %107 = sdiv i64 %104, 12
  %108 = urem i64 %107, 2
  %109 = mul i64 %104, -1
  %110 = add i64 %109, %90
  %111 = call i64 @llvm.smin.i64(i64 %110, i64 12)
  %112 = mul i64 %104, 512
  %113 = add i64 %112, %96
  %114 = mul i64 %108, 6144
  %115 = trunc i64 %108 to i32
  %116 = getelementptr inbounds float, ptr @gsm_0, i64 %113
  %117 = getelementptr inbounds float, ptr %10, i64 %114
  %118 = call i32 @dma_p2p_opt(ptr %116, i64 %111, i32 2048, i32 0, ptr %117, i64 %111, i32 2048, i32 0, i1 false, i32 0, i32 %115)
  br label %120

119:                                              ; preds = %103
  br label %120

120:                                              ; preds = %106, %119
  %121 = phi i32 [ 0, %119 ], [ %118, %106 ]
  br label %122

122:                                              ; preds = %120
  call void @dma_wait_p2p(i32 %101)
  %123 = sdiv i64 %100, 12
  %124 = urem i64 %123, 2
  %125 = mul i64 %100, -1
  %126 = add i64 %125, %90
  %127 = call i64 @llvm.smin.i64(i64 %126, i64 12)
  %128 = add i64 %127, 5
  %129 = udiv i64 %128, 6
  %130 = srem i64 %129, 2
  %131 = sub i64 %129, %130
  %132 = mul i64 %131, 6
  br label %133

133:                                              ; preds = %136, %122
  %134 = phi i64 [ %155, %136 ], [ 0, %122 ]
  %135 = icmp slt i64 %134, %132
  br i1 %135, label %136, label %156

136:                                              ; preds = %133
  %137 = mul i64 %134, 512
  %138 = mul i64 %124, 6144
  %139 = add i64 %137, %138
  %140 = mul i64 %134, 128
  %141 = mul i64 %100, 128
  %142 = add i64 %140, %141
  %143 = add i64 %142, %93
  %144 = getelementptr inbounds float, ptr %10, i64 %139
  %145 = getelementptr inbounds float, ptr %7, i64 %56
  %146 = getelementptr inbounds float, ptr %8, i64 %143
  call void @micro_kernel_asm_r6c128(ptr %144, ptr %145, ptr %146, i64 512, i64 512, i64 128)
  %147 = add i64 %134, 6
  %148 = mul i64 %147, 512
  %149 = add i64 %148, %138
  %150 = mul i64 %147, 128
  %151 = add i64 %150, %141
  %152 = add i64 %151, %93
  %153 = getelementptr inbounds float, ptr %10, i64 %149
  %154 = getelementptr inbounds float, ptr %8, i64 %152
  call void @micro_kernel_asm_r6c128(ptr %153, ptr %145, ptr %154, i64 512, i64 512, i64 128)
  %155 = add i64 %134, 12
  br label %133

156:                                              ; preds = %133
  br label %157

157:                                              ; preds = %160, %156
  %158 = phi i64 [ %171, %160 ], [ %132, %156 ]
  %159 = icmp slt i64 %158, %127
  br i1 %159, label %160, label %172

160:                                              ; preds = %157
  %161 = mul i64 %158, 512
  %162 = mul i64 %124, 6144
  %163 = add i64 %161, %162
  %164 = mul i64 %158, 128
  %165 = mul i64 %100, 128
  %166 = add i64 %164, %165
  %167 = add i64 %166, %93
  %168 = getelementptr inbounds float, ptr %10, i64 %163
  %169 = getelementptr inbounds float, ptr %7, i64 %56
  %170 = getelementptr inbounds float, ptr %8, i64 %167
  call void @micro_kernel_asm_r6c128(ptr %168, ptr %169, ptr %170, i64 512, i64 512, i64 128)
  %171 = add i64 %158, 6
  br label %157

172:                                              ; preds = %157
  br label %99

173:                                              ; preds = %99
  %174 = urem i64 %86, 2
  %175 = add i64 %174, 4
  %176 = trunc i64 %175 to i32
  %177 = getelementptr inbounds float, ptr %8, i64 %93
  %178 = getelementptr inbounds float, ptr %3, i64 %92
  %179 = call i32 @dma_p2p_opt(ptr %177, i64 %90, i32 512, i32 0, ptr %178, i64 %90, i32 512, i32 11776, i1 false, i32 0, i32 %176)
  %180 = icmp ne i64 %60, 0
  br i1 %180, label %181, label %182

181:                                              ; preds = %173
  call void @dma_wait_p2p(i32 %61)
  br label %182

182:                                              ; preds = %181, %173
  br label %59

183:                                              ; preds = %59
  call void @dma_wait_p2p(i32 %61)
  br label %34

184:                                              ; preds = %34
  call void @group_barrier(i32 0)
  br i1 %6, label %185, label %186

185:                                              ; preds = %184
  call void @dma_wait_p2p(i32 %22)
  br label %186

186:                                              ; preds = %185, %184
  call void @group_barrier(i32 0)
  %187 = add i64 %25, 786432
  %188 = add i64 %27, 8
  %189 = trunc i64 %188 to i32
  %190 = getelementptr inbounds float, ptr %1, i64 %187
  %191 = call i32 @dma_p2p_opt(ptr %190, i64 512, i32 512, i32 11776, ptr %32, i64 512, i32 512, i32 0, i1 false, i32 0, i32 %189)
  br label %192

192:                                              ; preds = %354, %186
  %193 = phi i64 [ %197, %354 ], [ %25, %186 ]
  %194 = phi i32 [ %211, %354 ], [ %191, %186 ]
  %195 = icmp slt i64 %193, 3072
  br i1 %195, label %196, label %355

196:                                              ; preds = %192
  %197 = add i64 %193, 1024
  %198 = icmp slt i64 %197, 3072
  br i1 %198, label %199, label %209

199:                                              ; preds = %196
  %200 = sdiv i64 %197, 1024
  %201 = urem i64 %200, 2
  %202 = mul i64 %201, 65536
  %203 = add i64 %197, 786432
  %204 = add i64 %201, 8
  %205 = trunc i64 %204 to i32
  %206 = getelementptr inbounds float, ptr %1, i64 %203
  %207 = getelementptr inbounds float, ptr %7, i64 %202
  %208 = call i32 @dma_p2p_opt(ptr %206, i64 512, i32 512, i32 11776, ptr %207, i64 512, i32 512, i32 0, i1 false, i32 0, i32 %205)
  br label %210

209:                                              ; preds = %196
  br label %210

210:                                              ; preds = %199, %209
  %211 = phi i32 [ 0, %209 ], [ %208, %199 ]
  br label %212

212:                                              ; preds = %210
  call void @dma_wait_p2p(i32 %194)
  %213 = sdiv i64 %193, 1024
  %214 = urem i64 %213, 2
  %215 = mul i64 %214, 65536
  %216 = getelementptr inbounds float, ptr %3, i64 %193
  %217 = call i32 @dma_p2p_opt(ptr %216, i64 96, i32 512, i32 11776, ptr %8, i64 96, i32 512, i32 0, i1 false, i32 0, i32 2)
  %218 = getelementptr inbounds float, ptr %2, i64 %193
  %219 = call i32 @dma_p2p_opt(ptr %218, i64 96, i32 512, i32 11776, ptr %9, i64 96, i32 512, i32 0, i1 false, i32 0, i32 4)
  br label %220

220:                                              ; preds = %353, %212
  %221 = phi i64 [ %227, %353 ], [ 0, %212 ]
  %222 = phi i32 [ %350, %353 ], [ 0, %212 ]
  %223 = phi i32 [ %252, %353 ], [ %217, %212 ]
  %224 = phi i32 [ %253, %353 ], [ %219, %212 ]
  %225 = icmp slt i64 %221, 512
  br i1 %225, label %226, label %354

226:                                              ; preds = %220
  %227 = add i64 %221, 96
  %228 = icmp slt i64 %227, 512
  br i1 %228, label %229, label %250

229:                                              ; preds = %226
  %230 = sdiv i64 %227, 96
  %231 = urem i64 %230, 3
  %232 = urem i64 %230, 2
  %233 = mul i64 %227, -1
  %234 = add i64 %233, 512
  %235 = call i64 @llvm.smin.i64(i64 %234, i64 96)
  %236 = mul i64 %227, 3072
  %237 = add i64 %193, %236
  %238 = mul i64 %232, 12288
  %239 = mul i64 %231, 12288
  %240 = add i64 %232, 2
  %241 = trunc i64 %240 to i32
  %242 = getelementptr inbounds float, ptr %3, i64 %237
  %243 = getelementptr inbounds float, ptr %8, i64 %239
  %244 = call i32 @dma_p2p_opt(ptr %242, i64 %235, i32 512, i32 11776, ptr %243, i64 %235, i32 512, i32 0, i1 false, i32 0, i32 %241)
  %245 = add i64 %232, 4
  %246 = trunc i64 %245 to i32
  %247 = getelementptr inbounds float, ptr %2, i64 %237
  %248 = getelementptr inbounds float, ptr %9, i64 %238
  %249 = call i32 @dma_p2p_opt(ptr %247, i64 %235, i32 512, i32 11776, ptr %248, i64 %235, i32 512, i32 0, i1 false, i32 0, i32 %246)
  br label %251

250:                                              ; preds = %226
  br label %251

251:                                              ; preds = %229, %250
  %252 = phi i32 [ 0, %250 ], [ %244, %229 ]
  %253 = phi i32 [ 0, %250 ], [ %249, %229 ]
  br label %254

254:                                              ; preds = %251
  call void @dma_wait_p2p(i32 %223)
  call void @dma_wait_p2p(i32 %224)
  %255 = sdiv i64 %221, 96
  %256 = urem i64 %255, 3
  %257 = urem i64 %255, 2
  %258 = mul i64 %221, -1
  %259 = add i64 %258, 512
  %260 = call i64 @llvm.smin.i64(i64 %259, i64 96)
  %261 = mul i64 %221, 3072
  %262 = add i64 %193, %261
  %263 = mul i64 %257, 12288
  %264 = mul i64 %256, 12288
  %265 = call i64 @llvm.smin.i64(i64 %259, i64 12)
  %266 = call i64 @llvm.smin.i64(i64 %265, i64 96)
  %267 = mul i64 %221, 512
  %268 = getelementptr inbounds float, ptr @gsm_0, i64 %267
  %269 = call i32 @dma_p2p_opt(ptr %268, i64 %266, i32 2048, i32 0, ptr %10, i64 %266, i32 2048, i32 0, i1 false, i32 0, i32 0)
  br label %270

270:                                              ; preds = %343, %254
  %271 = phi i64 [ %275, %343 ], [ 0, %254 ]
  %272 = phi i32 [ %292, %343 ], [ %269, %254 ]
  %273 = icmp slt i64 %271, %260
  br i1 %273, label %274, label %344

274:                                              ; preds = %270
  %275 = add i64 %271, 12
  %276 = icmp slt i64 %275, %260
  br i1 %276, label %277, label %290

277:                                              ; preds = %274
  %278 = sdiv i64 %275, 12
  %279 = urem i64 %278, 2
  %280 = mul i64 %275, -1
  %281 = add i64 %280, %260
  %282 = call i64 @llvm.smin.i64(i64 %281, i64 12)
  %283 = mul i64 %275, 512
  %284 = add i64 %283, %267
  %285 = mul i64 %279, 6144
  %286 = trunc i64 %279 to i32
  %287 = getelementptr inbounds float, ptr @gsm_0, i64 %284
  %288 = getelementptr inbounds float, ptr %10, i64 %285
  %289 = call i32 @dma_p2p_opt(ptr %287, i64 %282, i32 2048, i32 0, ptr %288, i64 %282, i32 2048, i32 0, i1 false, i32 0, i32 %286)
  br label %291

290:                                              ; preds = %274
  br label %291

291:                                              ; preds = %277, %290
  %292 = phi i32 [ 0, %290 ], [ %289, %277 ]
  br label %293

293:                                              ; preds = %291
  call void @dma_wait_p2p(i32 %272)
  %294 = sdiv i64 %271, 12
  %295 = urem i64 %294, 2
  %296 = mul i64 %271, -1
  %297 = add i64 %296, %260
  %298 = call i64 @llvm.smin.i64(i64 %297, i64 12)
  %299 = add i64 %298, 5
  %300 = udiv i64 %299, 6
  %301 = srem i64 %300, 2
  %302 = sub i64 %300, %301
  %303 = mul i64 %302, 6
  br label %304

304:                                              ; preds = %307, %293
  %305 = phi i64 [ %326, %307 ], [ 0, %293 ]
  %306 = icmp slt i64 %305, %303
  br i1 %306, label %307, label %327

307:                                              ; preds = %304
  %308 = mul i64 %305, 512
  %309 = mul i64 %295, 6144
  %310 = add i64 %308, %309
  %311 = mul i64 %305, 128
  %312 = mul i64 %271, 128
  %313 = add i64 %311, %312
  %314 = add i64 %313, %264
  %315 = getelementptr inbounds float, ptr %10, i64 %310
  %316 = getelementptr inbounds float, ptr %7, i64 %215
  %317 = getelementptr inbounds float, ptr %8, i64 %314
  call void @micro_kernel_asm_r6c128(ptr %315, ptr %316, ptr %317, i64 512, i64 512, i64 128)
  %318 = add i64 %305, 6
  %319 = mul i64 %318, 512
  %320 = add i64 %319, %309
  %321 = mul i64 %318, 128
  %322 = add i64 %321, %312
  %323 = add i64 %322, %264
  %324 = getelementptr inbounds float, ptr %10, i64 %320
  %325 = getelementptr inbounds float, ptr %8, i64 %323
  call void @micro_kernel_asm_r6c128(ptr %324, ptr %316, ptr %325, i64 512, i64 512, i64 128)
  %326 = add i64 %305, 12
  br label %304

327:                                              ; preds = %304
  br label %328

328:                                              ; preds = %331, %327
  %329 = phi i64 [ %342, %331 ], [ %303, %327 ]
  %330 = icmp slt i64 %329, %298
  br i1 %330, label %331, label %343

331:                                              ; preds = %328
  %332 = mul i64 %329, 512
  %333 = mul i64 %295, 6144
  %334 = add i64 %332, %333
  %335 = mul i64 %329, 128
  %336 = mul i64 %271, 128
  %337 = add i64 %335, %336
  %338 = add i64 %337, %264
  %339 = getelementptr inbounds float, ptr %10, i64 %334
  %340 = getelementptr inbounds float, ptr %7, i64 %215
  %341 = getelementptr inbounds float, ptr %8, i64 %338
  call void @micro_kernel_asm_r6c128(ptr %339, ptr %340, ptr %341, i64 512, i64 512, i64 128)
  %342 = add i64 %329, 6
  br label %328

343:                                              ; preds = %328
  br label %270

344:                                              ; preds = %270
  %345 = getelementptr inbounds float, ptr %8, i64 %264
  %346 = getelementptr inbounds float, ptr %9, i64 %263
  call void @elemwise_add_micro_kernel_v16(ptr %345, ptr %346, ptr %345, i64 %260, i64 128, i64 128)
  call void @relu_micro_kernel_n128(ptr %345, i64 %260, i64 128)
  %347 = add i64 %257, 6
  %348 = trunc i64 %347 to i32
  %349 = getelementptr inbounds float, ptr %3, i64 %262
  %350 = call i32 @dma_p2p_opt(ptr %345, i64 %260, i32 512, i32 0, ptr %349, i64 %260, i32 512, i32 11776, i1 false, i32 0, i32 %348)
  %351 = icmp ne i64 %221, 0
  br i1 %351, label %352, label %353

352:                                              ; preds = %344
  call void @dma_wait_p2p(i32 %222)
  br label %353

353:                                              ; preds = %352, %344
  br label %220

354:                                              ; preds = %220
  call void @dma_wait_p2p(i32 %222)
  br label %192

355:                                              ; preds = %192
  call void @group_barrier(i32 0)
  %356 = call i32 @scalar_free(ptr %10)
  %357 = call i32 @vector_free(ptr %9)
  %358 = call i32 @vector_free(ptr %8)
  %359 = call i32 @vector_free(ptr %7)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
