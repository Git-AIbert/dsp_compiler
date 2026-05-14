// // A1
// M 576
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// K 512
// M 576 A->A_gsm(multi_buffer)
// N 128 B->B_am()
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// M 576
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // F1
// M 864
// K 384 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 216 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 18 A_gsm->A_sm(multi_buffer) unroll(2)
// M 9 unroll(2)

// // C1
// M 600
// K 512 A->A_gsm(multi_buffer)
// N 160 B->B_am(multi_buffer)
// M 60 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // D1
// M 960
// K 512 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer)
// M 240 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12  A_gsm->A_sm(multi_buffer) unroll(2)
// M 6   unroll(2)

// // E1
// M 1120
// K 512 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer)
// M 280 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 14  A_gsm->A_sm(multi_buffer) unroll(2)
// M 7   unroll(2)

// // M1
// M 240
// K 512 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12  A_gsm->A_sm(multi_buffer) unroll(2)
// M 6   unroll(2)

// // M2
// M 280
// K 512 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 14  A_gsm->A_sm(multi_buffer) unroll(20)
// M 7   unroll(2)

// // M2'
// M 1120
// K 512 A->A_gsm(multi_buffer)
// M 280
// N 96  B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 14  A_gsm->A_sm(multi_buffer) unroll(2)
// M 7   unroll(2)

// // K1
// M 144
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // TVM
// M 12
// N 288 C->C_am() C_am->C()
// K 512 A->A_sm() B->B_am()
// M 6 unroll(2)
// N 96

// // N1
// M 240
// K 512 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12  A_gsm->A_sm(multi_buffer) unroll(2)
// M 6   unroll(2)

// K 512
// M 960 A->A_gsm(multi_buffer)
// N 96  B->B_am(multi_buffer)
// M 240 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // N2
// M 144
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// M 576
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // N3
// M 600
// K 512 A->A_gsm(multi_buffer)
// N 160 B->B_am(multi_buffer)
// M 60 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // N3'
// M 60
// K 512 A->A_gsm(multi_buffer)
// N 160 B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // A1' 261GFLOPS
// K 512
// M 576 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 6 A_gsm->A_sm(multi_buffer) unroll(4)

// // A2 293Gflops
// K 512
// M 1152 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer) unroll(2)
// M 6 unroll(2)

// // A2' 295Gflops
// K 512
// M 1152 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 6 A_gsm->A_sm(multi_buffer) unroll(4)

// // B1 304GFLOPS
// K 480
// M 640 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 160 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 16  A_gsm->A_sm(multi_buffer) unroll(2)
// M 8   unroll(2)

// K 480
// M 640 A->A_gsm(multi_buffer)
// N 128 B->B_am()
// M 160 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 16  A_gsm->A_sm(multi_buffer) unroll(10)
// M 8   unroll(2)

// // B1' 301GFLOPS
// K 480
// M 640 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 160 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 8  A_gsm->A_sm(multi_buffer) unroll(20)

// // B2 303GFLOPS
// K 480
// M 1280 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 160 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 16  A_gsm->A_sm(multi_buffer) unroll(10)
// M 8   unroll(2)

// // B2' 300GFLOPS
// K 480
// M 1280 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 160 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 8  A_gsm->A_sm(multi_buffer) unroll(20)

// // B3 306GFLOPS
// K 480
// M 768 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 192 C->C_am(multi_buffer) C_am->C(multi_buffer) unroll(4)
// M 16  A_gsm->A_sm(multi_buffer) unroll(12)
// M 8   unroll(2)

// // B3' 302GFLOPS
// K 480
// M 768 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 192 C->C_am(multi_buffer) C_am->C(multi_buffer) unroll(4)
// M 8  A_gsm->A_sm(multi_buffer) unroll(24)

// A3 192x4096x12288=305.56GFlops
// M 192
// K 2048 A->A_gsm(multi_buffer)
// N 96  C->C_am(multi_buffer) C_am->C(multi_buffer)
// K 512 B->B_am(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer)
// M 6  unroll(2)

// A4 64x576x50176=300.43Gflops 128x576x12544=301.14GFlops
// M 64 
// K 576 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer)
// M 6  unroll(2)

// A4' 128x576x12544=283.88GFlops
// M 120
// K 576 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12  A_gsm->A_sm(multi_buffer)
// M 6   unroll(2)

// B4 for 64x576x50176=265.17Gflops
// M 64 
// K 576 A->A_gsm(multi_buffer)
// N 256 C->C_am(multi_buffer) C_am->C()
// K 288 B->B_am(multi_buffer)
// M 16 A_gsm->A_sm(multi_buffer)
// M 4  unroll(6)

// // success
// M 576
// K 512 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer)
// M 6 unroll(2)

// // fail
// K 512
// M 576 A->A_gsm(multi_buffer)
// N 128 B->B_am(multi_buffer)
// M 144 C->C_am(multi_buffer) C_am->C(multi_buffer)
// M 12 A_gsm->A_sm(multi_buffer)
// M 6 unroll(2)

// multi_core
M 144
K 512 A->A_gsm(multi_buffer)
N 128 parallel(2) B->B_am(multi_buffer) C->C_am(multi_buffer) C_am->C(multi_buffer)
M 12 A_gsm->A_sm(multi_buffer) unroll(2)
M 6 unroll(2)