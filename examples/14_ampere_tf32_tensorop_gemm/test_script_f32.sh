#!/bin/bash

Prepare(){
    echo "----------Compling for F32 Testing-----------"
    nvcc ampere_tf32_tensorop_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_c_f32.exe
    nvcc ampere_tf32_tensorop_gemm_T_baseline.cu -O0 -I$HOME/cutlass/origin_cutlass/include -I$HOME/cutlass/origin_cutlass/tools/util/include -I$HOME/cutlass/origin_cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_baseline_f32.exe

    nvcc ampere_tf32_batched_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o boutT_f32.exe
    nvcc ampere_tf32_batched_gemm_T_baseline.cu -O0 -I$HOME/cutlass/origin_cutlass/include -I$HOME/cutlass/origin_cutlass/tools/util/include -I$HOME/cutlass/origin_cutlass/examples/common -std=c++17 -arch=sm_80 -o bloutT_f32.exe

    nvcc ampere_tf32_batched_gemm.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o bout_f32.exe
    nvcc ampere_tf32_batched_gemm_baseline.cu -O0 -I$HOME/cutlass/origin_cutlass/include -I$HOME/cutlass/origin_cutlass/tools/util/include -I$HOME/cutlass/origin_cutlass/examples/common -arch=sm_80 -std=c++17 -o blout_f32.exe
}

test_Transposed_GEMM_BF16(){
    echo ""
    echo "------Test Transposed GEMM-------"
    
    # nvcc ampere_bf16_tensorop_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_c_bf16.exe
    # nvcc ampere_bf16_tensorop_gemm_T_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_baseline_bf16.exe

    echo "***Attention: W(Q)/W(Attn_Output) (m = 8192, n = 4096, k = 4096)"
    echo "--SM-Checker:"
    ./outT_c_f32.exe --m=4096 --n=8192 --k=4096 --split=0 --iterations=5
    echo "--Baseline:"
    ./outT_baseline_f32.exe  --m=4096 --n=8192 --k=4096 --split=2 --iterations=5

    echo "***Attention: W(K)/W(V) (m = 8192, n = 1024, k = 4096)"
    echo "--SM-Checker:"
    ./outT_c_f32.exe --m=1024 --n=8192 --k=4096 --split=0 --iterations=5
    echo "--Baseline:"
    ./outT_baseline_f32.exe  --m=1024 --n=8192 --k=4096 --split=2 --iterations=5

    echo "***MLP: UP_Projection/Gate_Projection (m = 8192, n = 14336, k = 4096)"
    echo "--SM-Checker:"
    ./outT_c_bf16.exe --m=14336 --n=8192 --k=4096 --split=0 --iterations=5
    echo "--Baseline:"
    ./outT_baseline_f32.exe  --m=14336 --n=8192 --k=4096 --split=2 --iterations=5

    echo "***MLP: Down_Projection  (m = 8192, n = 4096, k = 14336)"
    echo "--SM-Checker:"
    ./outT_c_f32.exe --m=4096 --n=8192 --k=14336 --split=0 --iterations=5
    echo "--Baseline:"
    ./outT_baseline_f32.exe  --m=4096 --n=8192 --k=14336 --split=2 --iterations=5

    rm outT_c_f32.exe outT_baseline_f32.exe
}

test_Transposed_BGEMM_BF16(){
    echo ""
    echo "------Test Transposed BGEMM-------"

    # nvcc ampere_bf16_batched_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_c_bf16.exe -o boutT_bf16.exe
    # nvcc ampere_bf16_batched_gemm_T_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -std=c++17 -arch=sm_80 -o bloutT_bf16.exe

    echo "***Attention: Q * K_T (batch = 256, m = 1024, n = 1024, k = 128)"
    echo "--SM-Checker:"
    ./boutT_f32.exe --m=1024 --n=1024 --k=128 --batch=256 --split=1 --iterations=5 --validate=0
    echo "--Baseline:"
    ./bloutT_f32.exe  --m=1024 --n=1024 --k=128 --batch=256 --split=2 --iterations=5 --validate=0

    rm boutT_f32.exe bloutT_f32.exe
}

test_Non_Transposed_BGEMM_BF16(){
    echo ""
    echo "------Test Non Transposed BGEMM-------"

	# nvcc ampere_bf16_batched_gemm.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -std=c++17 -o outT_c_bf16.exe
    # nvcc ampere_bf16_batched_gemm_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -arch=sm_80 -std=c++17 -o blout_bf16.exe

    echo "***Attention: Attn_Prob * V (batch = 256, m = 1024, n = 128, k = 1024)"
    echo "--SM-Checker:"
    ./bout_f32.exe --m=128 --n=1024 --k=1024 --batch=256 --split=1 --iterations=5 --validate=0
    echo "--Baseline:"
    ./blout_f32.exe  --m=128 --n=1024 --k=1024 --batch=256 --split=2 --iterations=5 --validate=0

    rm bout_f32.exe blout_f32.exe
}

Prepare

test_Transposed_BGEMM_BF16

test_Non_Transposed_BGEMM_BF16

test_Transposed_GEMM_BF16