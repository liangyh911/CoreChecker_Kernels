#!/bin/bash

Prepare(){
    echo "----------Compling for Scripts-----------"

    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_bf16_batched_gemm_T_encode_A.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -std=c++17 -o boutT_bf16_A.exe
    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_bf16_batched_gemm_encode_A.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -std=c++17 -o bout_bf16_A.exe
    
    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_fp32_batched_gemm_T_encode_A.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -std=c++17 -o boutT_fp32_A.exe
    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_fp32_batched_gemm_encode_A.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -std=c++17 -o bout_fp32_A.exe

}

BGEMM_Runtime_BF16(){
    echo ""
    echo "Runtime Evaluation in BF16"
    echo "f" > ./control/0/FI.txt

    echo "------Transposed TCC_BGEMM Kernels Runtime Performance-------"
    ./boutT_bf16_A.exe --m=1024 --n=1024 --k=128 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=0

    echo "------Non-Transposed TCC_BGEMM Kernels Runtime Performance-------"
    ./bout_bf16_A.exe --m=128 --n=1024 --k=1024 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=0
    
    # rm boutT_bf16_A.exe bout_bf16_A.exe
}

BGEMM_Runtime_FP32(){
    echo ""
    echo "Runtime Evaluation in FP32"
    echo "f" > ./control/0/FI.txt

    echo "------Transposed TCC_BGEMM Kernels Runtime Performance-------"
    ./boutT_fp32_A.exe --m=1024 --n=1024 --k=128 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=0

    echo "------Non-Transposed TCC_BGEMM Kernels Runtime Performance-------"
    ./bout_fp32_A.exe --m=128 --n=1024 --k=1024 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=0
    
    rm boutT_fp32_A.exe bout_fp32_A.exe
}

BGEMM_Detection(){
    echo ""
    echo "Detection Evaluation in BF16"
    echo "t" > ./control/0/FI.txt
    echo "" > ./control/0/SM_checking_results.txt

    echo "Ground Truth Faulty SMID: $(head -n 1 ./control/0/plan.txt | awk '{print $1}')"
    
    echo "------Transposed TCC_BGEMM Kernels Detection Performance-------"
    ./boutT_bf16_A.exe --m=1024 --n=1024 --k=128 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=1
    python CoreChecer_kernels_evaluation/Detection.py

    echo "------Non-Transposed TCC_BGEMM Kernels Detection Performance-------"
    ./bout_bf16_A.exe --m=128 --n=1024 --k=1024 --batch=256 --split=1 --iterations=1 --validate=0 --eval_mode=1
    python CoreChecer_kernels_evaluation/Detection.py
    
    rm boutT_bf16_A.exe bout_bf16_A.exe
}

Prepare

BGEMM_Runtime_BF16
BGEMM_Detection

BGEMM_Runtime_FP32