#!/bin/bash

Prepare(){
    echo "----------Compling for Scripts-----------"

    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_bf16_tensorop_gemm_T.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -o outT_c_bf16.exe
    nvcc ./examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm_T.cu -O0 -I./include -I./tools/util/include -I./examples/common -arch=sm_90 -o outT_c_f32.exe
}

GEMM_Runtime_BF16(){
    echo ""
    echo "Runtime Evaluation in BF16"
    echo "f" > ./control/0/FI.txt

    echo "------TCC_GEMM Kernels Runtime Performance-------"
    ./outT_c_bf16.exe --m=8192 --n=8192 --k=8192 --split=0 --iterations=1 --eval_mode=0 
    # rm boutT_bf16_A.exe bout_bf16_A.exe
}

GEMM_Runtime_FP32(){
    echo ""
    echo "Runtime Evaluation in FP32"
    echo "f" > ./control/0/FI.txt

    echo "------TCC_GEMM Kernels Runtime Performance-------"
    ./outT_c_f32.exe --m=8192 --n=8192 --k=8192 --split=0 --iterations=1

    rm outT_c_f32.exe
}

GEMM_Detection(){
    echo ""
    echo "Detection Evaluation in BF16"
    echo "t" > ./control/0/FI.txt
    echo "" > ./control/0/SM_checking_results.txt

    echo "Ground Truth Faulty SMID: $(head -n 1 ./control/0/plan.txt | awk '{print $1}')"

    echo "------TCC_GEMM Kernels Detection Performance-------"
    ./outT_c_bf16.exe --m=8192 --n=8192 --k=8192 --split=0 --iterations=1 --eval_mode=1
    python CoreChecer_kernels_evaluation/Detection.py
    
    rm outT_c_bf16.exe
}

Prepare

GEMM_Runtime_BF16
GEMM_Detection

GEMM_Runtime_FP32