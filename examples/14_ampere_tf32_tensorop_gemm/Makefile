outT_c_bf16.exe: ampere_bf16_tensorop_gemm_T.cu
	nvcc ampere_bf16_tensorop_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -o outT_c_bf16.exe

outT_baseline_bf16.exe: ampere_bf16_tensorop_gemm_T_baseline.cu
	nvcc ampere_bf16_tensorop_gemm_T_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -arch=sm_80 -o outT_baseline_bf16.exe

blout_bf16.exe: ampere_bf16_batched_gemm_baseline.cu
	nvcc ampere_bf16_batched_gemm_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -arch=sm_80 -o blout_bf16.exe

bloutT_bf16.exe: ampere_bf16_batched_gemm_T_baseline.cu
	nvcc ampere_bf16_batched_gemm_T_baseline.cu -O0 -I$HOME/origin_cutlass/cutlass/include -I$HOME/origin_cutlass/cutlass/tools/util/include -I$HOME/origin_cutlass/cutlass/examples/common -arch=sm_80 -o bloutT_bf16.exe

boutT_bf16.exe: ampere_bf16_batched_gemm_T.cu
	nvcc ampere_bf16_batched_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -o outT_c_bf16.exe -o boutT_bf16.exe

bout_bf16.exe: ampere_bf16_batched_gemm.cu
	nvcc ampere_bf16_batched_gemm.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -o outT_c_bf16.exe -o bout_bf16.exe