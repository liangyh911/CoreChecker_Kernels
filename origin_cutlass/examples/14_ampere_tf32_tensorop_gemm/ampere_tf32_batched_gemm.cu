/***************************************************************************************************
* Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**************************************************************************************************/

#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include "helper.h"


#pragma warning( disable : 4503)

/*
This example demonstrates how to use cutlass to compute a batched strided gemm in two different ways:
  1. By specifying pointers to the first matrices of the batch and the stride between the consecutive
    matrices of the batch (this is called a strided batched gemm).
  2. By copying pointers to all matrices of the batch to the device memory (this is called an array gemm).
In this example, both A and B matrix are non-transpose and column major matrix
batched_C = batched_A x batched_B
As an example, matrix C can be seen as
-----------------------------------------------------------
(0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
-----------------------------------------------------------
(0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------------------------------------
(0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
-----------------------------------------------------------
(0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
-----------------------------------------------------------
(0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
-----------------------------------------------------------
(0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
-----------------------------------------------------------
          batch 0          |           batch 1
where we denote each element with (batch_idx, row_idx, column_idx)
In this example, batch size is 2, M is 6 and N is 3
The stride (batch_stride_C) between the first element of two batches is ldc * n

matrix A can be seen as
---------------------------------------
(0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
---------------------------------------
(0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
---------------------------------------
(0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
---------------------------------------
(0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
---------------------------------------
(0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
---------------------------------------
(0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
---------------------------------------
    batch 0      |      batch 1
, where batch size is 2, M is 6 and K is 2
The stride (batch_stride_A) between the first element of two batches is lda * k

matrix B can be seen as
-----------------------------
(0,0,0) | (0,0,1) | (0,0,2) |
----------------------------- batch 0
(0,1,0) | (0,1,1) | (0,1,2) |
-------------------------------------
(1,0,0) | (1,0,1) | (1,0,2) |
----------------------------- batch 1
(1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------
, where the batch size is 2, N is 3 and K is 2
The stride (batch_stride_B) between the first element of two batches is k

nvcc ampere_tf32_batched_gemm.cu -O0 -I/home/yuhangl/origin_cutlass/cutlass/include -I/home/yuhangl/origin_cutlass/cutlass/tools/util/include -I/home/yuhangl/origin_cutlass/cutlass/examples/common -arch=sm_90 -o blout.exe

*/

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4; 

// cudaError_t cutlass_array_sgemm(
//   int m,
//   int n,
//   int k,
//   float alpha,
//   float const * const *A,
//   int lda,
//   float const * const *B,
//   int ldb,
//   float * const *C,
//   int ldc,
//   float beta,
//   int batch_count) {

//   using Gemm = cutlass::gemm::device::GemmArray<
//     float, cutlass::layout::ColumnMajor,
//     float, cutlass::layout::ColumnMajor,
//     float, cutlass::layout::ColumnMajor
//   >;

//   Gemm gemm_op;

//   cutlass::Status status = gemm_op({
//     {m, n, k},
//     A, lda,
//     B, ldb,
//     C, ldc,
//     C, ldc,
//     {alpha, beta},
//     batch_count
//   });

//   if (status != cutlass::Status::kSuccess) {
//     return cudaErrorUnknown;
//   }

//   return cudaSuccess;
// }

cudaError_t cutlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float beta,
  int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp
    // SwizzleThreadBlock,
    // NumStages
  >;

  // typename Gemm::Arguments arguments{
  //     {m, n, k},
  //     {A, lda}, 
  //     batch_stride_A,
  //     {B, ldb}, 
  //     batch_stride_B,
  //     {C, ldc}, 
  //     batch_stride_C,
  //     {C, ldc}, 
  //     batch_stride_C,
  //     {alpha, beta},
  //     batch_count
  // };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  // size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  // Check the problem size is supported or not 
  // cutlass::Status status = gemm_op.can_implement(arguments);
  // CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  // status = gemm_op.initialize(arguments, workspace.get());
  // CUTLASS_CHECK(status);

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template<typename T> 
cudaError_t strided_batched_gemm_nn_reference(
  int m,
  int n,
  int k,
  T alpha,
  std::vector<T> const &A, 
  int lda,
  long long int batch_stride_A,
  std::vector<T> const &B, 
  int ldb,
  long long int batch_stride_B,
  std::vector<T> &C, 
  int ldc,
  long long int batch_stride_C,
  T beta,
  int batch_count) {
  /*
  strided batched gemm NN
  */
  
  cudaError_t result = cudaSuccess;

  if (A.size() < size_t(lda * k * batch_count)) {
    std::cout << "the size of A is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (B.size() < size_t(ldb * n)) {
    std::cout << "the size of B is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (C.size() < size_t(ldc * n * batch_count)) {
    std::cout << "the size of C is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  
  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

template<typename T>
void outputChk(std::vector<T> &A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col){
  for(int i = 0; i < nb; i++){
    printf("[ \n");
    for(int r = 0; r < row; r++){
      printf("|");
      for(int c = 0; c < col; c++){
        printf("%.6f", (float)(A[i*stride + c*ld + r]));
        printf(", ");
      }
      printf("!\n");
    }
    printf("]\n");
  }
}

cudaError_t run_batched_gemm(bool use_array) {

  const char* gemm_desc = use_array ? "array" : "strided batched";
  std::cout << "Running " << gemm_desc << " gemm" << std::endl;

  // Arbitrary problem size
  int const m = 128;
  int const n = 3;
  int const k = 1024;
  int const batch_count = 8;

  printf("%d x %d x %d x %d TF32 tensor op Matrix Multiply\n", \
    batch_count, m, n, k);

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  // alpha and beta
  float alpha = 1.0f;
  float beta = 0.0f;

  cudaError_t result = cudaSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  result = cudaMalloc(&A, count_A * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&B, count_B * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&C, count_C * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  // Limit range to avoid floating-point errors
  int const kRange = 8;

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange);
      }
    }
  }
  // fill B
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange);
      }
    }
  }
  
  // outputChk(host_B, batch_count, ldb, batch_stride_B, k, n);

  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
      }
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  // run cutlass
  // if (use_array) {
  //   // allocate the host memory for the pointers to the matrices of the batch
  //   std::vector<float*> host_ptr_A(batch_count);
  //   std::vector<float*> host_ptr_B(batch_count);
  //   std::vector<float*> host_ptr_C(batch_count);

  //   // permute the batch elements to emphasize that GemmArray does not depend on matrices being separated by a fixed stride
  //   std::vector<size_t> permutation = {14, 11, 3, 10, 1, 13, 9, 4, 6, 16, 8, 15, 7, 12, 0, 2, 5};
  //   for (size_t b_idx = 0; b_idx < batch_count; b_idx++) {
  //     host_ptr_A[b_idx] = A + permutation[b_idx] * batch_stride_A;
  //     host_ptr_B[b_idx] = B + permutation[b_idx] * batch_stride_B;
  //     host_ptr_C[b_idx] = C + permutation[b_idx] * batch_stride_C;
  //   }

  //   // allocate the corresponding device memory
  //   float const **ptr_A;
  //   float const **ptr_B;
  //   float **ptr_C;

  //   result = cudaMalloc(&ptr_A, batch_count * sizeof(float*));
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMalloc result = " << result << std::endl;
  //     return result;
  //   }
  //   result = cudaMalloc(&ptr_B, batch_count * sizeof(float*));
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMalloc result = " << result << std::endl;
  //     return result;
  //   }
  //   result = cudaMalloc(&ptr_C, batch_count * sizeof(float*));
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMalloc result = " << result << std::endl;
  //     return result;
  //   }

  //   // copy the matrix pointers to the device
  //   result = cudaMemcpy(ptr_A, host_ptr_A.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMemcpy result = " << result << std::endl;
  //     return result;
  //   }
  //   result = cudaMemcpy(ptr_B, host_ptr_B.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMemcpy result = " << result << std::endl;
  //     return result;
  //   }
  //   result = cudaMemcpy(ptr_C, host_ptr_C.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
  //   if (result != cudaSuccess) {
  //     std::cerr << "cudaMemcpy result = " << result << std::endl;
  //     return result;
  //   }

  //   result = cutlass_array_sgemm(m, n, k, alpha, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, beta, batch_count);

  //   if (result != cudaSuccess)
  //     return result;
  // } else {
    for(int i = 0; i < 5; i++){
      result = cutlass_strided_batched_sgemm(
        m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
        beta, batch_count);
      if (result != cudaSuccess)
        return result;
    }
  // }

  // copy device memory to host
  result = cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  //compare with reference code
  // result = strided_batched_gemm_nn_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C,
  //   beta, batch_count);
  // if (result != 0)
  //   return result;

  // Expect bit-level accuracy for this simple example
  if (ref_C != result_C) {
    std::cout << "CUTLASS " << gemm_desc << " gemm does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  // free memory
  result = cudaFree(A);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(B);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(C);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }

  return result;
}

int main() {

  cudaError_t result = cudaSuccess;
  for (bool use_array : {false}) {
    result = run_batched_gemm(use_array);
    if (result == cudaSuccess) {
      std::cout << "Passed." << std::endl;
    } else {
      break;
    }
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}
