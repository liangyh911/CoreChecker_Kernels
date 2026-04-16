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

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.

nvcc ampere_tf32_tensorop_gemm_T.cu -O0 -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -I$HOME/cutlass/examples/common -arch=sm_80 -o outT_c_f32.exe
./outT_c_f32.exe --m=8192 --n=8192 --k=8192 --split=0 --iterations=1

nvcc ampere_tf32_tensorop_gemm_T.cu -O0 -I/home/yuhangl/origin_cutlass/cutlass/include -I/home/yuhangl/origin_cutlass/cutlass/tools/util/include -I/home/yuhangl/origin_cutlass/cutlass/examples/common -arch=sm_90 -o outT_baseline_bf16.exe

ncu -f -o transpose --set full ./outT_c_f32.exe --m=4096 --n=8192 --k=4096 --split=0 --iterations=1

./outT_c.exe --m=14336 --n=8192 --k=4096 --split=0 --iterations=5

*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_baseline.h"
#include "cutlass/gemm/device/gemm_sign.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "helper.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  cutlass::gemm::GemmCoord problem_size_baseline;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;

  int partition;
  int if_split_phase;
  
  Options():
    help(false),
    // problem_size({5120, 4096, 4096}),
    problem_size({128*3, 128*3, 128*3}),
    problem_size_baseline({72,64,72}),
    // problem_size({72,64,72}),
    batch_count(1),
    reference_check(true),
    // iterations(20),
    iterations(1),
    alpha(1),
    beta(),
    partition(),
    if_split_phase(0) { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("split", if_split_phase);

    problem_size_baseline.m() = problem_size.m();
    problem_size_baseline.n() = problem_size.n();
    problem_size_baseline.k() = problem_size.k();

    // // cmd.get_cmd_line_argument("partition", partition);
    // partition = problem_size.m() / 128;
    // // add checksum size
    // if(if_split_phase == 1 || if_split_phase == 0){
    //   problem_size.m() += partition * 1;
    // }

    // cmd.get_cmd_line_argument("partition", partition);
    partition = problem_size.n() / 128;
    // add checksum size
    if(if_split_phase == 1 || if_split_phase == 0){
      problem_size.n() += partition * 1;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --partition=<int>           Number of partition of the matrix.\n\n"
      << "  --split=<int>               0-no split, 1-split, 2-baseline.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
// float ok, half ok
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
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

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_baseline = cutlass::gemm::device::GemmBaseline<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_Sign = cutlass::gemm::device::GemmSign<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

template <typename Element>
void encode_col_checksum(Element *A, int k, int n, int partition){
  int m = 1;
  // init checksum vector
  Element *chk_vector;
  chk_vector = (Element*)malloc(sizeof(Element)* k * 1);
  for(int c = 0; c < k; c++){
    chk_vector[c] = (Element)1;
    // chk_vector[c + k] = (float)(c+1);
  }
  // encode chksum - row major
  for(int p = 0; p < partition; p++){
    for(int r = 0; r < 1; r++){
      for(int c = 0; c < n; c++){
          Element sum = (Element)0.0;
          for(int i = 0; i < (k/partition); i++){
              Element a = chk_vector[r * k + i];
              Element b = *(A + (c + (i+(k/partition)*p) * n));
              sum += (a * b);
          }
          // printf("%f, ", sum);
          int idx = (k * n) + (r * n + c) + p * (1 * n);
          *(A + idx) = sum;
      }
      // printf("\n");
    }
  }
  free(chk_vector);
}

template <typename Element>
void encode_row_checksum(Element *A, int m, int k, int partition, int lda){
  int n = 1;
  int k_per_partion = k / partition;

  // init checksum vector
  float *chk_vector;
  chk_vector = (float*)malloc(sizeof(float)* k_per_partion * 1);
  for(int r = 0; r < k_per_partion; r++){
    chk_vector[r] = (float)1.f;
    // chk_vector[c + k] = (float)(c+1);
  }
  // encode row chksum - column major
  for(int p = 0; p < partition; p++){
    for(int c = 0; c < n; c++){
      for(int r = 0; r < m; r++){
        float sum = (float)0.0f;
        for(int i = 0; i < k_per_partion; i++){
          float a = static_cast<float>(A[r + (i + k_per_partion * p) * lda]);
          float b = chk_vector[i];
          sum += a * b;
        }
        // int idx = (k + p) + (c * lda) + (b_idx * stride);
        int idx = lda * (k + p) + r;
        A[idx] = static_cast<Element>(sum);
      }
    }
  }
  free(chk_vector);
}


int run(Options &options) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;
  // cutlass::gemm::GemmCoord problem_size2({1,1,1});

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions

  int kRange = 8;
  float DIV = 1;

  if(options.if_split_phase==1 || options.if_split_phase==0){
    for(int c = 0; c < (problem_size.n() - 1 * options.partition); c++){
      for(int r = 0; r < problem_size.k(); r++){
        int idx = c * problem_size.k() + r;
        *(tensor_b.host_data()+idx) = (ElementInputB)(static_cast<float>(idx % kRange) / DIV);
        // *(tensor_b.host_data()+idx) = (ElementInputB)1;
      }
    }
    encode_row_checksum<ElementInputB>(tensor_b.host_data(), problem_size.k(), (problem_size.n() - 1 * options.partition), options.partition,  problem_size.k());

    for(int c = 0; c < (problem_size.n() - 1 * options.partition); c++){
      for(int r = 0; r < problem_size.m(); r++){
        int idx = c * problem_size.m() + r;
        // *(tensor_c.host_data()+idx) = (ElementOutput)rand()/RAND_MAX;
        *(tensor_c.host_data()+idx) = (ElementOutput)0;
      }
    }
    encode_row_checksum<ElementOutput>(tensor_c.host_data(), problem_size.m(), (problem_size.n() - 1 * options.partition), options.partition, problem_size.m());
  }
  else{
    for(int c = 0; c < problem_size.n(); c++){
      for(int r = 0; r < problem_size.k(); r++){
        int idx = c * problem_size.k() + r;
        *(tensor_b.host_data()+idx) = (ElementInputB)(static_cast<float>(idx % kRange) / DIV);
        // *(tensor_b.host_data()+idx) = (ElementInputB)1;
      }
    }
    cutlass::reference::host::TensorFill(tensor_c.host_view()); 
  }
  for(int idx = 0; idx < (problem_size.m()*problem_size.k()); idx++){
    // *(tensor_a.host_data()+idx) = (ElementInputA)1;
    *(tensor_a.host_data()+idx) = (ElementInputA)(static_cast<float>(idx % kRange) / DIV);
  }
  
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
  
  printf("finish init matrix\n");

  // printf("A: %f\n", *(tensor_a.host_data()+1));

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Result structure
  Result result;

  // Instantiate CUTLASS kernel depending on templates
  Gemm_baseline gemm_op_baseline;
  // if(options.eval_mode == 0){
    // baseline init
    cutlass::gemm::GemmCoord problem_size_baseline = options.problem_size_baseline;
    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> baseline_a(
        problem_size_baseline.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> baseline_b(
        problem_size_baseline.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> baseline_c(
        problem_size_baseline.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> baseline_d(
        problem_size_baseline.mn());  // <- Create matrix D with dimensions M x N used to store output from
                            // CUTLASS kernel
    
    for(int idx = 0; idx < (problem_size_baseline.m()*problem_size_baseline.k()); idx++){
      *(baseline_a.host_data()+idx) = (ElementInputA)(static_cast<float>(idx % kRange) / DIV);
    }
    for(int c = 0; c < problem_size_baseline.n(); c++){
      for(int r = 0; r < problem_size_baseline.k(); r++){
        int idx = c * problem_size_baseline.k() + r;
        *(baseline_b.host_data()+idx) = (ElementInputB)(static_cast<float>(idx % kRange) / DIV);
      }
    }
    cutlass::reference::host::TensorFill(baseline_c.host_view()); 
    cutlass::reference::host::TensorFill(baseline_d.host_view());  // <- fill matrix D on host with zeros

    printf("baseline: finish init matrix\n");

    // Copy data from host to GPU
    baseline_a.sync_device();
    baseline_b.sync_device();
    baseline_c.sync_device();
    baseline_d.sync_device();

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm_baseline::Arguments arguments_baseline{problem_size_baseline,  // <- problem size of matrix multiplication
                                      baseline_a.device_ref(),  // <- reference to matrix A on device
                                      baseline_b.device_ref(),  // <- reference to matrix B on device
                                      baseline_c.device_ref(),  // <- reference to matrix C on device
                                      baseline_d.device_ref(),  // <- reference to matrix D on device
                                      {alpha, beta},          // <- tuple of alpha and beta
                                      split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size_baseline = Gemm_baseline::get_workspace_size(arguments_baseline);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace_baseline(workspace_size_baseline);

    // Check the problem size is supported or not 
    status = gemm_op_baseline.can_implement(arguments_baseline);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op_baseline.initialize(arguments_baseline, workspace_baseline.get());
    CUTLASS_CHECK(status);
  // }

  typename Gemm_Sign::Arguments arguments_sign{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size_sign = Gemm_Sign::get_workspace_size(arguments_sign);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_sign(workspace_size_sign);

  // Instantiate CUTLASS kernel depending on templates
  Gemm_Sign gemm_op_sign;

  // Check the problem size is supported or not 
  status = gemm_op_sign.can_implement(arguments_sign);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op_sign.initialize(arguments_sign, workspace_sign.get());
  CUTLASS_CHECK(status);

  //
  // Construct events
  //

  // cudaEvent_t events[2];

  // for (auto & event : events) {
  //   result.error = cudaEventCreate(&event);
  //   if (result.error != cudaSuccess) {
  //     std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
  //     return -1;
  //   }
  // }

  // // Record an event at the start of a series of GEMMs
  // result.error = cudaEventRecord(events[0]);
  // if (result.error != cudaSuccess) {
  //   std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
  //   return -1;
  // }

  //
  // Run profiling loop
  //

  float t_compute = 0.f; 
  float runtime_Grp, runtime_Glo, runtime_Q, runtime, runtime_D, runtime_A, runtime_baseline;
  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op(options.if_split_phase, options.partition, 2, &t_compute);
    runtime_A = (t_compute);

    cudaDeviceSynchronize();
    status = gemm_op(options.if_split_phase, options.partition, 0, &t_compute);
    runtime = (t_compute);

    cudaDeviceSynchronize();
    status = gemm_op(options.if_split_phase, options.partition, 1, &t_compute);
    runtime_D = (t_compute);

    // Launch baseline CUTLASS kernel
    cudaDeviceSynchronize();
    status = gemm_op_baseline(true, &t_compute);
    runtime_baseline = (t_compute);

    cudaDeviceSynchronize();
    status = gemm_op_sign(0, &t_compute);
    runtime_Glo = (t_compute);

    cudaDeviceSynchronize();
    status = gemm_op_sign(1, &t_compute);
    runtime_Grp = t_compute;

    cudaDeviceSynchronize();
    status = gemm_op_sign(2, &t_compute);
    runtime_Q = t_compute;

    float overhead = ((runtime_Glo - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SignTCC-Glob runtime: %.2f ms, SignTCC-Glob overhead: %.2f%\n", runtime_baseline, runtime_Glo, overhead);
    
    overhead = ((runtime_Grp - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SignTCC-Grp runtime: %.2f ms, SignTCC-Grp overhead: %.2f%\n", runtime_baseline, runtime_Grp, overhead);

    overhead = ((runtime_Q - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SignTCC-Q runtime: %.2f ms, SignTCC-Q overhead: %.2f%\n", runtime_baseline, runtime_Q, overhead);

    overhead = ((runtime - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SchTCC runtime: %.2f ms, SchTCC overhead: %.2f%\n", runtime_baseline, runtime, overhead);
    
    overhead = ((runtime_D - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SchTCC-D runtime: %.2f ms, SchTCC-D overhead: %.2f%\n", runtime_baseline, runtime_D, overhead);
    
    overhead = ((runtime_A - runtime_baseline) / runtime_baseline) * 100;
    printf("baseline runtime: %.2f ms, SchTCC-A runtime: %.2f ms, SchTCC-A overhead: %.2f%\n", runtime_baseline, runtime_A, overhead);
  }

  return 1;

  //
  // Stop profiling loop
  //

  // // Record an event when the GEMMs are complete
  // result.error = cudaEventRecord(events[1]);
  // if (result.error != cudaSuccess) {
  //   std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
  //   return -1;
  // }

  // // Wait for work on the device to complete.
  // result.error = cudaEventSynchronize(events[1]);
  // if (result.error != cudaSuccess) {
  //   std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
  //   return -1;
  // }

  // // Measure elapsed runtime
  // float runtime_ms = 0;
  // result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  // if (result.error != cudaSuccess) {
  //   std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
  //   return -1;
  // }

  // // Compute average runtime and GFLOPs.
  // result.runtime_ms = double(runtime_ms) / double(options.iterations);
  // result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // // Cleanup
  // for (auto event : events) {
  //   (void)cudaEventDestroy(event);
  // }

  // // Create instantiation for device reference gemm kernel
  // cutlass::reference::device::Gemm<ElementInputA,
  //                                  LayoutInputA,
  //                                  ElementInputB,
  //                                  LayoutInputB,
  //                                  ElementOutput,
  //                                  LayoutOutput,
  //                                  ElementComputeEpilogue,
  //                                  ElementComputeEpilogue>
  //     gemm_device;

  // // Launch device reference gemm kernel
  // gemm_device(problem_size,
  //             alpha,
  //             tensor_a.device_ref(),
  //             tensor_b.device_ref(),
  //             beta,
  //             tensor_c.device_ref(),
  //             tensor_ref_d.device_ref());

  // // Wait for kernels to finish
  // cudaDeviceSynchronize();

  // // Copy output data from CUTLASS and reference kernel to host for comparison
  // tensor_d.sync_host();
  // tensor_ref_d.sync_host();

  // // Check if output from CUTLASS kernel and reference kernel are equal or not
  // bool passed = cutlass::reference::host::TensorEquals(
  //   tensor_d.host_view(),
  //   tensor_ref_d.host_view());

  // if (passed) {
  //   std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
  //   std::cout << " GFLOPs: " << result.gflops << std::endl;
  // }

  // std::cout << (passed ? "Passed" : "Failed") << std::endl;

  // return (passed ? 0  : -1);
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d TF32 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}