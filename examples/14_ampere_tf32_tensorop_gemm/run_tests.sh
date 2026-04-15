#!/bin/bash

echo "------------run no abft------------"
for i in $(seq 1 5);
do
    ./cutlass_no_abft.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run paris(2 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_group_test_2.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run paris(5 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_group_test_5.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run paris(10 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_group_test_10.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run step------------"
for i in $(seq 1 5);
do
    ./cutlass_test.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run split pairs(2 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_split_group_2.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run split pairs(5 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_split_group_5.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run split pairs(10 in group)------------"
for i in $(seq 1 5);
do
    ./cutlass_split_group_10.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done

echo "------------run split step------------"
for i in $(seq 1 5);
do
    ./cutlass_split.exe --m=1280 --n=1280 --k=1280 --iterations=10000
done


# ./out.exe --m=128 --n=128 --k=128 --split=0 --iterations=1
# ./out.exe --m=256 --n=256 --k=256 --split=0 --iterations=1
# ./out.exe --m=640 --n=640 --k=640 --split=0 --iterations=1
# ./out.exe --m=1280 --n=1280 --k=1280 --split=0 --iterations=1
# ./out.exe --m=1408 --n=1408 --k=1408 --split=0 --iterations=1
# ./out.exe --m=1536 --n=1536 --k=1536 --split=0 --iterations=1
# ./out.exe --m=1920 --n=1920 --k=1920 --split=0 --iterations=1
# ./out.exe --m=2048 --n=2048 --k=2048 --split=0 --iterations=1
# ./out.exe --m=2560 --n=2560 --k=2560 --split=0 --iterations=1
# ./out.exe --m=2816 --n=2816 --k=2816 --split=0 --iterations=1

# ./out.exe --m=128 --n=1024 --k=1024 --split=0 --iterations=1

# ./out.exe --m=15360 --n=28160 --k=28160 --split=2 --iterations=1

# ./out_c.exe --m=4096 --n=8192 --k=4096 --split=0 --iterations=1

# conda install conda-forge::numactl-devel-conda-x86_64

# ./boutT.exe --m=1024 --n=1024 --k=128 --batch=256 --split=1 --iterations=1 --validate=1

conda create -n pytorch_cutlass_baseline \
  -c conda-forge \
  python=3.10 \
  gcc=12 \
  gxx=12

export CC=gcc
export CXX=g++
export CFLAGS="-std=c11"
export CXXFLAGS="-std=c++17"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

conda install conda-forge::numactl-devel-conda-x86_64

pip install --upgrade git+https://github.com/huggingface/transformers.git

# input size: torch.Size([8, 1024, 2048]); query size: torch.Size([8, 32, 1024, 64]); key size: torch.Size([8, 8, 1024, 64]); value size: torch.Size([8, 8, 1024, 64])

import torch
import torch.nn as nn
linear_layer = nn.Linear(in_features=4096, out_features=4096, bias=False).to(torch.bfloat16).to('cuda')
input_tensor = torch.randn(8192, 4096).to(torch.bfloat16).to('cuda')
output_tensor = linear_layer(input_tensor)

linear_layer_cpu = linear_layer.to('cpu')
input_tensor_cpu = input_tensor.to('cpu')
linear_layer_cpu(input_tensor_cpu)

# OK
a = torch.randn(8,32,1024,128).to(torch.bfloat16).to('cuda')
b = torch.randn(8,32,128,1024).to(torch.bfloat16).to('cuda')
c = torch.matmul(a,b)
c_cpu = torch.matmul(a.to('cpu'),b.to('cpu'))

# OK
a = torch.randn(256,128,1024).to('cuda')
b = torch.randn(256,1024,1024).to('cuda')
c = torch.matmul(a,b)
c_cpu = torch.matmul(a.to('cpu'),b.to('cpu'))

# OK
a = torch.randn(8,32,1024,1024).to('cuda')
b = torch.randn(8,32,1024,128).to('cuda')
c = torch.matmul(a,b)
c_cpu = torch.matmul(a.to('cpu'),b.to('cpu'))

# OK
a = torch.randn(8,32,1024,128).to(torch.bfloat16).to('cuda')
b = torch.randn(8,32,1024,128).to(torch.bfloat16).to('cuda')
c = torch.matmul(a,b.transpose(-2,-1))
c_cpu = torch.matmul(a.to('cpu'),b.transpose(-2,-1).to('cpu'))

a = torch.randn(2,1024,128).to('cuda')
b = torch.randn(2,1024,128).to('cuda')
c = torch.matmul(a,b.transpose(-2,-1))
c_cpu = torch.matmul(a.to('cpu'),b.transpose(-2,-1).to('cpu'))

print(input_tensor)
print(linear_layer.weight)
print(linear_layer.bias)


a = torch.randn(8,4,8).to('cuda')
b = torch.randn(8,8,6).to('cuda')
c = torch.matmul(a,b)