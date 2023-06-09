# A example for libtorch 2.0

# OS
Ubutnu 22.04

## Required

1. Download libtorch 2.0 from https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip
2. extract the downloaded file to folder ~/libtorch2.0.1 

3. sudo apt instll nvidia-cudnn

## build
cmake . -Bbuild  
cd build && make

## run 
./demo

## output
```
cuda is 1
The cuDNN version is 8.5
The CUDA runtime version is 11.05
The driver version is 11.07
```
