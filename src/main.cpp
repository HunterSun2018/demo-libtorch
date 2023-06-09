#include <iostream>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
using namespace std;


void print_CUDA_cuDNN_info();

int main(int argc, char* argv[])
{
    cout << "cuda is " << torch::cuda::is_available() << endl;

    print_CUDA_cuDNN_info();

    return 0;
}

void print_CUDA_cuDNN_info()
{
    long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
    cout << "The cuDNN version is " << double(cudnn_version) / 1000 << "\n";
    int runtimeVersion;
    AT_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    cout << "The CUDA runtime version is " << double(runtimeVersion) / 1000 << "\n";
    int version;
    AT_CUDA_CHECK(cudaDriverGetVersion(&version));
    cout << "The driver version is " << double(version) / 1000 << "\n";
}