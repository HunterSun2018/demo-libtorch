#include <iostream>
#include <sstream>
#include <chrono>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
using namespace std;


void print_CUDA_cuDNN_info();
int run_mnist(bool usingGPU = true);

int main(int argc, char* argv[])
{
    bool usingGPU = true;

    if (argc >= 2)
        istringstream(argv[1]) >> std::boolalpha >> usingGPU;

    cout << "cuda is " << torch::cuda::is_available() << endl;

    print_CUDA_cuDNN_info();

    auto start = std::chrono::steady_clock::now();

    run_mnist(usingGPU);

    auto end = std::chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

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