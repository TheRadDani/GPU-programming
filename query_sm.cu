#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 = device ID
    std::cout << "GPU name: " << prop.name << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    return 0;
}


__device__ unsigned get_smid() {
    unsigned smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__global__ void kernel() {
    unsigned smid = get_smid();
    printf("Thread %d running on SM %u\n", threadIdx.x, smid);
}
