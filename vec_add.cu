#include<iostream>
#include<cuda_runtime.h>

__global__ void vector_addition(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaError_t err;

    err = cudaMalloc((void**)&d_A, size);

    if(err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for A" << std::endl;
        return -1;
    }

    err = cudaMalloc((void**)&d_B, size);
    if(err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B" << std::endl;
        cudaFree(d_A);
        return -1;
    }

    err = cudaMalloc((void**)&d_C, size);
    if(err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for C" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
     // blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_addition<<<N + 255 / 256, 256>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if(err != cudaSuccess) 
        std::cerr << "Kernel launch failed: " 
        << cudaGetErrorString(err) << std::endl;
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if(err != cudaSuccess) 
        std::cerr << "Failed to copy result back to host: " 
        << cudaGetErrorString(err) << std::endl;

    for(int i = 0; i < std::min(N, 10); ++i) {
        std::cout<< "h_C[" << i << "] = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

