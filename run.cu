#include<cuda.h>
#include<iostream>

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    cuInit(0);
    cuDeviceGet(&device, 0);
    std::cout << "Device initialized\n";
    std::cout << "Device ID: " << device << "\n";
    cuCtxCreate(&context, 0, device);
    cuModuleLoad(&module, "kernel_addtion.cubin");

    cuModuleGetFunction(&kernel, 
                        module,
                        "_Z15vector_additionPfS_S_i");

    cuLaunchKernel(
        kernel,
        1,1,1,   // blocks
        1,1,1,   // threads
        0,
        0,
        nullptr,
        nullptr
    );

    cuCtxSynchronize();

    std::cout << "Kernel executed\n";
}