#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vectorAddCUDA(int K, int blockSize, int numBlocks) {
    int N = K * 1000000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel and time it
    auto start = std::chrono::high_resolution_clock::now();

    vectorAddKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for K=" << K << " million elements with "
              << numBlocks << " blocks and " << blockSize << " threads per block: "
              << elapsed.count() << " seconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int Ks[] = {1, 5, 10, 50, 100};

    for (int K : Ks) {
        std::cout << "\n--- Profiling for K = " << K << " million elements ---" << std::endl;

        // Scenario 1: One block with 1 thread
        std::cout << "Scenario 1: One block with 1 thread" << std::endl;
        vectorAddCUDA(K, 1, 1);

        // Scenario 2: One block with 256 threads
        std::cout << "Scenario 2: One block with 256 threads" << std::endl;
        vectorAddCUDA(K, 256, 1);

        // Scenario 3: Multiple blocks with 256 threads per block
        int N = K * 1000000;
        int numBlocks = (N + 255) / 256; // Calculate the number of blocks needed
        std::cout << "Scenario 3: Multiple blocks with 256 threads per block" << std::endl;
        vectorAddCUDA(K, 256, numBlocks);
    }

    return 0;
}