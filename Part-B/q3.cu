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

void vectorAddUnifiedMemory(int K, int blockSize, int numBlocks) {
    int N = K * 1000000;
    size_t size = N * sizeof(float);

    // Allocate Unified Memory accessible by CPU and GPU
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize arrays A and B
    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Measure the execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    vectorAddKernel<<<numBlocks, blockSize>>>(A, B, C, N);

    // Synchronize to wait for kernel to finish
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for K=" << K << " million elements with "
              << numBlocks << " blocks and " << blockSize << " threads per block: "
              << elapsed.count() << " seconds" << std::endl;

    // Free Unified Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    int Ks[] = {1, 5, 10, 50, 100};

    for (int K : Ks) {
        std::cout << "\n--- Profiling for K = " << K << " million elements ---" << std::endl;

        // Scenario 1: One block with 1 thread
        std::cout << "Scenario 1: One block with 1 thread" << std::endl;
        vectorAddUnifiedMemory(K, 1, 1);

        // Scenario 2: One block with 256 threads
        std::cout << "Scenario 2: One block with 256 threads" << std::endl;
        vectorAddUnifiedMemory(K, 256, 1);

        // Scenario 3: Multiple blocks with 256 threads per block
        int N = K * 1000000;
        int numBlocks = (N + 255) / 256; // Calculate number of blocks needed
        std::cout << "Scenario 3: Multiple blocks with 256 threads per block" << std::endl;
        vectorAddUnifiedMemory(K, 256, numBlocks);
    }

    return 0;
}