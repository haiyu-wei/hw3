#include <iostream>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    // Check if K is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K (in millions)>" << std::endl;
        return 1;
    }

    // Parse K from command line
    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "K must be a positive integer." << std::endl;
        return 1;
    }

    // Calculate the size of arrays
    size_t N = K * 1000000; // Convert K million to actual number of elements

    // Allocate memory for arrays A, B, and C
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Memory allocation failed." << std::endl;
        free(A);
        free(B);
        free(C);
        return 1;
    }

    // Initialize arrays A and B
    for (size_t i = 0; i < N; ++i) {
        A[i] = 1.0f; // Example value
        B[i] = 2.0f; // Example value
    }

    // Measure time for vector addition
    auto start = std::chrono::high_resolution_clock::now();

    // Vector addition
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for K=" << K << " million elements: " << elapsed.count() << " seconds" << std::endl;

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}