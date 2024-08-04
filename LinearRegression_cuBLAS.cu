#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1024

// Error checking macro for CUDA
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Error checking macro for cuBLAS
#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error at line %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    }

int main() {
    // Dataset
    float h_x[N], h_y[N];
    for (int i = 0; i < N; ++i) {
        h_x[i] = i % 100 + 1;  // Example initialization
        h_y[i] = i % 150 + 1;  // Example initialization
    }


    // cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK( cublasCreate(&handle) );
    const float alpha = 1.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    // GPU memory
    float *d_x, *d_y;
    float mean_x, mean_y;
    CUDA_CHECK( cudaMalloc((void**)&d_x, N * sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&d_y, N * sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_y, h_y, N * sizeof(int), cudaMemcpyHostToDevice) );

    // Compute sums
    CUBLAS_CHECK( cublasSasum(handle, N, d_x, 1, &sum_x) );
    CUBLAS_CHECK( cublasSasum(handle, N, d_y, 1, &sum_y) );
    // Compute means
    mean_x = sum_x / N;
    mean_y = sum_y / N;

    // Print means
    printf("Mean of x array: ", mean_x);
    printf("Mean of y array: ", mean_y);

    // Free / destroy
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
