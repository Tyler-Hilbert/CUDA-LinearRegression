// Implementing linear regression from scratch in CUDA

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to calculate coefficients
__global__ void calculateCoefficients(const int* x, const int* y, const int x_mean, const int y_mean, float* slope, float* bias, const int n) {
    // Shared memory for partial sums
    __shared__ float num[256];
    __shared__ float den[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;

    // Initialize shared memory
    num[local_idx] = 0.0f;
    den[local_idx] = 0.0f;

    // Load data and compute partial sums
    if (idx < n) {
        int x_diff = x[idx] - x_mean;
        int y_diff = y[idx] - y_mean;
        num[local_idx] = static_cast<float>(x_diff) * y_diff;
        den[local_idx] = static_cast<float>(x_diff) * x_diff;
    }

    // Synchronize threads within the block
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            num[local_idx] += num[local_idx + stride];
            den[local_idx] += den[local_idx + stride];
        }
        __syncthreads();
    }

    // Use atomicAdd to update global slope and bias
    if (local_idx == 0) {
        atomicAdd(slope, num[0]);
        atomicAdd(bias, den[0]);
    }

    __syncthreads();

    // Final calculation of coefficients
    if (idx == 0) {
        float final_slope = *slope / *bias;
        float final_bias = y_mean - final_slope * x_mean;
        *slope = final_slope;
        *bias = final_bias;
    }

}


// Kernel to calculate mean of x and y
__global__ void calculateMeans(const int* x, const int* y, int* x_mean, int* y_mean, const int n) {
    // Shared memory for partial sums
    __shared__ int partial_sum_x[256];
    __shared__ int partial_sum_y[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;

    // Initialize shared memory
    partial_sum_x[local_idx] = 0;
    partial_sum_y[local_idx] = 0;

    // Accumulate partial sums in shared memory
    if (idx < n) {
        partial_sum_x[local_idx] = x[idx];
        partial_sum_y[local_idx] = y[idx];
    }

    // Synchronize threads within the block
    __syncthreads();

    // Parallel reduction to sum values in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            partial_sum_x[local_idx] += partial_sum_x[local_idx + stride];
            partial_sum_y[local_idx] += partial_sum_y[local_idx + stride];
        }
        __syncthreads();
    }

    // The first thread in the block updates the global mean
    if (local_idx == 0) {
        atomicAdd(x_mean, partial_sum_x[0] / n);
        atomicAdd(y_mean, partial_sum_y[0] / n);
    }
}

// Kernel to calculate the Mean Square Error (MSE)
__global__ void calculateMSE(const int* y, const int* predictions, float* mse, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int diff = y[i] - predictions[i];
        atomicAdd(mse, diff * diff);
    }
}


// Uses mx+b to make predictions for dataset x
__global__ void makePredictions(const int* x, int* predictions, const int slope, const int bias, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        predictions[idx] = slope * x[idx] + bias;
    }
}

int main() {
    // The Dataset
    const int N = 10;
    int h_x[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int *d_x;
    int h_y[N] = {55, 58, 61, 66, 70, 72, 75, 78, 82, 85};
    int *d_y;

    // Block & Grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;


    // Calculate means
    // GPU memory
    int *d_x_mean, *d_y_mean;
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("1CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void**)&d_y, N * sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("2CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void**)&d_x_mean, sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("3CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void**)&d_y_mean, sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("4CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("5CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_y, h_y, N * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("6CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemset(d_x_mean, 0, sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("7CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemset(d_y_mean, 0, sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("8CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Calculate means kernel
    calculateMeans<<<gridSize, blockSize>>>(d_x, d_y, d_x_mean, d_y_mean, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("9CUDA error: %s\n", cudaGetErrorString(err));
    }
    int x_mean, y_mean;
    cudaMemcpy(&x_mean, d_x_mean, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("10CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(&y_mean, d_y_mean, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("11CUDA error: %s\n", cudaGetErrorString(err));
    }

    // GPU Cleanup
    cudaFree(d_x_mean);
    cudaFree(d_y_mean);



    // Calculate coefficients
    // GPU memory
    float *d_slope, *d_bias;
    float slope, bias;
    cudaMalloc((void**)&d_slope, sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("12CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void**)&d_bias, sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("13CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Calculates coefficients kernel
    calculateCoefficients<<<gridSize, blockSize>>>(d_x, d_y, x_mean, y_mean, d_slope, d_bias, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("14CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(&slope, d_slope, sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("15CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("16CUDA error: %s\n", cudaGetErrorString(err));
    }
    printf ("slope %f  bias %f\n", slope, bias);

    // GPU Cleanup
    cudaFree(d_slope);
    cudaFree(d_bias);



    // Make Predictions
    // GPU memory
    int *d_predictions, h_predictions[N];
    cudaMalloc((void**)&d_predictions, N * sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("17CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Run predictions kernel
    makePredictions<<<gridSize, blockSize>>>(d_x, d_predictions, slope, bias, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("18CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy data from GPU and print
    cudaMemcpy(h_predictions, d_predictions, N*sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("19CUDA error: %s\n", cudaGetErrorString(err));
    }
    printf("Predictions: ");
    for (int i = 0; i < N; i++) {
      printf ("%i ", h_predictions[i]);
    }
    printf ("\n");

    // GPU Cleanup
    cudaFree(d_x);


    // Calculate MSE
    // GPU Memory
    float* d_mse;
    float mse;

    // Allocate and initialize the MSE variable on the device
    cudaMalloc((void**)&d_mse, sizeof(float));
    if (err != cudaSuccess) {
        printf("20CUDA error (malloc): %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(d_mse, 0.0f, sizeof(float));
    if (err != cudaSuccess) {
        printf("21CUDA error (memset): %s\n", cudaGetErrorString(err));
        return;
    }
    // Run the kernel
    calculateMSE<<<gridSize, blockSize>>>(d_y, d_predictions, d_mse, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("22CUDA error (kernel launch): %s\n", cudaGetErrorString(err));
        return;
    }
    // Copy the result back to host
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("23CUDA error (memcpy): %s\n", cudaGetErrorString(err));
    }
    // Final MSE calculation on the host
    mse = mse / (float)N;
    printf ("MSE: %f\n", mse);

    // Free GPU memory
    cudaFree(d_mse);
    cudaFree(d_predictions);
    cudaFree(d_y);



    return 0;
}