// %%writefile LinearRegression_CUDA.cu

// Linear regression implemented from scratch in CUDA
// reference (python) -- https://www.geeksforgeeks.org/linear-regression-python-implementation/

#ifndef __LINEAR_REGRESSION_CUDA__
#define __LINEAR_REGRESSION_CUDA__

#include <cuda_runtime.h>
#include "CUDA_helpers.cu"

// Kernel to calculate coefficients
// Calculates numerator and denominator which are then used to calculate slope and intercept
static __global__ void calculatePartialCoefficients(const float* x, const float* y, const float x_mean, const float y_mean, float* num, float* dem, const int n) {
    extern __shared__ float cc_shared_mem[];
    float* num_shared = cc_shared_mem;
    float* dem_shared = cc_shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    num_shared[tid] = 0.0f;
    dem_shared[tid] = 0.0f;

    // Calculate partial results
    if (idx < n) {
        float x_diff = x[idx] - x_mean;
        float y_diff = y[idx] - y_mean;

        num_shared[tid] = x_diff * y_diff;
        dem_shared[tid] = x_diff * x_diff;
    }
    __syncthreads();

    // Block-wise reduction to sum partial results
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            num_shared[tid] += num_shared[tid + stride];
            dem_shared[tid] += dem_shared[tid + stride];
        }
        __syncthreads();
    }

    // Atomic operations to accumulate the block's result to global memory
    if (tid == 0) {
        atomicAdd(num, num_shared[0]);
        atomicAdd(dem, dem_shared[0]);
    }
}

// Kernel to calculate partial sums of x and y
// Calculates sum which is then used to calculate mean
static __global__ void calculatePartialSums(const float* x, const float* y, float* x_partial_sum, float* y_partial_sum, const int n) {
    extern __shared__ float shared_mem[];
    float* x_shared = shared_mem;
    float* y_shared = shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    x_shared[tid] = (idx < n) ? x[idx] : 0;
    y_shared[tid] = (idx < n) ? y[idx] : 0;
    __syncthreads();

    // Perform block-wise reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            x_shared[tid] += x_shared[tid + stride];
            y_shared[tid] += y_shared[tid + stride];
        }
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0) {
        atomicAdd(x_partial_sum, x_shared[0]);
        atomicAdd(y_partial_sum, y_shared[0]);
    }
}

// Kernel to calculate the Mean Square Error (MSE)
// Calculates squared error which is then used to calculate mean squared error
static __global__ void calculatePartialMSE(const float* y, const float* predictions, float* mse, const int n) {
    extern __shared__ float mse_shared_mem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    mse_shared_mem[tid] = 0.0f;
    __syncthreads();

    // Calculate squared difference and store in shared memory
    if (i < n) {
        float diff = y[i] - predictions[i];
        mse_shared_mem[tid] = diff * diff;
    }
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            mse_shared_mem[tid] += mse_shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Atomic operations to accumulate the block's result to global memory
    if (tid == 0) {
        atomicAdd(mse, mse_shared_mem[0]);
    }
}


// Uses mx+b to make predictions for dataset x
static __global__ void makePredictions(const float* x, float* predictions, const float slope, const float intercept, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        predictions[idx] = slope * x[idx] + intercept;
    }
}


class LinearRegression_CUDA {
    public:
        LinearRegression_CUDA (
            const int n,
            const int train_size,
            const int test_size,
            float* x,
            float* y
        ) {
            // Variables on stack
            this->n = n;
            this->trained = false;
            this->made_predictions = false;
            this->calculated_mse = false;

            // Variables on heap
            this->h_x = new float[n];
            this->h_y = new float[n];
            memcpy(h_x, x, n * sizeof(float));
            memcpy(h_y, y, n * sizeof(float));
            this->h_predictions = new float[n];

            // Block, Grid and Shared Memory size
            this->block_size = 256;
            this->grid_size = (n + block_size - 1) / block_size;
            this->shared_mem_size = block_size * 2 * sizeof(int);

            // Variables on GPU
            CUDA_CHECK( cudaMalloc(&d_x, n * sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&d_y, n * sizeof(float)) );
            CUDA_CHECK( cudaMemcpy(d_x, this->h_x, n * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_y, this->h_y, n * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMalloc(&d_predictions, n*sizeof(float)) );
        }

        ~LinearRegression_CUDA() {
            delete[] h_x;
            delete[] h_y;
            delete[] h_predictions;
            CUDA_CHECK( cudaFree(d_x) );
            CUDA_CHECK( cudaFree(d_y) );
            CUDA_CHECK( cudaFree(d_predictions) );
        }

        bool calculate_coefficients() {
            //// Calculate means
            float *d_x_mean, *d_y_mean;
            // GPU memory
            CUDA_CHECK( cudaMalloc((void**)&d_x_mean, sizeof(float)) );
            CUDA_CHECK( cudaMalloc((void**)&d_y_mean, sizeof(float)) );
            CUDA_CHECK( cudaMemset(d_x_mean, 0, sizeof(float)) );
            CUDA_CHECK( cudaMemset(d_y_mean, 0, sizeof(float)) );

            // Sums Kernel
            calculatePartialSums<<<grid_size, block_size, shared_mem_size>>>(d_x, d_y, d_x_mean, d_y_mean, n);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // Calculates means on host from sum on device
            float x_mean, y_mean;
            CUDA_CHECK( cudaMemcpy(&x_mean, d_x_mean, sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_CHECK( cudaMemcpy(&y_mean, d_y_mean, sizeof(float), cudaMemcpyDeviceToHost) );
            x_mean = x_mean / n;
            y_mean = y_mean / n;

            // GPU cleanup
            CUDA_CHECK( cudaFree(d_x_mean) );
            CUDA_CHECK( cudaFree(d_y_mean) );

            //// Calculate coefficients
            // GPU memory
            float *d_num;
            float *d_den;
            CUDA_CHECK( cudaMalloc((void**)&d_num, sizeof(float)) );
            CUDA_CHECK( cudaMalloc((void**)&d_den, sizeof(float)) );
            CUDA_CHECK( cudaMemset(d_num, 0.0f, sizeof(float)) );
            CUDA_CHECK( cudaMemset(d_den, 0.0f, sizeof(float)) );

            // Calculates coefficients kernel
            calculatePartialCoefficients<<<grid_size, block_size, shared_mem_size>>>(d_x, d_y, x_mean, y_mean, d_num, d_den, n);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // Calculate slope and intercept on host
            float numerator, denominator;
            CUDA_CHECK( cudaMemcpy(&numerator, d_num, sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_CHECK( cudaMemcpy(&denominator, d_den, sizeof(float), cudaMemcpyDeviceToHost) );
            this->slope = numerator / denominator;
            this->intercept = y_mean - slope * x_mean;

            // Update
            printf ("slope %f  intercept %f\n\n", slope, intercept);
            trained = true;
            return true;
        }


        bool make_predictions() {
            if (!trained) {
                printf ("error: not trained\n");
                return false;
            }
            if (made_predictions) {
                printf ("error: predictions already made\n");
                return false;
            }

            // Run predictions kernel
            makePredictions<<<grid_size, block_size>>>(d_x, d_predictions, slope, intercept, n);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // Copy data from GPU and print
            CUDA_CHECK( cudaMemcpy(h_predictions, d_predictions, n*sizeof(float), cudaMemcpyDeviceToHost) );
            printf("Predictions (first 10)\n");
            for (int i = 0; i < 10; i++) {
                printf ("%f : %f\n", h_x[i], h_predictions[i]);
            }
            printf ("\n");
        
            made_predictions = true;
            return true;
        }

        bool calculate_mse() {
            if (calculated_mse) {
                printf ("error: MSE already calculated");
                return false;
            }
            if (!made_predictions) {
                printf ("error: predictions not made\n");
                return false;
            }

            // GPU Memory
            float* d_mse;
            CUDA_CHECK( cudaMalloc((void**)&d_mse, sizeof(float)) );

            // Run the kernel to calculate SE
            calculatePartialMSE<<<grid_size, block_size, shared_mem_size>>>(d_y, d_predictions, d_mse, n);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // Final MSE calculation on the host
            CUDA_CHECK( cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost) );
            mse = mse / n;
            printf ("MSE: %f\n\n", mse);

            // Update
            calculated_mse = true;
            return true;
        }

    private:
        int n; // Dataset size
        int train_size, test_size; // Number of elements in train and test set
        float *h_x, *h_y; // Independent and dependent values on host
        float *d_x, *d_y; // Independent and dependent values on device

        cublasHandle_t handle; // cuBLAS handle

        bool trained; // If coefficients have been calculated
        float slope, intercept; // The model weights

        bool made_predictions; // If predictions have been made
        float *h_predictions; // Predictions on host
        float *d_predictions; // Predictions on device

        bool calculated_mse; // If MSE has been calculated
        float mse; // mean squared error

        // Block, Grid and Shared Memory size
        int block_size;
        int grid_size;
        int shared_mem_size;
};
#endif // __LINEAR_REGRESSION_CUDA__
