// Linear regression implemented from scratch in CUDA but without shared memory
//  calculate_coefficients: calculatePartialSums + calculatePartialCoefficients
//  make_predictions: makePredictions
//  calculate_mse: calculatePartialMSE

// reference (python) -- https://www.geeksforgeeks.org/linear-regression-python-implementation/

#ifndef __LINEAR_REGRESSION_NO_SHARED_MEMORY_CUDA__
#define __LINEAR_REGRESSION_NO_SHARED_MEMORY_CUDA__

#include <cuda_runtime.h>
#include "CUDA_helpers.cu"

// Kernel to calculate linear regression coefficients
// Calculates numerator and denominator which are then used to calculate slope and intercept
__global__ void calculatePartialCoefficients(
    const float* x,
    const float* y,
    const float x_mean,
    const float y_mean,
    float* num,
    float* dem,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x_diff = x[idx] - x_mean;
        float y_diff = y[idx] - y_mean;

        atomicAdd(num, x_diff * y_diff);
        atomicAdd(dem, x_diff * x_diff);
    }
}

// Kernel to calculate partial sums of x and y
// Calculates sum which is then used to calculate mean
__global__ void calculatePartialSums(
    const float* x,
    const float* y,
    float* x_partial_sum,
    float* y_partial_sum,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(x_partial_sum, x[idx]);
        atomicAdd(y_partial_sum, y[idx]);
    }
}

// Kernel to calculate the Mean Square Error (MSE)
// Calculates squared error which is then used to calculate mean squared error
__global__ void calculatePartialMSE(
    const float* y,
    const float* predictions,
    float* mse,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate squared difference directly in global memory
    if (idx < n) {
        float diff = y[idx] - predictions[idx];
        atomicAdd(mse, diff * diff);
    }
}

// Uses mx+b to make predictions for dataset x
__global__ void makePredictions(
    const float* x,
    float* predictions,
    const float slope,
    const float intercept,
    const int n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        predictions[idx] = slope * x[idx] + intercept;
    }
}

class LinearRegression_CUDA_NoSharedMemory {
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
            this->train_size = train_size;
            this->test_size = test_size;
            this->trained = false;
            this->made_predictions = false;
            this->calculated_mse = false;

            // Variables on heap
            this->h_x = new float[n];
            this->h_y = new float[n];
            memcpy(h_x, x, n * sizeof(float));
            memcpy(h_y, y, n * sizeof(float));
            this->h_predictions = new float[test_size];

            // Block and Grid size
            this->block_size = 256;
            this->grid_size_train = (train_size + block_size - 1) / block_size;
            this->grid_size_test = (train_size + block_size - 1) / block_size;

            // Variables on GPU
            CUDA_CHECK( cudaMalloc(&d_x, n * sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&d_y, n * sizeof(float)) );
            CUDA_CHECK( cudaMemcpy(d_x, this->h_x, n * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_y, this->h_y, n * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMalloc(&d_predictions, test_size * sizeof(float)) );
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
            calculatePartialSums<<<grid_size_train, block_size>>>(d_x, d_y, d_x_mean, d_y_mean, train_size);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
            // Calculates means on host from sum on device
            float x_mean, y_mean;
            CUDA_CHECK( cudaMemcpy(&x_mean, d_x_mean, sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_CHECK( cudaMemcpy(&y_mean, d_y_mean, sizeof(float), cudaMemcpyDeviceToHost) );
            x_mean = x_mean / train_size;
            y_mean = y_mean / train_size;

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
            calculatePartialCoefficients<<<grid_size_train, block_size>>>(d_x, d_y, x_mean, y_mean, d_num, d_den, train_size);
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
            float *d_x_test = d_x+train_size; // Pointer to where test data starts
            makePredictions<<<grid_size_test, block_size>>>(d_x_test, d_predictions, slope, intercept, test_size);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // Copy data from GPU and print
            CUDA_CHECK( cudaMemcpy(h_predictions, d_predictions, test_size*sizeof(float), cudaMemcpyDeviceToHost) );
            printf("Predictions (first 10)\n");
            for (int i = 0; i < 10; i++) {
                printf ("%f : %f\n", h_x[i+train_size], h_predictions[i]);
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
            float *d_y_test = d_y + train_size; // Pointer to where test data starts
            calculatePartialMSE<<<grid_size_test, block_size>>>(d_y_test, d_predictions, d_mse, test_size);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
            // Final MSE calculation on the host
            CUDA_CHECK( cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost) );
            mse = mse / test_size;
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

        bool trained; // If coefficients have been calculated
        float slope, intercept; // The model weights

        bool made_predictions; // If predictions have been made
        float *h_predictions; // Predictions on host
        float *d_predictions; // Predictions on device

        bool calculated_mse; // If MSE has been calculated
        float mse; // mean squared error

        // Block and Grid sizes (Removed shared memory size)
        int block_size;
        int grid_size_train; // Grid size when using training dataset
        int grid_size_test; // Grid size when using testing dataset
};
#endif // __LINEAR_REGRESSION_NO_SHARED_MEMORY_CUDA__
