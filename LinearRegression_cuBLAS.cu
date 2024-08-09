// Linear Regression implemented from scratch, but using cuBLAS.
//  calculate_coefficients: 2xcublasSasum + 2xcublasSdot
//  make_predictions: cublasSaxpy + 2xcudaMemcpy
//  calculate_mse: cublasSaxpy + cublasSdot

// reference (python) -- https://www.geeksforgeeks.org/linear-regression-python-implementation/

#ifndef __LINEAR_REGRESSION_CUBLAS__
#define __LINEAR_REGRESSION_CUBLAS__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CUDA_helpers.cu"


class LinearRegression_cuBLAS {
    public:
    LinearRegression_cuBLAS (
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

        // Variables on GPU
        CUDA_CHECK( cudaMalloc(&d_x, n * sizeof(float)) );
        CUDA_CHECK( cudaMalloc(&d_y, n * sizeof(float)) );
        CUDA_CHECK( cudaMemcpy(d_x, this->h_x, n * sizeof(float), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(d_y, this->h_y, n * sizeof(float), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMalloc(&d_predictions, test_size*sizeof(float)) );

        // cuBLAS
        CUBLAS_CHECK( cublasCreate(&handle) );
    }

    ~LinearRegression_cuBLAS() {
        delete[] h_x;
        delete[] h_y;
        delete[] h_predictions;
        CUDA_CHECK( cudaFree(d_x) );
        CUDA_CHECK( cudaFree(d_y) );
        CUDA_CHECK( cudaFree(d_predictions) );
        CUBLAS_CHECK( cublasDestroy(handle) );
    }

    bool calculate_coefficients() {
        if (made_predictions) {
            printf ("error: predictions already made\n");
            return false;
        }

        // Calculate means of x and y using cuBLAS
        float x_sum, y_sum;
        CUBLAS_CHECK( cublasSasum(handle, train_size, d_x, 1, &x_sum) );
        CUBLAS_CHECK( cublasSasum(handle, train_size, d_y, 1, &y_sum) );
        float x_mean = x_sum / train_size;
        float y_mean = y_sum / train_size;

        // Calculate xy and xx dot products using cuBLAS
        float xy_sum = 0, xx_sum = 0;
        CUBLAS_CHECK( cublasSdot(handle, train_size, d_x, 1, d_y, 1, &xy_sum) );
        CUBLAS_CHECK( cublasSdot(handle, train_size, d_x, 1, d_x, 1, &xx_sum) );

        // Calculate slope (b1) and intercept (b0) on host
        float nxy = train_size * x_mean * y_mean;
        float nxx = train_size * x_mean * x_mean;
        this->slope = (xy_sum - nxy) / (xx_sum - nxx);
        this->intercept = y_mean - slope * x_mean;

        // Print slope and intercept
        printf ("Slope: %f\n", this->slope);
        printf ("Intercept: %f\n", this->intercept);

        // Update
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

        // GPU Data
        for (int i = 0; i < test_size; i++) {
            h_predictions[i] = intercept;
        }
        CUDA_CHECK( cudaMemcpy(d_predictions, h_predictions, test_size * sizeof(float), cudaMemcpyHostToDevice) );

        // Compute y = mx + b using cuBLAS
        float *d_x_test = d_x+train_size; // Pointer to where test data starts
        CUBLAS_CHECK( cublasSaxpy(handle, test_size, &slope, d_x_test, 1, d_predictions, 1) );

        // Copy data back to host and print
        CUDA_CHECK( cudaMemcpy(h_predictions, d_predictions, test_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf ("Predictions (first 10)\n");
        for (int i = 0; i < 10; i++) {
            printf ("%f: %f\n", h_x[train_size+i], h_predictions[i]);
        }

        // Update
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

      // Calculate the difference on cuBLAS. Note: This is destructive as it will modify the d_predictions values
      float alpha = -1.0f; // To subtract
      float *d_y_train = d_y + train_size; // Pointer to where test data starts
      // Diff calculation on cuBLAS
      CUBLAS_CHECK( cublasSaxpy(handle, test_size, &alpha, d_y_train, 1, d_predictions, 1));
      // Calculate sum of the squares on cuBLAS
      float squared_error_sum = 0;
      CUBLAS_CHECK( cublasSdot(handle, test_size, d_predictions, 1, d_predictions, 1, &squared_error_sum) );
      // Mean
      mse = squared_error_sum / test_size;

      // Print MSE
      printf("Mean Squared Error: %f\n", mse);

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
        float mse; // Mean squared error
};

#endif // __LINEAR_REGRESSION_CUBLAS__