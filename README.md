This is a test of not using the shared memory for MSE  
calculatePartialMSE time:  0.040992 ms  <-- note runtime is marginally slower
__global__ void calculatePartialMSE(const int* y, const int* predictions, float* mse, int n) {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  

    // Calculate squared difference and store in shared memory  
    if (idx < n) {  
        int diff = y[idx] - predictions[idx];  
        mse[idx] = diff * diff;  
    }  
}  