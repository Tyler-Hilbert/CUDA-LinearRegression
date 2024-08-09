## Linear regression implemented from scratch in CUDA vs cuBLAS  
`LinearRegression_cuBLAS.cu` is linear regression written from scratch but utilizing the cuBLAS library.  
`LinearRegression_CUDA.cu` is linear regression with kernels written completely from scratch.  

## cuBLAS vs CUDA Performance  
Tested on T4 with 800,000 training points and 200,000 testing points using CUDA 12.2.  
![cuBLAS vs CUDA Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Perf/CUDA_vs_cuBLAS_Runtime_by_Task.png)  

### CUDA (Kernel Written from Scratch)  
#### Table 1: CUDA 'cuda_api_sum'  
| Time (%) | Total Time (ns) | Num Calls | Avg (ns)      | Med (ns)    | Min (ns) | Max (ns)     | StdDev (ns)    | Name                   |  
|----------|-----------------|-----------|---------------|-------------|----------|--------------|---------------|------------------------|  
| 50.1     | 126,094,891      | 8         | 15,761,861.4  | 13,477.5    | 2,781    | 125,828,040  | 44,473,478.8  | cudaMalloc              |  
| 48.3     | 121,654,725      | 4         | 30,413,681.3  | 34,383.0    | 19,211   | 121,566,748  | 60,768,712.3  | cudaLaunchKernel        |  
| 1.0      | 2,608,023        | 8         | 326,002.9     | 22,941.0    | 10,456   | 916,812      | 430,826.4     | cudaMemcpy              |  
| 0.3      | 640,070          | 4         | 160,017.5     | 4,877.0     | 2,505    | 627,811      | 311,870.1     | cudaMemset              |  
| 0.2      | 414,971          | 5         | 82,994.2      | 7,205.0     | 4,713    | 213,711      | 106,401.8     | cudaFree                |  
| 0.1      | 286,250          | 4         | 71,562.5      | 85,105.5    | 23,420   | 92,619       | 32,290.6      | cudaDeviceSynchronize   |  
| 0.0      | 988              | 1         | 988.0         | 988.0       | 988      | 988          | 0.0           | cuModuleGetLoadingMode  |  

#### Table 2: CUDA 'cuda_gpu_kern_sum'  
| Time (%) | Total Time (ns) | Instances | Avg (ns)   | Med (ns)   | Min (ns) | Max (ns) | StdDev (ns) | Name                                                                                         |  
|----------|-----------------|-----------|------------|------------|----------|----------|-------------|------------------------------------------------------------------------------------------------|  
| 31.8     | 88,766           | 1         | 88,766.0   | 88,766.0   | 88,766   | 88,766   | 0.0         | calculatePartialCoefficients(const float *, const float *, float, float, float *, float *, int)|  
| 31.2     | 87,326           | 1         | 87,326.0   | 87,326.0   | 87,326   | 87,326   | 0.0         | calculatePartialSums(const float *, const float *, float *, float *, int)                      |  
| 29.8     | 83,262           | 1         | 83,262.0   | 83,262.0   | 83,262   | 83,262   | 0.0         | calculatePartialMSE(const float *, const float *, float *, int)                                |  
| 7.2      | 20,128           | 1         | 20,128.0   | 20,128.0   | 20,128   | 20,128   | 0.0         | makePredictions(const float *, float *, float, float, int)                                     |  

#### Table 3: CUDA 'cuda_gpu_mem_time_sum'  
| Time (%) | Total Time (ns) | Count | Avg (ns)   | Med (ns)   | Min (ns) | Max (ns) | StdDev (ns) | Operation                   |  
|----------|-----------------|-------|------------|------------|----------|----------|-------------|-----------------------------|  
| 94.9     | 1,382,915        | 2     | 691,457.5  | 691,457.5  | 680,370  | 702,545  | 15,680.1    | [CUDA memcpy Host-to-Device] |  
| 4.9      | 71,134           | 6     | 11,855.7   | 1,840.0    | 1,600    | 62,174   | 24,651.7    | [CUDA memcpy Device-to-Host] |  
| 0.3      | 3,840            | 4     | 960.0      | 928.0      | 672      | 1,312    | 336.6       | [CUDA memset]                |  

### cuBLAS  
#### Table 4: cuBLAS 'cuda_api_sum'  
| Time (%) | Total Time (ns) | Num Calls | Avg (ns)      | Med (ns)    | Min (ns) | Max (ns)     | StdDev (ns)    | Name                       |  
|----------|-----------------|-----------|---------------|-------------|----------|--------------|---------------|----------------------------|  
| 67.5     | 97,130,331       | 6         | 16,188,388.5  | 114,106.0   | 9,119    | 96,736,963   | 39,460,617.3  | cudaMalloc                  |  
| 28.2     | 40,578,612       | 8         | 5,072,326.5   | 169,325.0   | 3,214    | 36,343,841   | 12,693,387.7  | cudaFree                    |  
| 1.6      | 2,257,018        | 4         | 564,254.5     | 565,201.0   | 170,728  | 955,888      | 434,917.8     | cudaMemcpy                  |  
| 1.2      | 1,702,178        | 12        | 141,848.2     | 8,823.5     | 4,870    | 1,509,709    | 431,077.1     | cudaLaunchKernel            |  
| 1.2      | 1,678,509        | 18        | 93,250.5      | 378.5       | 330      | 1,665,258    | 392,322.7     | cudaEventCreateWithFlags    |  
| 0.2      | 280,738          | 766       | 366.5         | 240.0       | 90       | 49,982       | 1,909.2       | cuGetProcAddress_v2         |  
| 0.1      | 141,881          | 5         | 28,376.2      | 30,934.0    | 21,578   | 33,780       | 5,627.5       | cudaMemcpyAsync             |  
| 0.0      | 19,337           | 5         | 3,867.4       | 2,467.0     | 1,029    | 10,421       | 3,822.9       | cudaEventRecord             |  
| 0.0      | 17,083           | 25        | 683.3         | 436.0       | 234      | 3,133        | 681.9         | cudaStreamGetCaptureInfo_v2_v11030 |  
| 0.0      | 15,637           | 4         | 3,909.3       | 2,011.0     | 988      | 10,627       | 4,524.7       | cudaDeviceSynchronize       |  
| 0.0      | 13,527           | 5         | 2,705.4       | 2,376.0     | 1,756    | 4,171        | 911.6         | cudaStreamSynchronize       |  
| 0.0      | 8,490            | 18        | 471.7         | 348.0       | 285      | 1,587        | 311.9         | cudaEventDestroy            |  
| 0.0      | 6,545            | 2         | 3,272.5       | 3,272.5     | 2,481    | 4,064        | 1,119.4       | cuInit                      |  
| 0.0      | 4,780            | 1         | 4,780.0       | 4,780.0     | 4,780    | 4,780        | 0.0           | cudaEventQuery              |  
| 0.0      | 1,477            | 3         | 492.3         | 334.0       | 168      | 975          | 426.2         | cuModuleGetLoadingMode      |  

#### Table 5: cuBLAS 'cuda_gpu_kern_sum'  
| Time (%) | Total Time (ns) | Instances | Avg (ns)   | Med (ns)   | Min (ns) | Max (ns) | StdDev (ns) | Name                                                                                              |  
|----------|-----------------|-----------|------------|------------|----------|----------|-------------|---------------------------------------------------------------------------------------------------|  
| 37.7     | 53,727           | 3         | 17,909.0   | 18,239.0   | 9,216    | 26,272   | 8,532.8     | void dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensor<const float>, cublasGemvT… |  
| 35.5     | 50,622           | 4         | 12,655.5   | 12,367.5   | 4,448    | 21,439   | 9,260.7     | void asum_kernel<int, float, float>(cublasAsumParams<T2, T3>)                                      |  
| 15.5     | 22,080           | 2         | 11,040.0   | 11,040.0   | 10,912   | 11,168   | 181.0       | void axpy_kernel_val<float, float>(cublasAxpyParamsVal<T1, T1, T2>)                                |  
| 11.3     | 16,096           | 3         | 5,365.3    | 5,280.0    | 5,056    | 5,760    | 359.7       | void reduce_1Block_kernel<float, (int)128, (int)7, cublasGemvTensorStridedBatched<float>, cublasGemv… |  

#### Table 6: cuBLAS 'cuda_gpu_mem_time_sum'  
| Time (%) | Total Time (ns) | Count | Avg (ns)   | Med (ns)   | Min (ns) | Max (ns) | StdDev (ns) | Operation                   |  
|----------|-----------------|-------|------------|------------|----------|----------|-------------|-----------------------------|  
| 95.6     | 1,533,312        | 3     | 511,104.0  | 701,362.0  | 68,031   | 763,919  | 384,985.2   | [CUDA memcpy Host-to-Device] |  
| 4.4      | 70,143           | 6     | 11,690.5   | 2,080.0    | 1,600    | 60,639   | 23,980.9    | [CUDA memcpy Device-to-Host] |  


## cuBLAS  
### calculate_coefficients
2 x cublasSasum + 2 x cublasSdot  
### make_predictions  
cublasSaxpy  
### calculate_mse  
cublasSaxpy + cublasSdot  

![cuBLAS Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Perf/cuBLAS_Runtime_by_Kernel.png)  

### Example Output  
(n=1024)  
Slope: 24.999807  
Intercept: 49.886719  
Predictions  
2741: 68574.359375  
2715: 67924.367188  
913: 22874.710938  
...  
Mean Squared Error: 27.672958  

## Kernels (Plain CUDA)  
### calculatePartialCoefficients 
Calculates the numerator and denominator to be used for slope and bias.  
``slope = numerator / denominator;  bias = y_mean - slope * x_mean;``  

### calculatePartialSums 
Calculates the sums of X and Y, which are used to calculate the mean of X and Y by dividing by N.  

### calculatePartialMSE
Calculates the squared error, which is used to calculate the MSE by dividing by N.  

### makePredictions
Calculates predictions for the array of values x based off slope and bias.  
 
![CUDA Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Perf/CUDA_Runtime_by_Kernel.png)  

### Example Output  
(n=1024)
slope 24.999805  intercept 49.890625  

Predictions  
2741 : 68574.359375  
2715 : 67924.359375  
913 : 22874.712891  
...  

MSE: 27.675436  
