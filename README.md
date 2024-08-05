## Linear regression implemented from scratch in CUDA  
`LinearRegression_cuBLAS.cu` is linear regression written from scratch but utilizing the cuBLAS library.  
`LinearRegression_CUDA.cu` is linear regression with kernels written completely from scratch.  

## cuBLAS vs CUDA Performance  
Tested on T4 with 820 training points and 204 testing points using CUDA 12.2.  
![cuBLAS vs CUDA Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/8d2e2abda8d3b83918a7df89ad9eb6898b810db9/cuBLAS_vs_CUDA_Compare.png)

| Task (T4 1024 Data Points) | cuBLAS Time (ms) | CUDA Time (ms) |
|-------------------------|------------------|----------------|
| Setting up GPU Memory   | 117.60           | 109.55         |
| Calculating Coefficients| 1.33             | 0.41           |
| Making Predictions      | 0.45             | 0.09           |
| Calculating MSE         | 0.04             | 0.05           |

### Observations

- **Setting up GPU Memory** is slightly faster with CUDA, saving about 8 ms.
- **Calculating Coefficients** is significantly faster with CUDA, achieving the task in about 0.41 ms compared to 1.33 ms with cuBLAS.
- **Making Predictions** is also more efficient with CUDA, with 0.09 ms versus 0.45 ms for cuBLAS.
- **Calculating MSE** is slightly faster with cuBLAS, but the difference is minimal.

This table shows a side-by-side comparison of the task runtimes, allowing for easy visualization of performance differences between cuBLAS and CUDA implementations on the T4 GPU. If you need further assistance or more information, feel free to ask!

### Observations

- **Setting up GPU Memory** is slightly faster with CUDA.
- **Calculating Coefficients** is significantly faster with CUDA.
- **Making Predictions** shows better performance with CUDA.
- **Calculating MSE** is marginally faster with cuBLAS.

This table provides a clear and concise comparison of how each task performs using cuBLAS versus CUDA. If there's anything else you need or any additional analysis you would like, feel free to ask!

## Performance Test (cuBLAS)  
Tested on Google Colab T4 with 820 training points and 204 testing points using CUDA 12.2.  
![cuBLAS Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/cf5575b62348c939271295fd0e2cec309fc54636/cuBLAS.png)

### Example Output
Slope: 24.999807  
Intercept: 49.886719  
Predictions  
2741: 68574.359375  
2715: 67924.367188  
913: 22874.710938  
...  
Mean Squared Error: 27.672958  
constructor:    117601702 ns  
coefficients:   1334041 ns  
predictions:    449526 ns  
mse:	           43622 ns

## Kernels (Plain CUDA)  

### calculatePartialCoefficients 
Calculates the numerator and denominator to be used for slope and bias.  
``slope = numerator / denominator;  bias = y_mean - slope * x_mean;``  
Runtime (1024 points): 0.053 ms  

### calculatePartialSums 
Calculates the sums of X and Y, which are used to calculate the mean of X and Y by dividing by N.  
Runtime (1024 points): 0.279 ms  

### calculatePartialMSE
Calculates the squared error, which is used to calculate the MSE by dividing by N.  
Runtime (1024 points): 0.038 ms  

### makePredictions
Calculates predictions for the array of values x based off slope and bias.  
Runtime (1024 points): 0.167 ms

## Performance Test (Plain CUDA)  
Tested on Google Colab T4 with 820 training points and 204 testing points using CUDA 12.2.  
![CUDA Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/8d2e2abda8d3b83918a7df89ad9eb6898b810db9/CUDA.png)

### Example Output  
slope 24.999805  intercept 49.890625  

Predictions  
2741 : 68574.359375  
2715 : 67924.359375  
913 : 22874.712891  
...  

MSE: 27.675436  

h data:	        2784 ns  
constructor:	109547919 ns  
coefficients:	407586 ns  
predictions:	92424 ns  
mse:	        53760 ns  
