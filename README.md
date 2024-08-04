## Linear regression implemented from scratch in CUDA  
`LinearRegression.cu` is kernels written completely from scratch.  
`LinearRegression_cuBLAS.cu` is linear regression written from scratch but utilizing the cuBLAS library.  


## Performance Test (cuBLAS)  
Tested on Google Colab T4 with 820 training points and 204 testing points.  
![cuBLAS Performance Test](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/cf5575b62348c939271295fd0e2cec309fc54636/cuBLAS.png)

### Example Output
Slope: 24.999807  
Intercept: 49.886719  
Predictions (first 10)  
2741.000000: 68574.359375  
2715.000000: 67924.367188  
913.000000: 22874.710938  
696.000000: 17449.751953  
67.000000: 1724.873779  
2998.000000: 74999.312500  
441.000000: 11074.801758  
1077.000000: 26974.679688  
711.000000: 17824.750000  
3687.000000: 92224.179688  
Mean Squared Error: 27.672958  
constructor:    117601702 ns  
coefficients:   1334041 ns  
predictions:    449526 ns  
mse:	        43622 ns
## Kernels (Plain CUDA)  

### calculatePartialCoefficients 
Calculates the numerator and denominator to be used for slope and bias.  
``slope = numerator / denominator;  bias = y_mean - slope * x_mean;``  
Runtime: 0.053 ms  

### calculatePartialSums 
Calculates the sums of X and Y, which are used to calculate the mean of X and Y by dividing by N.  
Runtime: 0.279 ms  

### calculatePartialMSE
Calculates the squared error, which is used to calculate the MSE by dividing by N.  
Runtime: 0.038 ms  

### makePredictions
Calculates predictions for the array of values x based off slope and bias.  
Runtime: 0.167 ms

## Performance Test (Plain CUDA)  
Test ran on Google Collab A100. CUDA 12.2    

### Example Output  
Linear regression from scratch in CUDA. Dataset in .cu file  

 -- calculatePartialSums time:  0.315680 ms  

slope 24.999830  bias 53.343750  

 -- calculatePartialCoefficients time:  0.058016 ms  

Predictions:  
2260 : 54293  
2981 : 71597  
56 : 1397  
747 : 17981  
2724 : 65429  
1475 : 35453  
198 : 4805  
1313 : 31565  
2185 : 52493  
1461 : 35117  

 -- makePredictions time:  0.050976 ms  

MSE: 5552492.500000  

 -- calculatePartialMSE time:  0.046944 ms    
