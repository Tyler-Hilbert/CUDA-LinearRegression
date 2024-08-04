## Linear regression implemented from scratch in CUDA  

## Kernels  

### calculatePartialCoefficients 
Calculates the numerator and denominator to be used for slope and bias.  
``slope = numerator / denominator;  bias = y_mean - slope * x_mean;``  
Runtime: 0.053024 ms  

### calculatePartialSums 
Calculates the sums of X and Y, which are used to calculate the mean of X and Y by dividing by N.  
Runtime: 0.279648 ms  

### calculatePartialMSE
Calculates the squared error, which is used to calculate the MSE by dividing by N.  
Runtime: 0.038624 ms  

### makePredictions
Calculates predictions for the array of values x based off slope and bias.  
Runtime: 0.167296 ms

## Performance Test  
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