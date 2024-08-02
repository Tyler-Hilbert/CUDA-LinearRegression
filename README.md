## Linear regression implemented from scratch in CUDA  

## Kernels  

### calculatePartialCoefficients 
Calculates the numerator and denominator to be used for slope and bias.  
``slope = numerator / denominator;  bias = y_mean - slope * x_mean;``  

### calculatePartialSums 
Calculates the sums of X and Y, which are used to calculate the mean of X and Y by dividing by N.

### calculatePartialMSE
Calculates the squared error, which is used to calculate the MSE by dividing by N.  

### makePredictions
Calculates predictions for the array of values x based off slope and bias.
