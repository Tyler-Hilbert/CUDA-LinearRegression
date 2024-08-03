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
Test ran on Google Collab A100  
nvidia-smi  
Sat Aug  3 23:22:33 2024  
+---------------------------------------------------------------------------------------+  
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |  
|-----------------------------------------+----------------------+----------------------+  
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |  
|                                         |                      |               MIG M. |  
|=========================================+======================+======================|  
|   0  NVIDIA A100-SXM4-40GB          Off | 00000000:00:04.0 Off |                    0 |  
| N/A   32C    P0              46W / 400W |      2MiB / 40960MiB |      0%      Default |  
|                                         |                      |             Disabled |  
+-----------------------------------------+----------------------+----------------------+  

## Example Output  
calculatePartialSums time:  0.279648 ms  
slope 24.999830  bias 53.343750  
calculatePartialCoefficients time:  0.053024 ms  
Predictions: 54293 71597 1397 17981 65429 35453 4805 31565 52493 35117 36677 96677 40181 51773 86813 10733 96989 25613 40853 77285...  
makePredictions time:  0.167296 ms  
MSE: 5552489.000000  
calculatePartialMSE time:  0.038624 ms  
