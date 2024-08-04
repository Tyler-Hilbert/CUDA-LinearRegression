This branch is testing cuBLAS vs CUDA from scratch for sum.  

 -- calculatePartialSums time:  105.296387 ms  
 -- cuBLAS time:  9.174368 ms  

 An interesting thing to not is that when the CUDA from scratch kernel was changed to use ints instead of floats, it ran in under 1ms.