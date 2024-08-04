//%%writefile CUDA_helpers.cu

// CUDA and cuBLAS error checkers

#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(ans) { \
    if ((ans) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %s %s %d\n", cublasGetErrorString(ans), __FILE__, __LINE__); \
        exit(ans); \
    } \
}

// Inline function to map cublasStatus_t to string within the macro
inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "UNKNOWN_CUBLAS_STATUS";
    }
}

#endif // __CUDA_HELPERS__