//Code from https://github.com/matter-labs/z-prize-msm-gpu-combined/tree/7699c8f1e7cd093bf569ed364ee2b5faf71418c3
#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(statement)                                                                                                                           \
  {                                                                                                                                                            \
    cudaError_t hce_result = (statement);                                                                                                                      \
    if (hce_result != cudaSuccess)                                                                                                                             \
      return hce_result;                                                                                                                                       \
  }

#ifdef __CUDA_ARCH__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
