#include "cuda_runtime.h"
#include <stdio.h>

__global__ void _test_int_add(int *a, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;


    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    for (int i = start; i < end; i++)
    {
        a[i] += 1;
    }
    printf("do add %p\n", a);
}

extern "C"
{
    void test_int_add(int blocks, int threads, int *a, int n)
    {
        printf("do add %d %d %p\n", threads, blocks, a);
        _test_int_add<<<blocks, threads>>>(a, n);
    }

    int _main();
}
