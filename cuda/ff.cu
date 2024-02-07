#include "cuda_runtime.h"
#include <stdio.h>

#include "common.cuh"
#include "ff.cuh"

__device__ const ulong Bn254FrModulus[4] = {
    0x43e1f593f0000001ul,
    0x2833e84879b97091ul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

typedef Field<4, Bn254FrModulus> Bn254FrField;

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

__global__ void _test_bn254_field_compare(Bn254FrField *a, Bn254FrField *b, bool *c, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        c[i] = Bn254FrField::gte(&a[i], &b[i]);
    }
}

__global__ void _test_bn254_field_add(Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254FrField::add(&a[i], &b[i], &c[i]);
    }
}

__global__ void _test_bn254_field_sub(Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254FrField::sub(&a[i], &b[i], &c[i]);
    }
}

extern "C"
{
    void test_int_add(int blocks, int threads, int *a, int n)
    {
        _test_int_add<<<blocks, threads>>>(a, n);
    }

    void test_bn254_field_compare(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, bool *c, int n)
    {
        _test_bn254_field_compare<<<blocks, threads>>>(a, b, c, n);
    }

    void test_bn254_field_add(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
    {
        _test_bn254_field_add<<<blocks, threads>>>(a, b, c, n);
    }

    cudaError_t  test_bn254_field_sub(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
    {
        _test_bn254_field_sub<<<blocks, threads>>>(a, b, c, n);
        return cudaGetLastError();
    }
}
