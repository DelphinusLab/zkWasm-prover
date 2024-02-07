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

__device__ const ulong Bn254FrR2[4] = {
    0x1bb8e645ae216da7ul,
    0x53fe3ab1e35c59e3ul,
    0x8c49833d53bb8085ul,
    0x0216d0b17f4e44a5ul,
};

__device__ const ulong Bn254FrInv = 0xc2e1f593effffffful;

typedef Field<4, Bn254FrModulus, Bn254FrR2, Bn254FrInv> Bn254FrField;

// Tests

__global__ void _test_bn254_fr_field_compare(Bn254FrField *a, Bn254FrField *b, bool *c, int n)
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

__global__ void _test_bn254_fr_field_add(Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
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

__global__ void _test_bn254_fr_field_sub(Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
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

__global__ void _test_bn254_fr_field_mul(Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254FrField::mul(&a[i], &b[i], &c[i]);
    }
}

__global__ void _test_bn254_fr_field_mont(Bn254FrField *a, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254FrField::unmont(&a[i]);
        Bn254FrField::mont(&a[i]);
    }
}

__global__ void _test_bn254_fr_field_unmont(Bn254FrField *a, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254FrField::unmont(&a[i]);
    }
}

extern "C"
{
    cudaError_t test_bn254_fr_field_compare(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, bool *c, int n)
    {
        _test_bn254_fr_field_compare<<<blocks, threads>>>(a, b, c, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fr_field_add(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
    {
        _test_bn254_fr_field_add<<<blocks, threads>>>(a, b, c, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fr_field_sub(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
    {
        _test_bn254_fr_field_sub<<<blocks, threads>>>(a, b, c, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fr_field_mul(int blocks, int threads, Bn254FrField *a, Bn254FrField *b, Bn254FrField *c, int n)
    {
        _test_bn254_fr_field_mul<<<blocks, threads>>>(a, b, c, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fr_field_mont(int blocks, int threads, Bn254FrField *a, int n)
    {
        _test_bn254_fr_field_mont<<<blocks, threads>>>(a, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fr_field_unmont(int blocks, int threads, Bn254FrField *a, int n)
    {
        _test_bn254_fr_field_unmont<<<blocks, threads>>>(a, n);
        return cudaGetLastError();
    }
}
