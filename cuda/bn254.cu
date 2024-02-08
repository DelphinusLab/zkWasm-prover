#include "cuda_runtime.h"
#include <stdio.h>

#include "common.cuh"
#include "ff.cuh"
#include "ec.cuh"

__device__ const ulong BN254_FR_MODULUS[4] = {
    0x43e1f593f0000001ul,
    0x2833e84879b97091ul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FR_NEG_TWO[4] = {
    0x43e1f593effffffful,
    0x2833e84879b97091ul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FR_R[4] = {
    0xac96341c4ffffffbul,
    0x36fc76959f60cd29ul,
    0x666ea36f7879462eul,
    0x0e0a77c19a07df2ful,
};

__device__ const ulong BN254_FR_R2[4] = {
    0x1bb8e645ae216da7ul,
    0x53fe3ab1e35c59e3ul,
    0x8c49833d53bb8085ul,
    0x0216d0b17f4e44a5ul,
};

__device__ const ulong Bn254_FR_INV = 0xc2e1f593effffffful;

typedef Field<4, BN254_FR_MODULUS, BN254_FR_NEG_TWO, BN254_FR_R, BN254_FR_R2, Bn254_FR_INV> Bn254FrField;

__device__ const ulong BN254_FP_MODULUS[4] = {
    0x3c208c16d87cfd47ul,
    0x97816a916871ca8dul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FP_NEG_TWO[4] = {
    0x3c208c16d87cfd45ul,
    0x97816a916871ca8dul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FP_R[4] = {
    0xd35d438dc58f0d9dul,
    0x0a78eb28f5c70b3dul,
    0x666ea36f7879462cul,
    0x0e0a77c19a07df2ful,
};

__device__ const ulong BN254_FP_R2[4] = {
    0xf32cfc5b538afa89ul,
    0xb5e71911d44501fbul,
    0x47ab1eff0a417ff6ul,
    0x06d89f71cab8351ful,
};

__device__ const ulong Bn254_FP_INV = 0x87d20782e4866389ul;

typedef Field<4, BN254_FP_MODULUS, BN254_FP_NEG_TWO, BN254_FP_R, BN254_FP_R2, Bn254_FP_INV> Bn254FpField;

typedef CurveAffine<Bn254FpField> Bn254G1Affine;
typedef Curve<Bn254FpField> Bn254G1;

// Tests

__global__ void _test_bn254_ec(
    const Bn254G1Affine *a,
    const Bn254G1Affine *b,
    Bn254G1Affine *add,
    Bn254G1Affine *sub,
    Bn254G1Affine *_double,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        Bn254G1 _a(a[i]);
        Bn254G1 _b(b[i]);

        add[i] = _a + _b;
        assert(add[i] == _a + b[i]);
        sub[i] = _a - _b;
        _double[i] = _a + _a;
        assert(_double[i] == _a.ec_double());
        assert(a[i] == _a + Bn254G1::identity());
        assert(a[i] == _a + Bn254G1Affine::identity());
        assert(a[i] == Bn254G1::identity() + _a);
        assert(a[i] == Bn254G1::identity() + _a);
        assert(_a - a[i] == Bn254G1::identity());
    }
}

__global__ void _test_bn254_fr_field(
    const Bn254FrField *a,
    const Bn254FrField *b,
    const ulong *exp,
    Bn254FrField *add,
    Bn254FrField *sub,
    Bn254FrField *mul,
    Bn254FrField *sqr,
    Bn254FrField *inv,
    Bn254FrField *pow,
    Bn254FrField *unmont,
    bool *compare,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        add[i] = Bn254FrField::add(&a[i], &b[i]);
        sub[i] = Bn254FrField::sub(&a[i], &b[i]);
        mul[i] = Bn254FrField::mul(&a[i], &b[i]);
        sqr[i] = Bn254FrField::sqr(&a[i]);
        inv[i] = Bn254FrField::inv(&a[i]);
        pow[i] = Bn254FrField::pow(&a[i], exp[i]);

        {
            unmont[i] = a[i];
            Bn254FrField::unmont(&unmont[i]);
        }

        {
            Bn254FrField t = unmont[i];
            Bn254FrField::mont(&t);
            assert(Bn254FrField::eq(&t, &a[i]));
        }

        {
            Bn254FrField l = a[i];
            Bn254FrField r = b[i];
            Bn254FrField::unmont(&l);
            Bn254FrField::unmont(&r);
            compare[i] = Bn254FrField::gte(&l, &r);
        }
    }
}

__global__ void _test_bn254_fp_field(
    const Bn254FpField *a,
    const Bn254FpField *b,
    const ulong *exp,
    Bn254FpField *add,
    Bn254FpField *sub,
    Bn254FpField *mul,
    Bn254FpField *sqr,
    Bn254FpField *inv,
    Bn254FpField *pow,
    Bn254FpField *unmont,
    bool *compare,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;

    for (int i = start; i < end; i++)
    {
        add[i] = Bn254FpField::add(&a[i], &b[i]);
        sub[i] = Bn254FpField::sub(&a[i], &b[i]);
        mul[i] = Bn254FpField::mul(&a[i], &b[i]);
        sqr[i] = Bn254FpField::sqr(&a[i]);
        inv[i] = Bn254FpField::inv(&a[i]);
        pow[i] = Bn254FpField::pow(&a[i], exp[i]);

        {
            unmont[i] = a[i];
            Bn254FpField::unmont(&unmont[i]);
        }

        {
            Bn254FpField t = unmont[i];
            Bn254FpField::mont(&t);
            assert(Bn254FpField::eq(&t, &a[i]));
        }

        {
            Bn254FpField l = a[i];
            Bn254FpField r = b[i];
            Bn254FpField::unmont(&l);
            Bn254FpField::unmont(&r);
            compare[i] = Bn254FpField::gte(&l, &r);
        }
    }
}

extern "C"
{
    cudaError_t test_bn254_fr_field(
        int blocks, int threads,
        const Bn254FrField *a,
        const Bn254FrField *b,
        const ulong *exp,
        Bn254FrField *add,
        Bn254FrField *sub,
        Bn254FrField *mul,
        Bn254FrField *sqr,
        Bn254FrField *inv,
        Bn254FrField *pow,
        Bn254FrField *unmont,
        bool *compare,
        int n)
    {
        _test_bn254_fr_field<<<blocks, threads>>>(a, b, exp, add, sub, mul, sqr, inv, pow, unmont, compare, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_fp_field(
        int blocks, int threads,
        const Bn254FpField *a,
        const Bn254FpField *b,
        const ulong *exp,
        Bn254FpField *add,
        Bn254FpField *sub,
        Bn254FpField *mul,
        Bn254FpField *sqr,
        Bn254FpField *inv,
        Bn254FpField *pow,
        Bn254FpField *unmont,
        bool *compare,
        int n)
    {
        _test_bn254_fp_field<<<blocks, threads>>>(a, b, exp, add, sub, mul, sqr, inv, pow, unmont, compare, n);
        return cudaGetLastError();
    }

    cudaError_t test_bn254_ec(
        int blocks, int threads,
        const Bn254G1Affine *a,
        const Bn254G1Affine *b,
        Bn254G1Affine *add,
        Bn254G1Affine *sub,
        Bn254G1Affine *_double,
        int n)
    {
        _test_bn254_ec<<<blocks, threads>>>(a, b, add, sub, _double, n);
        return cudaGetLastError();
    }
}
