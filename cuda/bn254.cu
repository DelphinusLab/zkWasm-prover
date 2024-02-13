#include "cuda_runtime.h"
#include <stdio.h>
#include <assert.h>

#include "bn254.cuh"

__global__ void _msm_mont_unmont(
    Bn254G1Affine *p,
    Bn254FrField *s,
    bool mont,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        if (mont)
        {
            s[i].mont_assign();
        }
        else
        {
            s[i].unmont_assign();
        }
    }
}

__global__ void _msm_core(
    Bn254G1 *res,
    const Bn254G1Affine *p,
    Bn254FrField *s,
    int n)
{
    int group_idx = blockIdx.x;
    int worker = blockDim.x * gridDim.y;
    int size_per_worker = (n + worker - 1) / worker;
    int inner_idx = threadIdx.x;
    int window_idx = inner_idx + blockIdx.y * blockDim.x;
    int start = window_idx * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    __shared__ Bn254G1 thread_res[256];

    Bn254G1 buckets[256];

    for (int i = start; i < end; i++)
    {
        int v = s[i].get_8bits(group_idx);
        if (v--)
        {
            buckets[v] = buckets[v] + p[i];
        }
    }

    if (end > start)
    {
        Bn254G1 round;
        Bn254G1 acc;
        for (int i = 254; i >= 0; i--)
        {
            round = round + buckets[i];
            acc = acc + round;
        }

        thread_res[inner_idx] = acc;
    }

    __syncthreads();
    if (inner_idx == 0)
    {
        Bn254G1 acc;
        for (int i = 0; i < blockDim.x; i++)
        {
            acc = acc + thread_res[i];
        }
        res[group_idx + blockIdx.y * gridDim.x] = acc;
    }
}
/*
__global__ void _ntt_core(
    Bn254FrField *buf,
    const Bn254FrField *omega,
    int log_n,
    int _round)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int n = 1 << (log_n - 1);
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int round = 0; round < 8; round++)
    {
        int shift = 1 << round;
        int mask = shift - 1;

        int twiddle_chunk = log_n - (round + 1);

        for (int i = start; i < end; i++)
        {
            int inner = i & mask;
            int outer = (i - inner);
            int l = (outer << 1) + inner;
            int r = l + shift;

            Bn254FrField t = buf[r];
            if (inner != 0)
            {
                const Bn254FrField *twiddle = &omega[0];
                t = t * *twiddle;
            }
            buf[r] = buf[l] - t;
            buf[l] += t;
        }

        __syncthreads();
    }
}
*/

__device__ uint bit_reverse(uint n, uint bits)
{
    uint r = 0;
    for (int i = 0; i < bits; i++)
    {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

__device__ Bn254FrField pow_lookup(const Bn254FrField *bases, uint exponent)
{
    Bn254FrField res(1);
    uint i = 0;
    while (exponent > 0)
    {
        if (exponent & 1)
            res = res * bases[i];
        exponent = exponent >> 1;
        i++;
    }
    return res;
}

__global__ void _ntt_core(
    const Bn254FrField *_x,
    Bn254FrField *_y,
    const Bn254FrField *pq,
    const Bn254FrField *omegas,
    uint n,     // Number of elements
    uint log_p, // Log2 of `p` (Read more in the link above)
    uint deg,   // 1=>radix2, 2=>radix4, 3=>radix8, ...
    uint max_deg,
    uint grids) // Maximum degree supported, according to `pq` and `omegas`
{
    uint lid = threadIdx.x;
    uint lsize = blockDim.x;
    uint t = n >> deg;
    uint p = 1 << log_p;

    uint count = 1 << deg;
    uint counth = count >> 1;
    uint counts = count / lsize * lid;
    uint counte = counts + count / lsize;

    const uint pqshift = max_deg - deg;

    for (uint gridIdx = 0; gridIdx < grids; gridIdx++)
    {
        uint index = blockIdx.x + gridIdx * gridDim.x;
        uint k = index & (p - 1);

        const Bn254FrField *x = _x + index;
        Bn254FrField *y = _y + ((index - k) << deg) + k;

        __shared__ Bn254FrField u[512];
        const Bn254FrField twiddle = pow_lookup(omegas, (n >> log_p >> deg) * k);
        Bn254FrField tmp = Bn254FrField::pow(&twiddle, counts);
        for (uint i = counts; i < counte; i++)
        {
            u[i] = tmp * x[i * t];
            tmp = tmp * twiddle;
        }
        __syncthreads();

        for (uint rnd = 0; rnd < deg; rnd++)
        {
            const uint bit = counth >> rnd;
            for (uint i = counts >> 1; i < counte >> 1; i++)
            {
                const uint di = i & (bit - 1);
                const uint i0 = (i << 1) - di;
                const uint i1 = i0 + bit;
                tmp = u[i0];
                u[i0] += u[i1];
                u[i1] = tmp - u[i1];

                if (di != 0)
                    u[i1] = pq[di << rnd << pqshift] * u[i1];
            }

            __syncthreads();
        }

        for (uint i = counts >> 1; i < counte >> 1; i++)
        {
            y[i * p] = u[bit_reverse(i, deg)];
            y[(i + counth) * p] = u[bit_reverse(i + counth, deg)];
        }

        __syncthreads();
    }
}

extern "C"
{
    cudaError_t ntt(
        Bn254FrField *buf,
        Bn254FrField *tmp,
        const Bn254FrField *pq,
        const Bn254FrField *omegas,
        int log_n,
        int max_deg,
        bool *swap)
    {
        int p = 0;

        Bn254FrField *src = buf;
        Bn254FrField *dst = tmp;
        int len = 1 << log_n;
        int total = 1 << (log_n - 1);
        while (p < log_n)
        {
            int res = log_n - p;
            int round = (res + max_deg - 1) / max_deg;
            int deg = (res + round - 1) / round;

            int threads = 1 << (deg - 1);
            int blocks = total >> (deg - 1);
            blocks = blocks > 65536 ? 65536 : blocks;
            int grids = (total / blocks) >> (deg - 1);
            _ntt_core<<<blocks, threads>>>(src, dst, pq, omegas, len, p, deg, max_deg, grids);

            Bn254FrField *t = src;
            src = dst;
            dst = t;
            p += deg;
            *swap = !*swap;
        }
        return cudaGetLastError();
    }

    cudaError_t msm(
        int msm_blocks,
        int max_msm_threads,
        Bn254G1 *res,
        Bn254G1Affine *p,
        Bn254FrField *s,
        int n)
    {
        int threads = n >= 256 ? 256 : 1;
        int blocks = (n + threads - 1) / threads;
        _msm_mont_unmont<<<blocks, threads>>>(p, s, false, n);
        _msm_core<<<dim3(32, msm_blocks), threads>>>(res, p, s, n);
        _msm_mont_unmont<<<blocks, threads>>>(p, s, true, n);
        return cudaGetLastError();
    }
}
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
        Bn254G1 zero;
        Bn254G1 _a(a[i]);
        Bn254G1 _b(b[i]);

        add[i] = _a + _b;
        assert(add[i] == _a + b[i]);
        assert(zero + a[i] + b[i] == add[i]);
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

/*
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
*/

extern "C"
{
    /*
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
    */

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