#ifndef FF_CUH
#define FF_CUH

#include <assert.h>
#include "common.cuh"

typedef unsigned long FieldLimb;

template <
    const uint LIMBS,
    const FieldLimb MODULUS[LIMBS],
    const FieldLimb R2[LIMBS],
    const FieldLimb INV>
class Field
{
private:
    // little-endian
    FieldLimb limbs_le[LIMBS];

    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
    __forceinline__ __device__ static bool _mont_reduce(
        FieldLimb *lo, ulong h0, ulong h1, ulong h2, ulong h3)
    {
        if (LIMBS == 4)
        {
            mont_reduce_u64x8(
                (ulong *)&lo[0], (ulong *)&lo[1], (ulong *)&lo[2], (ulong *)&lo[3],
                &h0, &h1, &h2, &h3,
                MODULUS, INV);
            lo[0] = h0;
            lo[1] = h1;
            lo[2] = h2;
            lo[3] = h3;
            if (_gte(lo, MODULUS))
            {
                sub_u64x4((ulong *)lo, (ulong *)MODULUS, (ulong *)lo);
            }
        }
        else
        {
            assert(0);
        }
    }

    __forceinline__ __device__ static bool _gte(const FieldLimb *a, const FieldLimb *b)
    {
#pragma unroll
        for (int i = LIMBS - 1; i >= 0; i--)
        {
            if (a[i] > b[i])
                return true;
            if (a[i] < b[i])
                return false;
        }
        return true;
    }

    __forceinline__ __device__ static void _mul(const FieldLimb *a, const FieldLimb *b, FieldLimb *c)
    {
        assert(a != c);
        assert(b != c);
        if (LIMBS == 4)
        {
            ulong h0 = 0;
            ulong h1 = 0;
            ulong h2 = 0;
            ulong h3 = 0;
            mul_u64x4(
                (ulong *)&c[0], (ulong *)&c[1], (ulong *)&c[2], (ulong *)&c[3],
                &h0, &h1, &h2, &h3,
                (ulong *)a, (ulong *)b);
            _mont_reduce(c, h0, h1, h2, h3);
            if (_gte(c, MODULUS))
            {
                sub_u64x4((ulong *)c, (ulong *)MODULUS, (ulong *)c);
            }
        }
        else
        {
            assert(0);
        }
    }

public:
    __device__ Field()
    {
        memset(limbs_le, 0, sizeof(FieldLimb) * LIMBS);
    }

    __device__ ~Field() {}

    __device__ static bool gte(const Field *a, const Field *b)
    {
        return _gte(a->limbs_le, b->limbs_le);
    }

    __device__ static bool eq(const Field *a, const Field *b)
    {
#pragma unroll
        for (uint i = 0; i < LIMBS; i++)
            if (a->limbs_le[i] != b->limbs_le[i])
                return false;
        return true;
    }

    __device__ static void add(const Field *a, const Field *b, Field *c)
    {
        if (LIMBS == 4)
        {
            add_u64x4((ulong *)a, (ulong *)b, (ulong *)c);
            if (_gte(c->limbs_le, MODULUS))
            {
                sub_u64x4((ulong *)c->limbs_le, (ulong *)MODULUS, (ulong *)c->limbs_le);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void mul(const Field *a, const Field *b, Field *c)
    {
        Field tmp;
        _mul(a->limbs_le, b->limbs_le, tmp.limbs_le);
        *c = tmp;
    }

    __device__ static void sub(const Field *a, const Field *b, Field *c)
    {
        if (LIMBS == 4)
        {
            uint borrow = sub_u64x4_with_borrow((ulong *)a, (ulong *)b, (ulong *)c);
            if (borrow)
            {
                add_u64x4((ulong *)c, (ulong *)MODULUS, (ulong *)c);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void unmont(Field *a)
    {
        if (LIMBS == 4)
        {
            _mont_reduce(a->limbs_le, 0, 0, 0, 0);
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void mont(Field *a)
    {
        if (LIMBS == 4)
        {
            Field tmp;
            _mul(a->limbs_le, R2, tmp.limbs_le);
            *a = tmp;
        }
        else
        {
            assert(0);
        }
    }
};
#endif