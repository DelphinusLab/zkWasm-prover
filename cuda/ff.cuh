#ifndef FF_CUH
#define FF_CUH

#include <assert.h>
#include "common.cuh"

typedef ulong FieldLimb;

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
    __device__ static bool _mont_reduce(
        ulong *lo, ulong h0, ulong h1, ulong h2, ulong h3)
    {
        if (LIMBS == 4)
        {
            mont_reduce_u64x8(&lo[0], &lo[1], &lo[2], &lo[3], &h0, &h1, &h2, &h3, MODULUS, INV);
            lo[0] = h0;
            lo[1] = h1;
            lo[2] = h2;
            lo[3] = h3;
            if (_gte(lo, MODULUS))
            {
                sub_u64x4((ulong *)lo, MODULUS, (ulong *)lo);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static bool _gte(const ulong *a, const ulong *b)
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

public:
    Field(/* args */) {}
    ~Field() {}

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
            if (gte(c, (Field *)MODULUS))
            {
                sub_u64x4((ulong *)c, MODULUS, (ulong *)c);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void sub(const Field *a, const Field *b, Field *c)
    {
        if (LIMBS == 4)
        {
            uint borrow = sub_u64x4_with_borrow((ulong *)a, (ulong *)b, (ulong *)c);
            if (borrow)
            {
                add_u64x4((ulong *)c, MODULUS, (ulong *)c);
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
            assert(0);
        }
        else
        {
            assert(0);
        }
    }
};
#endif