#ifndef FF_CUH
#define FF_CUH

#include <assert.h>
#include "common.cuh"

typedef ulong FieldLimb;

template <
    const uint LIMBS,
    const FieldLimb MODULUS[LIMBS],
    const FieldLimb R[LIMBS],
    const FieldLimb R2[LIMBS],
    const FieldLimb INV>
class Field
{
private:
    // little-endian
    FieldLimb limbs_le[LIMBS];

    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
    __device__ static bool _mont_reduce(
        FieldLimb *lo, ulong h0, ulong h1, ulong h2, ulong h3)
    {
        if (LIMBS == 4)
        {
            mont_reduce_u64x8(
                (ulong *)&lo[0], (ulong *)&lo[1], (ulong *)&lo[2], (ulong *)&lo[3],
                &h0, &h1, &h2, &h3,
                (const ulong *)MODULUS, (ulong)INV);
            lo[0] = h0;
            lo[1] = h1;
            lo[2] = h2;
            lo[3] = h3;
            if (_gte(lo, MODULUS))
            {
                sub_u64x4((ulong *)lo, (ulong *)lo, (ulong *)MODULUS);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static bool _gte(const FieldLimb *a, const FieldLimb *b)
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

    __device__ static void _mul(FieldLimb *out, const FieldLimb *a, const FieldLimb *b)
    {
        assert(a != out);
        assert(b != out);
        if (LIMBS == 4)
        {
            ulong h0 = 0;
            ulong h1 = 0;
            ulong h2 = 0;
            ulong h3 = 0;
            mul_u64x4(
                (ulong *)&out[0], (ulong *)&out[1], (ulong *)&out[2], (ulong *)&out[3],
                &h0, &h1, &h2, &h3,
                (const ulong *)a, (const ulong *)b);
            _mont_reduce(out, h0, h1, h2, h3);
        }
        else
        {
            assert(0);
        }
    }

public:
    __device__ Field(bool reset = false)
    {
        if (reset)
        {
            memset(limbs_le, 0, sizeof(FieldLimb) * LIMBS);
        }
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

    __device__ static void add_no_copy(Field *out, const Field *a, const Field *b)
    {
        if (LIMBS == 4)
        {
            add_u64x4((ulong *)out, (const ulong *)a, (const ulong *)b);
            if (_gte(out->limbs_le, MODULUS))
            {
                sub_u64x4((ulong *)out->limbs_le, (const ulong *)out->limbs_le, (const ulong *)MODULUS);
            }
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void sub_no_copy(Field *out, const Field *a, const Field *b)
    {
        if (LIMBS == 4)
        {
            uint borrow = sub_u64x4_with_borrow((ulong *)out, (const ulong *)a, (const ulong *)b);
            if (borrow)
            {
                add_u64x4((ulong *)out, (const ulong *)out, (const ulong *)MODULUS);
            }
        }
        else
        {
            assert(0);
        }
    }

    // out can't overlap with a or b
    __device__ static void mul_no_copy(Field *out, const Field *a, const Field *b)
    {
        memset(out, 0, sizeof(*out));
        _mul(out->limbs_le, a->limbs_le, b->limbs_le);
    }

    __device__ static Field add(const Field *a, const Field *b)
    {
        Field tmp;
        add_no_copy(&tmp, a, b);
        return tmp;
    }

    __device__ static Field sub(const Field *a, const Field *b)
    {
        Field tmp;
        sub_no_copy(&tmp, a, b);
        return tmp;
    }

    __device__ static Field mul(const Field *a, const Field *b)
    {
        Field tmp(true);
        _mul(tmp.limbs_le, a->limbs_le, b->limbs_le);
        return tmp;
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
            Field tmp(true);
            _mul(tmp.limbs_le, a->limbs_le, R2);
            *a = tmp;
        }
        else
        {
            assert(0);
        }
    }

    __forceinline__ __device__ static Field sqr(Field *out, const Field *a)
    {
        mul(out, a, a);
    }

    __forceinline__ __device__ static Field one()
    {
        Field tmp;
        memcpy(tmp->limbs_le, R, LIMBS * sizeof(FieldLimb));
        return tmp;
    }

    __forceinline__ __device__ static Field zero()
    {
        return Field(true);
    }

    __device__ static void exp(Field *out, const Field *a, ulong m)
    {
        Field acc = Field::one();
        if (m > 0)
        {
            Field base = *a;

            if (m & 1)
            {
                acc = mul(&acc, &base);
                base = sqr(&base);
            }
        }

        *out = acc;
    }
};
#endif