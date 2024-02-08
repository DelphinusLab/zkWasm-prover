#ifndef FF_CUH
#define FF_CUH

#include <assert.h>
#include "common.cuh"

typedef ulong FieldLimb;

template <
    const uint LIMBS,
    const FieldLimb MODULUS[LIMBS],
    const FieldLimb NEG_TOW[LIMBS],
    const FieldLimb R[LIMBS],
    const FieldLimb R2[LIMBS],
    const FieldLimb INV>
class Field
{
private:
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
            ulong l0 = 0;
            ulong l1 = 0;
            ulong l2 = 0;
            ulong l3 = 0;

            mul_u64x4(
                &l0, &l1, &l2, &l3,
                &h0, &h1, &h2, &h3,
                (const ulong *)a, (const ulong *)b);

            out[0] = l0;
            out[1] = l1;
            out[2] = l2;
            out[3] = l3;
            _mont_reduce(out, h0, h1, h2, h3);
        }
        else
        {
            assert(0);
        }
    }

    __device__ static void _pow_at_leading(Field *acc, const Field *base, ulong exp)
    {
        Field t = *base;

        while (exp > 0)
        {
            if (exp & 1)
            {
                *acc = mul(acc, &t);
            }

            exp >>= 1;

            if (exp > 0)
            {
                t = sqr(&t);
            }
        }
    }

    __device__ static void _pow_at_nonleading(Field *acc, const Field *base, ulong exp)
    {
        for (ulong bit = 1ul << (sizeof(ulong) * 8 - 1); bit != 0; bit >>= 1)
        {
            *acc = sqr(acc);
            if (exp & bit)
            {
                *acc = mul(acc, base);
            }
        }
    }

public:
    // little-endian
    FieldLimb limbs_le[LIMBS];

    __device__ Field()
    {
    }

    __device__ Field(ulong v)
    {
        if (v == 0)
        {
            memset(limbs_le, 0, sizeof(FieldLimb) * LIMBS);
        }
        else if (v == 1)
        {
            memcpy(limbs_le, R, sizeof(FieldLimb) * LIMBS);
        }
        else
        {
            FieldLimb t[LIMBS] = {v};
            Field::_mul(limbs_le, t, R);
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
        _mul((FieldLimb *)&out->limbs_le, a->limbs_le, b->limbs_le);
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
        Field tmp;
        _mul((FieldLimb *)&tmp.limbs_le, a->limbs_le, b->limbs_le);
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
            Field tmp;
            _mul((FieldLimb *)&tmp.limbs_le, a->limbs_le, R2);
            *a = tmp;
        }
        else
        {
            assert(0);
        }
    }

    __device__ static Field sqr(const Field *a)
    {
        return mul(a, a);
    }

    __forceinline__ __device__ static Field one()
    {
        Field tmp;
        memcpy(tmp.limbs_le, R, LIMBS * sizeof(FieldLimb));
        return tmp;
    }

    __forceinline__ __device__ static Field zero()
    {
        return Field(0);
    }

    __device__ static Field pow(const Field *a, ulong exp)
    {
        Field acc = Field::one();
        if (exp > 0)
        {
            _pow_at_leading(&acc, a, exp);
        }
        return acc;
    }

    __device__ static Field pow(const Field *a, const FieldLimb *exp_le, int exp_len)
    {
        Field acc = Field::one();
        int i = exp_len - 1;
        for (; i >= 0; i--)
        {
            if (exp_le[i] != 0)
            {
                _pow_at_leading(&acc, a, exp_le[i]);
                i--;
                break;
            }
        }
        for (; i >= 0; i--)
        {
            _pow_at_nonleading(&acc, a, exp_le[i]);
        }
        return acc;
    }

    __device__ static Field inv(const Field *a)
    {
        return pow(a, NEG_TOW, LIMBS);
    }
};
#endif