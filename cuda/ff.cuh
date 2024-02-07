#ifndef FF_CUH
#define FF_CUH

#include <assert.h>
#include "common.cuh"

typedef ulong FieldLimb;

template <const uint LIMBS, const FieldLimb FIELD_P[LIMBS]>
class Field
{
private:
    // little-endian
    FieldLimb limbs_le[LIMBS];

public:
    Field(/* args */) {}
    ~Field() {}

    __device__ static bool gte(const Field *a, const Field *b)
    {
#pragma unroll
        for (int i = LIMBS - 1; i >= 0; i--)
        {
            if (a->limbs_le[i] > b->limbs_le[i])
                return true;
            if (a->limbs_le[i] < b->limbs_le[i])
                return false;
        }
        return true;
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
            if (gte(c, (Field *)FIELD_P)) {
                sub_u64x4((ulong *)c, (ulong *)FIELD_P, (ulong *)c);
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
            if (borrow) {
                add_u64x4((ulong *)c, (ulong *)FIELD_P, (ulong *)c);
            }
        }
        else
        {
            assert(0);
        }
    }
};
#endif