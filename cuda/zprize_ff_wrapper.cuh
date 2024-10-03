#pragma once

#include "zprize/ff_dispatch_st.cuh"

template <class FD, typename Storage>
class Field
{
private:
    Storage value;
    __device__ __host__ Field(Storage v) : value(v) {}

public:
    __device__ __host__ Field(uint v)
    {
        if (v == 0)
        {
            this->value = {0};
        }
        else
        {
            assert(v == 1);
            this->value = FD::get_one();
        }
    }

    __device__ __host__ Field()
    {
        this->value = {0};
    }

    __device__ Field unmont_assign()
    {
        this->value = FD::from_montgomery(this->value);
    }

    __device__ Field inv() const
    {
        return FD::inverse(this->value);
    }

    __device__ uint nonzero_bytes() const
    {
        uint i = 32;
        while (i > 0 && FD::extract_bits(this->value, i * 8 - 8, 8) == 0)
        {
            i--;
        }
        return i;
    }

    __device__ ulong get_8bits(uint i) const
    {
        return FD::extract_bits(this->value, i * 8, 8);
    }

    __device__ ulong get_nbits(uint start, uint size) const
    {
        return FD::extract_bits(this->value, start, size);
    }

    __device__ ulong get_32bits(uint i) const
    {
        return FD::extract_bits(this->value, i * 32, 32);
    }

    __device__ Field sqr() const
    {
        return FD::sqr(this->value);
    }

    __device__ bool is_zero() const
    {
        return FD::is_zero(this->value);
    }

    __device__ static void _pow_at_leading(Field *acc, const Field *base, ulong exp)
    {
        Field t = *base;

        while (exp > 0)
        {
            if (exp & 1)
            {
                *acc = *acc * t;
            }

            exp >>= 1;

            if (exp > 0)
            {
                t = t.sqr();
            }
        }
    }

    __device__ static Field pow(const Field *a, ulong exp)
    {
        Field acc = Field(1);
        if (exp > 0)
        {
            _pow_at_leading(&acc, a, exp);
        }
        return acc;
    }

    __device__ Field operator+(const Field &b) const
    {
        return FD::add(this->value, b.value);
    }

    __device__ Field operator-(const Field &b) const
    {
        return FD::sub(this->value, b.value);
    }

    __device__ void operator+=(const Field &b)
    {
        this->value = FD::add(this->value, b.value);
    }

    __device__ void operator-=(const Field &b)
    {
        this->value = FD::sub(this->value, b.value);
    }

    __device__ Field operator*(const Field &b) const
    {
        return FD::mul(this->value, b.value);
    }

    __device__ void operator*=(const Field &b)
    {
        this->value = FD::mul(this->value, b.value);
    }

    __device__ bool operator==(const Field &b) const
    {
        return FD::eq(this->value, b.value);
    }

    __device__ Field operator-() const
    {
        return FD::neg(this->value);
    }
};

typedef Field<fd_q, fd_q::storage> Bn254FrField;
typedef Field<fd_p, fd_p::storage> Bn254FpField;
