#pragma once

#include "zprize/ec.cuh"
#include "zprize_ff_wrapper.cuh"

template <class EC, class FD, class Point>
class CurveAffine
{
public:
    Point value;

    __device__ bool is_on_curve() const
    {
        FD fd;
        return Point::is_on_curve(this->value, fd);
    }
};

template <class EC, class FD, class Point, class Affine>
class Curve
{
public:
    Point value;
    __device__ Curve()
    {
        FD fd;
        this->value = Point::point_at_infinity(fd);
    }

    __device__ Curve(Point v) : value(v) {}

    __device__ Curve ec_double() const
    {
        FD fd;
        return EC::dbl(this->value, fd);
    }

    __device__ bool is_on_curve() const
    {
        FD fd;
        return Point::is_on_curve(this->value, fd);
    }

    // operator
    __device__ Curve operator+(const Curve &b)
    {
        FD fd;
        return EC::add(this->value, b.value, fd);
    }

    // operator
    __device__ void operator+=(const Curve &b)
    {
        FD fd;
        this->value = EC::add(this->value, b.value, fd);
    }

    __device__ Curve operator+(const Affine &b)
    {
        FD fd;
        Curve res = EC::add(this->value, b.value, fd);
        return res;
    }

    __device__ static Curve identity()
    {
        FD fd;
        return Point::point_at_infinity(fd);
    }
};

typedef CurveAffine<ec<fd_p>, fd_p, ec<fd_p>::point_affine> Bn254G1Affine;
typedef Curve<ec<fd_p>, fd_p, ec<fd_p>::point_jacobian, Bn254G1Affine> Bn254G1;
