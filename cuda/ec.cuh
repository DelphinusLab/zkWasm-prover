#ifndef EC_CUH
#define EC_CUH

template <class F>
class Curve;

template <class F>
class CurveAffine
{
public:
    F x;
    F y;

    __device__ CurveAffine() {}

    __device__ CurveAffine(const F &_x, const F &_y)
    {
        x = _x;
        y = _y;
    }

    __device__ static CurveAffine identity()
    {
        CurveAffine p(F(0), F(0));
        return p;
    }

    __device__ bool is_identity() const
    {
        return x.is_zero() && y.is_zero();
    }

    __device__ bool eq(const CurveAffine &rhs) const
    {
        return this->x == rhs.x && this->y == rhs.y;
    }

    __device__ bool operator==(const CurveAffine &rhs) const
    {
        return this->eq(rhs);
    }

    __device__ bool operator==(const Curve<F> &rhs) const
    {
        return this->eq(rhs.to_affine());
    }

    __device__ void operator=(const Curve<F> &rhs)
    {
        *this = rhs.to_affine();
    }

    __device__ CurveAffine ec_neg() const
    {
        return CurveAffine(this->x, -this->y);
    }
};

template <class F>
class Curve
{
public:
    F x;
    F y;
    F z;

    __device__ Curve(const F &_x, const F &_y, const F &_z)
    {
        x = _x;
        y = _y;
        z = _z;
    }

    __device__ Curve(const CurveAffine<F> &p)
    {
        x = p.x;
        y = p.y;
        z = F(1);
    }

    __device__ static Curve identity()
    {
        return Curve(F(0), F(0), F(1));
    }

    __device__ CurveAffine<F> to_affine() const
    {
        F zi = this->z.inv();
        F zizi = zi * zi;

        CurveAffine p(this->x * zizi, this->y * zizi * zi);
        return p;
    }

    __device__ bool is_identity() const
    {
        return x.is_zero() && y.is_zero();
    }

    __device__ Curve ec_double() const
    {
        F x2 = x.sqr();
        F y2 = y.sqr();
        F x4 = x2.sqr();
        F xy2 = x * y2;

        F _x = x4 - xy2;
        _x += _x;
        _x += _x;
        _x += _x;
        _x += x4;

        F t = y2 * y2; //  y4
        t += t;        // 2y4
        t += t;        // 4y4

        F _y = xy2;
        _y += _y; // 2xy2
        _y += _y; // 4xy2
        _y -= _x; // 4xy2 - _x
        _y *= x2; // x2(4xy2 - _x)
        F u = _y; // x2(4xy2 - _x)
        _y -= t;  // x2(4xy2 - _x) - 4y4
        _y += _y; // 2x2(4xy2 - _x) - 8y4
        _y += u;  // 3x2(4xy2 - _x) - 8y4

        F _z = y * z;
        _z += _z;

        return Curve(_x, _y, _z);
    }

    __device__ Curve ec_add(const Curve &rhs) const
    {
        if (rhs.is_identity())
        {
            return *this;
        }
        else if (this->is_identity())
        {
            return rhs;
        }
        else
        {
            F z1z1 = this->z * this->z;
            F z2z2 = rhs.z * rhs.z;
            F u1 = z2z2 * this->x;
            F u2 = z1z1 * rhs.x;
            F s1 = this->y * rhs.z * z2z2;
            F s2 = rhs.y * this->z * z1z1;

            if (u1 == u2)
            {
                if (s1 == s2)
                {
                    return this->ec_double();
                }
                else
                {
                    return identity();
                }
            }
            else
            {
                F h = u2 - u1;
                F r = s2 - s1;
                F hh = h * h;
                F hhh = hh * h;
                F u1hh = u1 * hh;

                F _x = u1hh + u1hh; // 2u1hh
                _x += hhh;          // hhh + 2u1hh
                _x = r * r - _x;    // rr - hhh - 2u1hh

                F _y = r * (u1hh - _x) - s1 * hhh;
                F _z = h * rhs.z * this->z;
                return Curve(_x, _y, _z);
            }
        }
    }

    __device__ Curve ec_add(const CurveAffine<F> &rhs) const
    {
        if (rhs.is_identity())
        {
            return *this;
        }
        else if (this->is_identity())
        {
            return rhs;
        }
        else
        {
            F z1z1 = this->z * this->z;
            F u1 = this->x;
            F u2 = z1z1 * rhs.x;
            F s1 = this->y;
            F s2 = rhs.y * this->z * z1z1;

            if (u1 == u2)
            {
                if (s1 == s2)
                {
                    return this->ec_double();
                }
                else
                {
                    return identity();
                }
            }
            else
            {
                F h = u2 - u1;
                F r = s2 - s1;
                F hh = h * h;
                F hhh = hh * h;
                F u1hh = u1 * hh;

                F _x = u1hh + u1hh; // 2u1hh
                _x += hhh;          // hhh + 2u1hh
                _x = r * r - _x;    // rr - hhh - 2u1hh

                F _y = r * (u1hh - _x) - s1 * hhh;
                F _z = h * this->z;
                return Curve(_x, _y, _z);
            }
        }
    }

    __device__ Curve ec_neg() const
    {
        return Curve(this->x, -this->y, this->z);
    }

    // operator
    __device__ Curve operator+(const Curve &b)
    {
        return this->ec_add(b);
    }

    __device__ Curve operator+(const CurveAffine<F> &b)
    {
        return this->ec_add(b);
    }

    __device__ Curve operator-(const Curve &b)
    {
        return this->ec_add(b.ec_neg());
    }

    __device__ Curve operator-(const CurveAffine<F> &b)
    {
        return this->ec_add(b.ec_neg());
    }

    __device__ Curve operator-()
    {
        return this->ec_neg();
    }

    __device__ bool operator==(const Curve &b) const
    {
        return this->to_affine() == b.to_affine();
    }
};

#endif