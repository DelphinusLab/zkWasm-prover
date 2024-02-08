#ifndef EC_CUH
#define EC_CUH

template <class F>
class CurveAffine
{
public:
    F x;
    F y;

    CurveAffine(F &_x, F &_y)
    {
        x = _x;
        y = _y;
    }

    __device__ CurveAffine identity()
    {
        CurveAffine p;
        p.x = F(0);
        p.y = F(0);
    }
};

template <class F>
class Curve
{
public:
    F x;
    F y;
    F z;

    __device__ Curve(F &_x, F &_y, F &_z)
    {
        x = _x;
        y = _y;
        z = _z;
    }

    __device__ Curve(CurveAffine<F> p)
    {
        x = p.x;
        y = p.y;
        z = F(1);
    }

    __device__ Curve identity()
    {
        Curve p;
        p.x = F(0);
        p.y = F(0);
        p.z = F(1);
        return p;
    }

    __device__ CurveAffine<F> to_affine() const
    {
        F z_inv = this->z.inv();
        CurveAffine p(this->x * z_inv, this->y * z_inv);
        return p;
    }

    __device__ bool is_identity() const
    {
        return x.is_zero() && y.is_zero();
    }

    __device__ Curve<F> ec_double() const
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

    __device__ Curve<F> ec_add(const Curve<F> &rhs) const
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

    __device__ Curve<F> ec_add(CurveAffine<F> &rhs)
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
                F _z = h * rhs.z * this->z;
                return Curve(_x, _y, _z);
            }
        }
    }

    // operator
    __device__ Curve operator+(const Curve<F> &b)
    {
        return this->ec_add(b);
    }

    __device__ Curve operator+(const CurveAffine<F> &b)
    {
        return this->ec_add(b);
    }
};

#endif