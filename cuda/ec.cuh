#ifndef EC_CUH
#define EC_CUH

template<class F>
class CurveAffine {
public:
    F x;
    F y;

    CurveAffine(F _x, F _y) {
        x = _x;
        y = _y;
    }
};

template<class F>
class Curve {
public:
    F x;
    F y;
    F z;

    __device__ Curve(CurveAffine<F> p) {
        x = p.x;
        y = p.y;
        z = F(1);
    }

    __device__ CurveAffine<F> to_affine() {
        F z_inv = this->z.inv();
        //return CurveAffine(this.x * z_inv, this.y * z_inv);
    }
};

#endif