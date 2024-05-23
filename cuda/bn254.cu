#include "cuda_runtime.h"
#include <stdio.h>
#include <assert.h>

#include "bn254.cuh"

__global__ void _eval_lookup_z_step1(
    Bn254FrField *z,
    Bn254FrField *permuted_input,
    Bn254FrField *permuted_table,
    Bn254FrField *beta_gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    z[i] = (permuted_input[i] + beta_gamma[0]) * (permuted_table[i] + beta_gamma[1]);
}

__global__ void _eval_lookup_z_batch_invert(
    Bn254FrField *z,
    Bn254FrField *tmp,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bn254FrField t(1);
    Bn254FrField u(1);
    for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++) {
        u = t * z[j];
        tmp[j] = t;
        t = u;
    }

    t = t.inv();

    for (int j = i * size_per_worker + size_per_worker - 1; j >= i * size_per_worker; j--) {
        u = z[j];
        z[j] = t * tmp[j];
        t = t * u;
        tmp[j] = u;
    }
}

__global__ void _eval_lookup_z_step2(
    Bn254FrField *z,
    Bn254FrField *input,
    Bn254FrField *table,
    Bn254FrField *beta_gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    z[i] = z[i] * (input[i] + beta_gamma[0]) * (table[i] + beta_gamma[1]);
}

__global__ void _eval_lookup_z_product_batch(
    Bn254FrField *z,
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) {
        i--;
        Bn254FrField t(1);
        for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++) {
            t *= z[j];
        }
        res[i + 1] = t;
    } else {
        res[i] = Bn254FrField(1);
    }
}

__global__ void _eval_lookup_z_product_single_spread(
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 1; i < size_per_worker; i++) {
        res[i] *= res[i - 1];
    }
}

__global__ void _eval_lookup_z_product_batch_spread(
    Bn254FrField *z,
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    z[i * size_per_worker] *= res[i];
    for (int j = i * size_per_worker + 1; j < i * size_per_worker + size_per_worker; j++) {
        z[j] *= z[j - 1];
    }
}

// Place a Bn254FrField::one() in the front
__global__ void _eval_lookup_z_product_batch_spread_skip(
    Bn254FrField *z,
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bn254FrField t = res[i];
    Bn254FrField u;
    for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++) {
        u = z[j] * t;
        z[j] = t;
        t = u;
    }
}

__global__ void _poly_eval(
    Bn254FrField *p,
    Bn254FrField *out,
    const Bn254FrField *x,
    int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = p[i * 2] + p[i * 2 + 1] * x[deg];
}

__global__ void _msm_mont_unmont(
    Bn254G1Affine *p,
    Bn254FrField *s,
    bool mont,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        if (mont)
        {
            s[i].mont_assign();
        }
        else
        {
            s[i].unmont_assign();
        }
    }
}

__global__ void _msm_core(
    Bn254G1 *res,
    const Bn254G1Affine *p,
    Bn254FrField *s,
    int n)
{
    int group_idx = blockIdx.x;
    int worker = blockDim.x * gridDim.y;
    int size_per_worker = (n + worker - 1) / worker;
    int inner_idx = threadIdx.x;
    int window_idx = inner_idx + blockIdx.y * blockDim.x;
    int start = window_idx * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    __shared__ Bn254G1 thread_res[128];

    Bn254G1 buckets[256];

    for (int i = start; i < end; i++)
    {
        int v = s[i].get_8bits(group_idx);
        if (v--)
        {
            buckets[v] = buckets[v] + p[i];
        }
    }

    if (end > start)
    {
        Bn254G1 round;
        Bn254G1 acc;
        for (int i = 254; i >= 0; i--)
        {
            round = round + buckets[i];
            acc = acc + round;
        }

        thread_res[inner_idx] = acc;
    }

    __syncthreads();
    if (inner_idx == 0)
    {
        Bn254G1 acc;
        for (int i = 0; i < blockDim.x; i++)
        {
            acc = acc + thread_res[i];
        }
        res[group_idx + blockIdx.y * gridDim.x] = acc;
    }
}

__device__ uint bit_reverse(uint n, uint bits)
{
    uint r = 0;
    for (int i = 0; i < bits; i++)
    {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

__device__ Bn254FrField pow_lookup(const Bn254FrField *bases, uint exponent)
{
    Bn254FrField res(1);
    uint i = 0;
    while (exponent > 0)
    {
        if (exponent & 1)
            res = res * bases[i];
        exponent = exponent >> 1;
        i++;
    }
    return res;
}

// Learn from ec-gpu
__global__ void _ntt_core(
    const Bn254FrField *_x,
    Bn254FrField *_y,
    const Bn254FrField *pq,
    const Bn254FrField *omegas,
    uint n,     // Number of elements
    uint log_p, // Log2 of `p` (Read more in the link above)
    uint deg,   // 1=>radix2, 2=>radix4, 3=>radix8, ...
    uint max_deg,
    uint grids) // Maximum degree supported, according to `pq` and `omegas`
{
    uint lid = threadIdx.x;
    uint lsize = blockDim.x;
    uint t = n >> deg;
    uint p = 1 << log_p;

    uint count = 1 << deg;
    uint counth = count >> 1;
    uint counts = count / lsize * lid;
    uint counte = counts + count / lsize;

    const uint pqshift = max_deg - deg;

    for (uint gridIdx = 0; gridIdx < grids; gridIdx++)
    {
        uint index = blockIdx.x + gridIdx * gridDim.x;
        uint k = index & (p - 1);

        const Bn254FrField *x = _x + index;
        Bn254FrField *y = _y + ((index - k) << deg) + k;

        __shared__ Bn254FrField u[512];
        uint base_exp = (n >> log_p >> deg) * k;
        for (uint i = counts; i < counte; i++)
        {
            u[i] = omegas[base_exp * i] * x[i * t];
        }
        __syncthreads();

        for (uint rnd = 0; rnd < deg; rnd++)
        {
            const uint bit = counth >> rnd;
            for (uint i = counts >> 1; i < counte >> 1; i++)
            {
                const uint di = i & (bit - 1);
                const uint i0 = (i << 1) - di;
                const uint i1 = i0 + bit;
                Bn254FrField tmp = u[i0];
                u[i0] += u[i1];
                u[i1] = tmp - u[i1];

                if (di != 0)
                    u[i1] = pq[di << rnd << pqshift] * u[i1];
            }

            __syncthreads();
        }

        for (uint i = counts >> 1; i < counte >> 1; i++)
        {
            y[i * p] = u[bit_reverse(i, deg)];
            y[(i + counth) * p] = u[bit_reverse(i + counth, deg)];
        }
    }
}

__global__ void _field_sum(
    Bn254FrField *res,
    Bn254FrField **v,
    Bn254FrField **v_c,
    int *v_rot,
    Bn254FrField *omegas,
    int v_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        Bn254FrField fl(0), fr;
        for (int j = 0; j < v_n; j++)
        {
            int v_i = i;

            int omega_exp = ((n + v_rot[j]) * i) & (n - 1);

            fr = v[j][v_i] * omegas[omega_exp];

            if (v_c[j])
            {
                fr = fr * *v_c[j];
            }

            if (j == 0)
            {
                fl = fr;
            }
            else
            {
                fl += fr;
            }
        }

        res[i] = fl;
    }
}

__global__ void _field_op_batch_mul_sum(
    Bn254FrField *res,
    Bn254FrField **v, // coeff0, a00, a01, null, coeff1, a10, a11, null,
    int *rot,
    int n_v,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;

    Bn254FrField fl(0), fr;
    int v_idx = 0;
    int rot_idx = 0;
    while (v_idx < n_v)
    {
        fr = *v[v_idx++]; // first one is coeff
        while (v[v_idx])
        {
            int idx;
            idx = (n + i + rot[rot_idx]) & (n - 1);
            fr = fr * v[v_idx][idx];
            v_idx++;
            rot_idx++;
        }

        fl += fr;
        v_idx++;
    }

    res[i] += fl;
}

__global__ void _field_mul_unaligned(
    Bn254FrField *l,
    Bn254FrField *r,
    int r_n,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    l[i] = l[i] * r[i % r_n];
}

__global__ void _field_op(
    Bn254FrField *res,
    Bn254FrField *l,
    int l_rot,
    Bn254FrField *l_c,
    Bn254FrField *r,
    int r_rot,
    Bn254FrField *r_c,
    int n,
    int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bn254FrField fl, fr;

    if (l)
        if (l_c)
            fl = l[(i + l_rot) % n] * l_c[0];
        else
            fl = l[(i + l_rot) % n];
    else
        fl = l_c[0];

    if (r)
        if (r_c)
            fr = r[(i + r_rot) % n] * r_c[0];
        else
            fr = r[(i + r_rot) % n];
    else
        if (r_c)
            fr = r_c[0];

    // add
    if (op == 0)
    {
        res[i] = fl + fr;
    }
    // mul
    else if (op == 1)
    {
        res[i] = fl * fr;
    }
    // uop
    else if (op == 2)
    {
        res[i] = fl;
    }
    // sub
    else if (op == 3)
    {
        res[i] = fl - fr;
    }
    else
    {
        assert(0);
    }
}

__global__ void _extended_prepare(
    Bn254FrField *s,
    Bn254FrField *coset_powers,
    uint coset_powers_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        int index = i % coset_powers_n;
        if (index != 0)
        {
            s[i] = s[i] * coset_powers[index - 1];
        }
    }
}

__global__ void _permutation_eval_h_p1(
    Bn254FrField *res,
    const Bn254FrField *first_set,
    const Bn254FrField *last_set,
    const Bn254FrField *l0,
    const Bn254FrField *l_last,
    const Bn254FrField *y,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    Bn254FrField t1, t2;

    for (int i = start; i < end; i++)
    {
        t1 = res[i];

        // l_0(X) * (1 - z_0(X)) = 0
        t1 = t1 * y[0];
        t2 = Bn254FrField(1);
        t2 -= first_set[i];
        t2 = t2 * l0[i];
        t1 += t2;

        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
        t1 = t1 * y[0];
        t2 = last_set[i].sqr();
        t2 -= last_set[i];
        t2 = t2 * l_last[i];
        t1 += t2;

        res[i] = t1;
    }
}

__global__ void _permutation_eval_h_p2(
    Bn254FrField *res,
    const Bn254FrField **set,
    const Bn254FrField *l0,
    const Bn254FrField *l_last,
    const Bn254FrField *y,
    int n_set,
    int rot,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    Bn254FrField t1, t2;

    for (int i = start; i < end; i++)
    {
        int r_prev = (i + n + rot) & (n - 1);
        t1 = res[i];

        for (int j = 1; j < n_set; j++)
        {
            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
            t1 = t1 * y[0];
            t2 = set[j][i] - set[j - 1][r_prev];
            t2 = t2 * l0[i];
            t1 += t2;
        }

        res[i] = t1;
    }
}

__global__ void _permutation_eval_h_l(
    Bn254FrField *res,
    const Bn254FrField *beta,
    const Bn254FrField *gamma,
    const Bn254FrField *p,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        Bn254FrField t = p[i];
        t = t * beta[0];
        if (i == 0)
        {
            t += gamma[0];
        }
        res[i] += t;
    }
}

__global__ void _permutation_eval_h_r(
    Bn254FrField *res,
    const Bn254FrField *delta,
    const Bn254FrField *gamma,
    const Bn254FrField *value,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = (n + worker - 1) / worker;
    int start = gid * size_per_worker;
    int end = start + size_per_worker;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        Bn254FrField t = value[i];
        if (i == 0)
        {
            t += gamma[0];
        }

        if (i == 1)
        {
            t += delta[0];
        }

        res[i] = t;
    }
}

__global__ void _lookup_eval_h(
    Bn254FrField *res,
    const Bn254FrField *input,
    const Bn254FrField *table,
    const Bn254FrField *permuted_input,
    const Bn254FrField *permuted_table,
    const Bn254FrField *z,
    const Bn254FrField *l0,
    const Bn254FrField *l_last,
    const Bn254FrField *l_active_row,
    const Bn254FrField *y,
    const Bn254FrField *beta,
    const Bn254FrField *gamma,
    int rot,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;
    int r_next = (i + rot) & (n - 1);
    int r_prev = (i + n - rot) & (n - 1);

    Bn254FrField t, u, p;
    t = res[i];

    // l_0(X) * (1 - z(X)) = 0
    t = t * *y;
    u = Bn254FrField(1) - z[i];
    u = l0[i] * u;
    t += u;

    // l_last(X) * (z(X)^2 - z(X)) = 0
    t = t * *y;
    u = z[i] * z[i];
    u -= z[i];
    u = l_last[i] * u;
    t += u;

    // (1 - (l_last(X) + l_blind(X))) * (
    //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
    //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
    //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
    // ) = 0
    t = t * *y;
    u = permuted_input[i] + *beta;
    p = permuted_table[i] + *gamma;
    u = u * p;
    u = u * z[r_next];
    Bn254FrField x = input[i] * table[i];
    u -= z[i] * x;
    u = u * l_active_row[i];
    t += u;

    // l_0(X) * (a'(X) - s'(X)) = 0
    t = t * *y;
    p = permuted_input[i] - permuted_table[i];
    u = l0[i] * p;
    t += u;

    // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
    t = t * *y;
    u = permuted_input[i] - permuted_input[r_prev];
    u = u * p;
    u = u * l_active_row[i];
    t += u;

    res[i] = t;
}


__global__ void _shuffle_eval_h(
    Bn254FrField *res,
    const Bn254FrField *input,
    const Bn254FrField *table,
    const Bn254FrField *z,
    const Bn254FrField *l0,
    const Bn254FrField *l_last,
    const Bn254FrField *l_active_row,
    const Bn254FrField *y,
    int rot,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;
    int r_next = (i + rot) & (n - 1);
    int r_prev = (i + n - rot) & (n - 1);

    Bn254FrField t, u, p;
    t = res[i];

    // l_0(X) * (1 - z(X)) = 0
    t = t * *y;
    u = Bn254FrField(1) - z[i];
    u = l0[i] * u;
    t += u;

    // l_last(X) * (z(X)^2 - z(X)) = 0
    t = t * *y;
    u = z[i] * z[i];
    u -= z[i];
    u = l_last[i] * u;
    t += u;

    // (1 - (l_last(X) + l_blind(X))) *
    // (z(\omega X) (s(X) + \beta^i)- z(X) (a(X) + \beta^i))=0
    t = t * *y;
    u = table[i] * z[r_next];
    u -= input[i] * z[i];
    u = u * l_active_row[i];
    t += u;

    res[i] = t;
}


__global__ void _expand_omega_buffer(
    Bn254FrField *buf,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int workers = gridDim.x * blockDim.x;
    int tasks = n / workers;
    int start = gid * tasks;
    int end = start + tasks;

    start = start < 2 ? 2 : start;
    end = end > n ? n : end;

    Bn254FrField x = buf[1];
    Bn254FrField curr = Bn254FrField::pow(&x, start);

    for (int i = start; i < end; i++)
    {
        buf[i] = curr;
        curr = curr * x;
    }
}

__global__ void _field_mul_zip(
    Bn254FrField *buf,
    Bn254FrField *coeff,
    int coeff_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;

    buf[i] = buf[i] * coeff[i % coeff_n];
}

__global__ void _shplonk_h_x_merge(
    Bn254FrField *res,
    Bn254FrField *v,
    Bn254FrField *values,
    Bn254FrField *omegas,
    Bn254FrField *diff_points,
    int diff_points_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;

    Bn254FrField t = values[i];
    for (int j = 0; j < diff_points_n; j++) {
        t = t * (omegas[i] - diff_points[j]);
    }
    res[i] = res[i] * v[0] + t;
}

__global__ void _shplonk_h_x_div_points(
    Bn254FrField *values,
    Bn254FrField *omegas,
    Bn254FrField *points,
    int points_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid;

    Bn254FrField t = omegas[i] - points[0];
    for (int j = 1; j < points_n; j++) {
        t = t * (omegas[i] - points[j]);
    }
    assert(!(t.inv() == Bn254FrField(0)));
    values[i] = values[i] * t.inv();
}

extern "C"
{
    cudaError_t field_sum(
        Bn254FrField *res,
        Bn254FrField **v,
        Bn254FrField **v_c,
        int *v_rot,
        Bn254FrField *omegas,
        int v_n,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _field_sum<<<blocks, threads>>>(res, v, v_c, v_rot, omegas, v_n, n);
        return cudaGetLastError();
    }

    cudaError_t extended_prepare(
        Bn254FrField *s,
        Bn254FrField *coset_powers,
        uint coset_powers_n,
        int size,
        int extended_size,
        int to_coset,
        CUstream_st *stream)
    {
        int threads = size >= 64 ? 64 : 1;
        int blocks = size / threads;
        if (to_coset)
        {
            _extended_prepare<<<blocks, threads, 0, stream>>>(s, coset_powers, coset_powers_n, extended_size);
        }
        else
        {
            cudaMemsetAsync(&s[size], 0, (extended_size - size) * sizeof(Bn254FrField), stream);
            _extended_prepare<<<blocks, threads, 0, stream>>>(s, coset_powers, coset_powers_n, size);
        }
        return cudaGetLastError();
    }

    cudaError_t field_op_batch_mul_sum(
        Bn254FrField *res,
        Bn254FrField **v, // coeff0, a00, a01, null, coeff1, a10, a11, null,
        int *rot,
        int n_v,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _field_op_batch_mul_sum<<<blocks, threads>>>(res, v, rot, n_v, n);
        return cudaGetLastError();
    }

    cudaError_t field_op(
        Bn254FrField *res,
        Bn254FrField *l,
        int l_rot,
        Bn254FrField *l_c,
        Bn254FrField *r,
        int r_rot,
        Bn254FrField *r_c,
        int n,
        int op,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        assert(threads * blocks == n);
        _field_op<<<blocks, threads, 0, stream>>>(res, l, l_rot, l_c, r, r_rot, r_c, n, op);
        return cudaGetLastError();
    }

    cudaError_t permutation_eval_h_p1(
        Bn254FrField *res,
        const Bn254FrField *first_set,
        const Bn254FrField *last_set,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *y,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _permutation_eval_h_p1<<<blocks, threads>>>(res, first_set, last_set, l0, l_last, y, n);
        return cudaGetLastError();
    }

    cudaError_t permutation_eval_h_p2(
        Bn254FrField *res,
        const Bn254FrField **set,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *y,
        int n_set,
        int rot,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _permutation_eval_h_p2<<<blocks, threads>>>(res, set, l0, l_last, y, n_set, rot, n);
        return cudaGetLastError();
    }

    cudaError_t permutation_eval_h_l(
        Bn254FrField *res,
        const Bn254FrField *beta,
        const Bn254FrField *gamma,
        const Bn254FrField *p,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _permutation_eval_h_l<<<blocks, threads>>>(res, beta, gamma, p, n);
        return cudaGetLastError();
    }

    cudaError_t ntt(
        Bn254FrField *buf,
        Bn254FrField *tmp,
        const Bn254FrField *pq,
        const Bn254FrField *omegas,
        int log_n,
        int max_deg,
        bool *swap,
        CUstream_st *stream)
    {
        int p = 0;

        Bn254FrField *src = buf;
        Bn254FrField *dst = tmp;
        int len = 1 << log_n;
        int total = 1 << (log_n - 1);
        while (p < log_n)
        {
            int res = log_n - p;
            int round = (res + max_deg - 1) / max_deg;
            int deg = (res + round - 1) / round;

            int threads = 1 << (deg - 1);
            int blocks = total >> (deg - 1);
            blocks = blocks > 65536 ? 65536 : blocks;
            int grids = (total / blocks) >> (deg - 1);
            _ntt_core<<<blocks, threads, 0, stream>>>(src, dst, pq, omegas, len, p, deg, max_deg, grids);

            Bn254FrField *t = src;
            src = dst;
            dst = t;
            p += deg;
            *swap = !*swap;
        }
        return cudaGetLastError();
    }

    cudaError_t msm(
        int msm_blocks,
        int max_msm_threads,
        Bn254G1 *res,
        Bn254G1Affine *p,
        Bn254FrField *s,
        int n)
    {
        int threads = n >= max_msm_threads ? max_msm_threads : 1;
        int blocks = (n + threads - 1) / threads;
        _msm_mont_unmont<<<blocks, threads>>>(p, s, false, n);
        _msm_core<<<dim3(32, msm_blocks), threads>>>(res, p, s, n);
        _msm_mont_unmont<<<blocks, threads>>>(p, s, true, n);
        cudaDeviceSynchronize();
        return cudaGetLastError();
    }

    cudaError_t lookup_eval_h(
        Bn254FrField *res,
        const Bn254FrField *input,
        const Bn254FrField *table,
        const Bn254FrField *permuted_input,
        const Bn254FrField *permuted_table,
        const Bn254FrField *z,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *l_active_row,
        const Bn254FrField *y,
        const Bn254FrField *beta,
        const Bn254FrField *gamma,
        int rot,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _lookup_eval_h<<<blocks, threads>>>(
            res,
            input, table, permuted_input, permuted_table, z,
            l0, l_last, l_active_row,
            y, beta, gamma,
            rot, n);
        return cudaGetLastError();
    }

    cudaError_t shuffle_eval_h(
        Bn254FrField *res,
        const Bn254FrField *input,
        const Bn254FrField *table,
        const Bn254FrField *z,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *l_active_row,
        const Bn254FrField *y,
        int rot,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _shuffle_eval_h<<<blocks, threads>>>(
            res,
            input, table, z,
            l0, l_last, l_active_row,
            y, rot, n);
        return cudaGetLastError();
    }


    cudaError_t expand_omega_buffer(
        Bn254FrField *res,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _expand_omega_buffer<<<blocks, threads>>>(res, n);
        return cudaGetLastError();
    }

    cudaError_t field_mul_zip(
        Bn254FrField *buf,
        Bn254FrField *coeff,
        int coeff_n,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _field_mul_zip<<<blocks, threads>>>(buf, coeff, coeff_n, n);
        return cudaGetLastError();
    }

    cudaError_t shplonk_h_x_merge(
        Bn254FrField *res,
        Bn254FrField *v,
        Bn254FrField *values,
        Bn254FrField *omegas,
        Bn254FrField *diff_points,
        int diff_points_n,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _shplonk_h_x_merge<<<blocks, threads>>>(res, v, values, omegas, diff_points, diff_points_n, n);
        return cudaGetLastError();
    }

    cudaError_t shplonk_h_x_div_points(
        Bn254FrField *values,
        Bn254FrField *omegas,
        Bn254FrField *points,
        int points_n,
        int n)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _shplonk_h_x_div_points<<<blocks, threads>>>(values, omegas, points, points_n, n);
        return cudaGetLastError();
    }

    cudaError_t poly_eval(
        Bn254FrField *p,
        Bn254FrField *res,
        Bn254FrField *tmp,
        const Bn254FrField *x,
        int n)
    {
        Bn254FrField *in = p;
        Bn254FrField *out = res;
        int deg = 0;
        while (n > 1)
        {
            int threads = n / 2 >= 64 ? 64 : 1;
            int blocks = n / threads / 2;
            _poly_eval<<<blocks, threads>>>(in, out, x, deg);
            n >>= 1;

            if (n > 1)
            {
                if (deg == 0)
                {
                    in = res;
                    out = tmp;
                }
                else
                {
                    Bn254FrField *t = in;
                    in = out;
                    out = t;
                }
            }
            deg++;
        }

        if (out != res)
        {
            cudaMemcpy(res, out, sizeof(Bn254FrField), cudaMemcpyDeviceToDevice);
        }

        return cudaGetLastError();
    }

    cudaError_t eval_lookup_z(
        Bn254FrField *z,
        Bn254FrField *input,
        Bn254FrField *table,
        Bn254FrField *permuted_input,
        Bn254FrField *permuted_table,
        Bn254FrField *beta_gamma,
        int n,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _eval_lookup_z_step1<<<blocks, threads, 0, stream>>>(
            z, permuted_input, permuted_table, beta_gamma);

        int worker = 64 * 128;
        int size_per_worker = n / worker;
        _eval_lookup_z_batch_invert<<<128, 64, 0, stream>>>(
            z, permuted_input, size_per_worker);

        _eval_lookup_z_step2<<<blocks, threads, 0, stream>>>(
            z, input, table, beta_gamma);

        worker = 64 * 64;
        size_per_worker = n / worker;
        _eval_lookup_z_product_batch<<<64, 64, 0, stream>>>(
            z, permuted_input, size_per_worker);
        _eval_lookup_z_product_batch<<<8, 8, 0, stream>>>(
            permuted_input, permuted_table, 64);
        _eval_lookup_z_product_single_spread<<<1, 1, 0, stream>>>(
            permuted_table, 64);
        _eval_lookup_z_product_batch_spread<<<8, 8, 0, stream>>>(
            permuted_input, permuted_table, 64);
        _eval_lookup_z_product_batch_spread_skip<<<64, 64, 0, stream>>>(
            z, permuted_input, size_per_worker);

        return cudaGetLastError();
    }
}
