#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>
#include <cub/device/device_radix_sort.cuh>

#include "zprize_ff_wrapper.cuh"

#if false
#include "ec.cuh"
typedef CurveAffine<Bn254FpField> Bn254G1Affine;
typedef Curve<Bn254FpField> Bn254G1;
#else
#include "zprize_ec_wrapper.cuh"
#endif

#include "msm.cu"

__global__ void logup_sum_input_inv_kernel(
    Bn254FrField *accu,
    const Bn254FrField *z,
    Bn254FrField *tmp,
    const Bn254FrField *beta_ptr,
    int init,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = i * size_per_worker;
    int end = i * size_per_worker + size_per_worker;

    Bn254FrField beta = beta_ptr[0];
    tmp[start] = z[start] + beta;
    for (int j = start + 1; j < end; j++)
    {
        tmp[j] = tmp[j - 1] * (z[j] + beta);
    }

    Bn254FrField t = tmp[end - 1].inv();

    for (int j = end - 1; j > start; j--)
    {
        if (init == 0)
        {
            accu[j] = t * tmp[j - 1];
        }
        else
        {
            accu[j] += t * tmp[j - 1];
        }
        t = t * (z[j] + beta);
    }

    if (init == 0)
    {
        accu[start] = t;
    }
    else
    {
        accu[start] += t;
    }
}

__global__ void batch_add_beta_invert_kernel(
    Bn254FrField *z,
    Bn254FrField *tmp,
    const Bn254FrField *beta_ptr,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = i * size_per_worker;
    int end = i * size_per_worker + size_per_worker;

    Bn254FrField beta = beta_ptr[0];
    tmp[start] = z[start] + beta;
    for (int j = start + 1; j < end; j++)
    {
        tmp[j] = tmp[j - 1] * (z[j] + beta);
    }

    Bn254FrField t = tmp[end - 1].inv();

    for (int j = end - 1; j > start; j--)
    {
        Bn254FrField u = t * (z[j] + beta);
        z[j] = t * tmp[j - 1];
        t = u;
    }

    z[start] = t;
}

__global__ void batch_invert_kernel(
    Bn254FrField *z,
    Bn254FrField *tmp,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bn254FrField t(1);
    Bn254FrField u(1);
    for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++)
    {
        u = t * z[j];
        tmp[j] = t;
        t = u;
    }

    t = t.inv();

    for (int j = i * size_per_worker + size_per_worker - 1; j >= i * size_per_worker; j--)
    {
        u = z[j];
        z[j] = t * tmp[j];
        t = t * u;
        tmp[j] = u;
    }
}

// logup
__global__ void _eval_logup_z_grand_sum_batch(
    Bn254FrField *z,
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Bn254FrField t(0);
    for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++)
    {
        t += z[j];
        z[j] = t - z[j];
    }

    if (i != gridDim.x * blockDim.x - 1)
    {
        res[i + 1] = t;
    }
    else
    {
        res[0] = Bn254FrField(0);
    }
}
__global__ void _eval_logup_z_grand_sum_batch_init(
    Bn254FrField *z,
    Bn254FrField *res,
    Bn254FrField *init_z,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0)
    {
        i--;
        Bn254FrField t(0);
        for (int j = i * size_per_worker; j < i * size_per_worker + size_per_worker; j++)
        {
            t += z[j];
        }
        res[i + 1] = t;
    }
    else
    {
        res[i] = init_z[0];
    }
}

__global__ void _eval_logup_z_grand_sum_single_spread(
    Bn254FrField *res,
    int size_per_worker)
{
    for (int i = 1; i < size_per_worker; i++)
    {
        res[i] += res[i - 1];
    }
}

__global__ void _eval_logup_z_grand_sum_batch_spread(
    Bn254FrField *z,
    Bn254FrField *res,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    z[i * size_per_worker] += res[i];
    for (int j = i * size_per_worker + 1; j < i * size_per_worker + size_per_worker; j++)
    {
        z[j] += z[j - 1];
    }
}

__global__ void _eval_logup_z_grand_sum_batch_spread_skip(
    Bn254FrField *z,
    Bn254FrField *res,
    Bn254FrField *last_z,
    int last_z_index,
    int size_per_worker)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bn254FrField t = res[i / size_per_worker];
    z[i] += t;
    if (i == last_z_index)
    {
        last_z[0] = z[i];
    }
}

__global__ void _eval_logup_add_c(
    Bn254FrField *input,
    const Bn254FrField *beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    input[i] = input[i] + beta[0];
}

__global__ void _eval_logup_z_step2(
    Bn254FrField *z,
    Bn254FrField *input,
    Bn254FrField *table,
    Bn254FrField *multiplicity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    z[i] = input[i] - table[i] * multiplicity[i];
}

__global__ void _eval_logup_accu_input(
    Bn254FrField *accu,
    Bn254FrField *input,
    int init)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (init == 0)
    {
        accu[i] = input[i];
    }
    else
    {
        accu[i] = accu[i] + input[i];
    }
}

__global__ void _poly_eval(
    Bn254FrField *p,
    Bn254FrField *res,
    const Bn254FrField *x_table,
    int deg,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = (gid * count) << deg;
    int step = 1 << deg;
    int end = start + ((count - 1) << deg);
    Bn254FrField x = x_table[deg];

    Bn254FrField acc = p[end];
    for (int i = end - step; i > start; i -= step)
    {
        acc *= x;
        acc += p[i];
    }
    res[start] = acc * x + p[start];
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

// porting from GPU-NTT https://github.com/Alisah-Ozcan/GPU-NTT/blob/main/src/lib/ntt_merge/ntt.cu
__device__ void CooleyTukeyUnit(Bn254FrField &U, Bn254FrField &V, const Bn254FrField &root)
{
    Bn254FrField v_ = V * root;

    V = U - v_;
    U = U + v_;
}

__global__ void merge_ntt_kernel(const Bn254FrField *polynomial_in,
                                 Bn254FrField *polynomial_out,
                                 const Bn254FrField *root_of_unity_table,
                                 const Bn254FrField *n_inv,
                                 int shared_index,
                                 int logm,
                                 int outer_iteration_count,
                                 int N_power,
                                 bool not_last_kernel)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    extern __shared__ Bn254FrField shared_memory[];

    int t_2 = N_power - logm - 1;
    unsigned offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    unsigned m = (unsigned)1 << logm;

    unsigned global_addresss =
        idx_x +
        (unsigned)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (unsigned)(blockDim.x * block_x) +
        (unsigned)(2 * block_y * offset);

    unsigned omega_addresss =
        idx_x +
        (unsigned)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (unsigned)(blockDim.x * block_x) + (unsigned)(block_y * offset);

    unsigned shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    unsigned current_root_index;
    if (not_last_kernel)
    {
#pragma unroll
        for (int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            current_root_index = (omega_addresss >> t_2);

            CooleyTukeyUnit(shared_memory[in_shared_address],
                            shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
        {
            __syncthreads();
            current_root_index = (omega_addresss >> t_2);

            CooleyTukeyUnit(shared_memory[in_shared_address],
                            shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < 6; lp++)
        {
            current_root_index = (omega_addresss >> t_2);
            CooleyTukeyUnit(shared_memory[in_shared_address],
                            shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }
    }

    if (not_last_kernel)
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
    else if (!n_inv)
    {
        polynomial_out[bit_reverse(global_addresss, N_power)] = shared_memory[shared_addresss];
        polynomial_out[bit_reverse(global_addresss + offset, N_power)] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
    else
    {
        polynomial_out[bit_reverse(global_addresss, N_power)] = shared_memory[shared_addresss] * *n_inv;
        polynomial_out[bit_reverse(global_addresss + offset, N_power)] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] * *n_inv;
    }
}

// Learn from ec-gpu
__global__ void _ntt_core(
    const Bn254FrField *_x,
    Bn254FrField *_y,
    const Bn254FrField *pq,
    const Bn254FrField *omegas,
    const Bn254FrField *n_inv,
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

        extern __shared__ Bn254FrField u[];
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
            if (n >> log_p >> deg == 1 && n_inv)
            {
                y[i * p] = u[bit_reverse(i, deg)] * *n_inv;
                y[(i + counth) * p] = u[bit_reverse(i + counth, deg)] * *n_inv;
            }
            else
            {
                y[i * p] = u[bit_reverse(i, deg)];
                y[(i + counth) * p] = u[bit_reverse(i + counth, deg)];
            }
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
    else if (r_c)
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

// logarithmic derivative lookup
__global__ void _logup_eval_h(
    Bn254FrField *res,
    const Bn254FrField *input_product,
    const Bn254FrField *input_product_sum,
    const Bn254FrField *table,
    const Bn254FrField *multiplicity,
    const Bn254FrField *z_first,
    const Bn254FrField *z_last,
    const Bn254FrField *l0,
    const Bn254FrField *l_last,
    const Bn254FrField *l_active_row,
    const Bn254FrField *y,
    int rot,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r_next = (i + rot) & (n - 1);

    Bn254FrField t, u, p;
    t = res[i];

    // l_0(X) * z_0(X) = 0
    t = t * *y;
    u = l0[i] * z_first[i];
    t += u;

    // l_last(X) * z_l(X) = 0
    t = t * *y;
    u = l_last[i] * z_last[i];
    t += u;

    // (1 - (l_last(X) + l_blind(X))) * (
    //   τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
    //   - ∑_i τ(X) * Π_{j != i} φ_j(X) + m(X) * Π(φ_i(X))
    // ) = 0
    //=>(1 - (l_last(X) + l_blind(X))) * (
    //   (τ(X) * (ϕ(gX) - ϕ(X))+m(X))* Π(φ_i(X))
    //   - ∑_i τ(X) * Π_{j != i} φ_j(X)
    // ) = 0
    t = t * *y;
    u = z_first[r_next] - z_first[i];
    u = u * table[i];
    u = u + multiplicity[i];
    u = u * input_product[i];
    p = table[i] * input_product_sum[i];
    u = u - p;
    u = u * l_active_row[i];
    t += u;

    res[i] = t;
}

// logup extra inputs and z check
__global__ void _logup_eval_h_extra_inputs(
    Bn254FrField *res,
    const Bn254FrField *input_product,
    const Bn254FrField *input_product_sum,
    const Bn254FrField *z,
    const Bn254FrField *l_active_row,
    const Bn254FrField *y,
    int rot,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r_next = (i + rot) & (n - 1);

    Bn254FrField t, u;
    t = res[i];

    // (1 - (l_last(X) + l_blind(X))) * (
    //   Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
    //   - ∑_i Π_{j != i} φ_j(X))
    // ) = 0
    t = t * *y;
    u = z[r_next] - z[i];
    u = u * input_product[i];
    u = u - input_product_sum[i];
    u = u * l_active_row[i];
    t += u;

    res[i] = t;
}

// logup z set check, z is grand_sum
__global__ void _logup_eval_h_z(
    Bn254FrField *res,
    const Bn254FrField *z_curr,
    const Bn254FrField *z_prev,
    const Bn254FrField *l0,
    const Bn254FrField *y,
    int rot,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r_prev = (i + n + rot) & (n - 1);

    Bn254FrField t, u, p;
    t = res[i];

    // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
    t = t * *y;
    u = z_curr[i] - z_prev[r_prev];
    u = l0[i] * u;
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

__global__ void _shuffle_eval_h_v2(
    Bn254FrField *res,
    const Bn254FrField **input,
    const Bn254FrField **table,
    const Bn254FrField *beta,
    const int group_len,
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

    Bn254FrField l, r;
    l = input[0][i] + beta[0];
    r = table[0][i] + beta[0];

    for (int j = 1; j < group_len; j++)
    {
        l = l * (input[j][i] + beta[j]);
        r = r * (table[j][i] + beta[j]);
    }

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
    u = r * z[r_next];
    u -= l * z[i];
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

__global__ void _bitreverse_expand_omega_buffer(
    Bn254FrField *buf,
    int log_n,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int workers = gridDim.x * blockDim.x;
    int tasks = n / workers;
    int start = gid * tasks;
    int end = start + tasks;
    end = end > n ? n : end;

    for (int i = start; i < end; i++)
    {
        // bit reverse
        int bit_reversed_i = bit_reverse(i, log_n);
        if (bit_reversed_i < i)
        {
            Bn254FrField x = buf[i];
            buf[i] = buf[bit_reversed_i];
            buf[bit_reversed_i] = x;
        }
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
    for (int j = 0; j < diff_points_n; j++)
    {
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
    for (int j = 1; j < points_n; j++)
    {
        t = t * (omegas[i] - points[j]);
    }
    assert(!(t.inv() == Bn254FrField(0)));
    values[i] = values[i] * t.inv();
}

__global__ void _logup_eval_h_inputs_product_sum(
    Bn254FrField *product,
    Bn254FrField *product_sum,
    const Bn254FrField **set,
    int n_set,
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
        Bn254FrField p, u, sum;
        // product
        p = Bn254FrField(1);
        for (int j = 0; j < n_set; j++)
        {
            p = p * set[j][i];
        }
        product[i] = p;

        // product_sum
        sum = Bn254FrField(0);
        for (int j = 0; j < n_set; j++)
        {
            u = Bn254FrField(1);
            for (int k = 0; k < n_set; k++)
            {
                if (k != j)
                {
                    u = u * set[k][i];
                }
            }
            sum = sum + u;
        }
        product_sum[i] = sum;
    }
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
        int n,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _field_op_batch_mul_sum<<<blocks, threads, 0, stream>>>(res, v, rot, n_v, n);
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
        int n,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _permutation_eval_h_l<<<blocks, threads, 0, stream>>>(res, beta, gamma, p, n);
        return cudaGetLastError();
    }

    cudaError_t expand_omega_buffer(
        Bn254FrField *res,
        int log_n,
        int may_bit_reversed)
    {
        if (may_bit_reversed && (log_n == 22 || log_n == 24))
        {
            log_n--;
            int n = 1 << log_n;
            int threads = n >= 64 ? 64 : 1;
            int blocks = n / threads;
            _expand_omega_buffer<<<blocks, threads>>>(res, n);
            _bitreverse_expand_omega_buffer<<<blocks, threads>>>(res, log_n, n);
        }
        else
        {
            int n = 1 << log_n;
            int threads = n >= 64 ? 64 : 1;
            int blocks = n / threads;
            _expand_omega_buffer<<<blocks, threads>>>(res, n);
        }
        return cudaGetLastError();
    }

    cudaError_t ntt(
        Bn254FrField *buf,
        Bn254FrField *tmp,
        const Bn254FrField *pq,
        const Bn254FrField *omegas,
        const Bn254FrField *n_inv,
        int log_n,
        int max_deg,
        bool *swap,
        CUstream_st *stream)
    {
        if (log_n == 22)
        {
            merge_ntt_kernel<<<dim3(8192, 1), dim3(8, 32),
                               512 * sizeof(Bn254FrField), stream>>>(
                buf, buf, omegas, n_inv, 8,
                0, 6, log_n, true);
            merge_ntt_kernel<<<dim3(128, 64), dim3(4, 64),
                               512 * sizeof(Bn254FrField), stream>>>(
                buf, buf, omegas, n_inv, 8,
                6, 7, log_n, true);
            merge_ntt_kernel<<<dim3(1, 8192), dim3(256, 1),
                               512 * sizeof(Bn254FrField), stream>>>(
                buf, tmp, omegas, n_inv, 8,
                13, 9, log_n, false);
            *swap = true;
        }
        else if (log_n == 24)
        {
            merge_ntt_kernel<<<dim3(16384, 1), dim3(8, 64),
                               1024 * sizeof(Bn254FrField), stream>>>(
                buf, buf, omegas, n_inv, 9,
                0, 7, log_n, true);
            merge_ntt_kernel<<<dim3(128, 128), dim3(8, 64),
                               1024 * sizeof(Bn254FrField), stream>>>(
                buf, buf, omegas, n_inv, 9,
                7, 7, log_n, true);
            merge_ntt_kernel<<<dim3(1, 16384), dim3(512, 1),
                               1024 * sizeof(Bn254FrField), stream>>>(
                buf, tmp, omegas, n_inv, 9,
                14, 10, log_n, false);
            *swap = true;
        }
        else
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
                _ntt_core<<<blocks, threads, 512 * sizeof(Bn254FrField), stream>>>(src, dst, pq, omegas, n_inv, len, p, deg, max_deg, grids);

                Bn254FrField *t = src;
                src = dst;
                dst = t;
                p += deg;
                *swap = !*swap;
            }
        }
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
        int n,
        cudaStream_t stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _lookup_eval_h<<<blocks, threads, 0, stream>>>(
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

    cudaError_t logup_eval_h(
        Bn254FrField *res,
        const Bn254FrField *input_product,
        const Bn254FrField *input_product_sum,
        const Bn254FrField *table,
        const Bn254FrField *multiplicity,
        const Bn254FrField *z_first,
        const Bn254FrField *z_last,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *l_active_row,
        const Bn254FrField *y,
        int rot,
        int n,
        cudaStream_t stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _logup_eval_h<<<blocks, threads, 0, stream>>>(
            res,
            input_product, input_product_sum, table, multiplicity, z_first, z_last,
            l0, l_last, l_active_row,
            y, rot, n);
        return cudaGetLastError();
    }

    cudaError_t logup_eval_h_extra_inputs(
        Bn254FrField *res,
        const Bn254FrField *input_product,
        const Bn254FrField *input_product_sum,
        const Bn254FrField *z,
        const Bn254FrField *l_active_row,
        const Bn254FrField *y,
        int rot,
        int n,
        cudaStream_t stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _logup_eval_h_extra_inputs<<<blocks, threads, 0, stream>>>(
            res,
            input_product, input_product_sum, z,
            l_active_row, y, rot, n);
        return cudaGetLastError();
    }

    cudaError_t logup_eval_h_z_set(
        Bn254FrField *res,
        const Bn254FrField **set,
        const Bn254FrField *l0,
        const Bn254FrField *l_last,
        const Bn254FrField *y,
        int n_set,
        int rot,
        int n,
        cudaStream_t stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;
        _permutation_eval_h_p2<<<blocks, threads, 0, stream>>>(res, set, l0, l_last, y, n_set, rot, n);
        return cudaGetLastError();
    }

    cudaError_t shuffle_eval_h_v2(
        Bn254FrField *res,
        const Bn254FrField **input,
        const Bn254FrField **table,
        const Bn254FrField *betas,
        const int group_len,
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
        _shuffle_eval_h_v2<<<blocks, threads>>>(
            res,
            input, table, betas, group_len,
            z,
            l0, l_last, l_active_row,
            y, rot, n);
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
        const Bn254FrField *x,
        int n,
        CUstream_st *stream)
    {
        int deg = 0;
        int max_round_deg = 4;
        int remain_deg = __builtin_ctz(n);

        while (n > 1)
        {
            int current_deg = remain_deg < max_round_deg ? remain_deg : max_round_deg;
            int count = n >> current_deg;
            int threads = count >= 64 ? 64 : count;
            int blocks = count / threads;
            assert(blocks * threads == count);
            assert(current_deg > 0);
            _poly_eval<<<blocks, threads, 0, stream>>>(deg == 0 ? p : res, res, x, deg, 1 << current_deg);
            n >>= current_deg;
            deg += current_deg;
            remain_deg -= current_deg;
        }

        return cudaGetLastError();
    }
}

__global__ void init_table_lookup_index_kernel(
    Bn254FrField *table,
    unsigned *table_lowest_u32,
    unsigned *index,
    unsigned k,
    unsigned unusable_rows_start)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    table_lowest_u32[gid] = ((gid >= unusable_rows_start) << (k + 2)) | (*(unsigned *)(&table[gid]) & ((1 << (k + 2)) - 1));
    index[gid] = gid;
}

__global__ void init_table_lookup_start_offset(
    unsigned *sorted_table_lowest_u32,
    unsigned *start_offset,
    unsigned k)
{
    unsigned curr = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned next = curr + 1;
    unsigned curr_bits = sorted_table_lowest_u32[curr];

    bool assign = true;
    if (next < 1 << k)
    {
        unsigned next_bits = sorted_table_lowest_u32[next];
        assign = curr_bits != next_bits;
    }

    if (assign)
    {
        start_offset[curr_bits] = curr;
    }
}

__global__ void find_matched_table_index_kernel(
    Bn254FrField *table,
    Bn254FrField *input,
    const unsigned *sorted_index,
    const unsigned *start_offset,
    unsigned *match_index,
    unsigned k,
    unsigned unusable_rows_start)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (gid < unusable_rows_start)
    {
        for (int i = start_offset[input[gid].get_nbits(0, (k + 2))]; i >= 0; i--)
        {
            unsigned idx = sorted_index[i];
            if (input[gid] == table[idx])
            {
                match_index[gid] = idx;
                return;
            }
        }

        assert(false);
    }
    else
    {
        match_index[gid] = 0xffffffff;
    }
}

__global__ void merge_index_init(
    Bn254FrField *m,
    unsigned *match_index_in,
    unsigned *match_index_out,
    unsigned unusable_rows_start,
    unsigned size)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tpw = (size + workers - 1) / workers;
    unsigned start = tpw * gid;
    unsigned end = start + tpw > size ? size : start + tpw;

    unsigned index = match_index_in[start];
    unsigned count = 1;
    for (int i = start + 1; i < end; i++)
    {
        unsigned new_index = match_index_in[i];
        if (new_index == index)
        {
            count += 1;
        }
        else
        {
            if (index < unusable_rows_start)
            {
                m[index] += Bn254FrField(count);
            }
            index = new_index;
            count = 1;
        }
    }

    match_index_out[gid * 2] = index;
    match_index_out[gid * 2 + 1] = count;
}

__global__ void merge_index(
    Bn254FrField *m,
    unsigned *match_index_in,
    unsigned *match_index_out,
    unsigned unusable_rows_start,
    unsigned size)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tpw = (size + workers - 1) / workers;
    unsigned start = tpw * gid;
    unsigned end = start + tpw > size ? size : start + tpw;

    unsigned index = match_index_in[start * 2];
    unsigned count = match_index_in[start * 2 + 1];
    for (int i = start + 1; i < end; i++)
    {
        unsigned new_index = match_index_in[i * 2];
        unsigned new_count = match_index_in[i * 2 + 1];
        if (new_index == index)
        {
            count += new_count;
        }
        else
        {
            assert(index < new_index);
            if (index < unusable_rows_start)
            {
                m[index] += Bn254FrField(count);
            }
            index = new_index;
            count = new_count;
        }
    }

    if (workers > 1)
    {
        match_index_out[gid * 2] = index;
        match_index_out[gid * 2 + 1] = count;
    }
    else if (index < unusable_rows_start)
    {
        m[index] += Bn254FrField(count);
    }
}

extern "C"
{
#define CHECK_RETURN(x)         \
    {                           \
        cudaError_t err = x;    \
        if (err != cudaSuccess) \
        {                       \
            return err;         \
        }                       \
    }

    cudaError_t calc_m(
        Bn254FrField *m,
        Bn254FrField *table,
        Bn254FrField *input,
        const unsigned *sorted_index,
        const unsigned *start_offset,
        unsigned *matched_index,
        unsigned *sorted_matched_index,
        unsigned *candidate_sort_temp_storage,
        unsigned candidate_sort_temp_storage_bytes,
        unsigned k,
        unsigned unusable_rows_start,
        cudaStream_t stream)
    {
        find_matched_table_index_kernel<<<(1 << (k - 5)), 32, 0, stream>>>(table, input, sorted_index, start_offset, matched_index, k, unusable_rows_start);

        unsigned *sort_indices_temp_storage{};
        size_t sort_indices_temp_storage_bytes = 0;

        CHECK_RETURN(cub::DeviceRadixSort::SortKeys(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, matched_index, sorted_matched_index,
            1 << k, 0, k, stream));

        bool alloc_for_sort = sort_indices_temp_storage_bytes > candidate_sort_temp_storage_bytes;
        if (alloc_for_sort)
        {
            printf("realloc for bytes %lu\n", sort_indices_temp_storage_bytes);
            CHECK_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
        }
        else
        {
            sort_indices_temp_storage = candidate_sort_temp_storage;
        }

        CHECK_RETURN(cub::DeviceRadixSort::SortKeys(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, matched_index, sorted_matched_index,
            1 << k, 0, k, stream));

        if (alloc_for_sort)
        {
            CHECK_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
        }

        unsigned worker_bits = k - 1;
        unsigned blocks = worker_bits > 5 ? (1 << (worker_bits - 5)) : 1;
        unsigned threads = worker_bits > 5 ? 32 : (1 << worker_bits);
        merge_index_init<<<blocks, threads, 0, stream>>>(m, sorted_matched_index, matched_index, unusable_rows_start, 1 << k);

        bool dir = false;
        for (int worker_bits = k - 2; worker_bits >= 0; worker_bits--)
        {
            unsigned blocks = worker_bits > 5 ? (1 << (worker_bits - 5)) : 1;
            unsigned threads = worker_bits > 5 ? 32 : (1 << worker_bits);
            if (dir)
            {
                merge_index<<<blocks, threads, 0, stream>>>(m, sorted_matched_index, matched_index, unusable_rows_start, 1 << (worker_bits + 1));
            }
            else
            {
                merge_index<<<blocks, threads, 0, stream>>>(m, matched_index, sorted_matched_index, unusable_rows_start, 1 << (worker_bits + 1));
            }
            dir = !dir;
        }

        return cudaGetLastError();
    }

    cudaError_t prepare_table_lookup(
        Bn254FrField *table,
        unsigned *table_lowest_u32,
        unsigned *sorted_table_lowest_u32,
        unsigned *index,
        unsigned *sorted_index,
        unsigned *start_offset,
        unsigned *candidate_sort_temp_storage,
        unsigned candidate_sort_temp_storage_bytes,
        unsigned k,
        unsigned unusable_rows_start,
        cudaStream_t stream)
    {
        init_table_lookup_index_kernel<<<(1 << (k - 5)), 32, 0, stream>>>(
            table, table_lowest_u32, index, k, unusable_rows_start);

        unsigned *sort_indices_temp_storage{};
        size_t sort_indices_temp_storage_bytes = 0;

        CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, table_lowest_u32, sorted_table_lowest_u32,
            index, sorted_index, 1 << k, 0, k + 3, stream));

        bool alloc_for_sort = sort_indices_temp_storage_bytes > candidate_sort_temp_storage_bytes;
        if (alloc_for_sort)
        {
            printf("realloc for bytes %lu\n", sort_indices_temp_storage_bytes);
            CHECK_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
        }
        else
        {
            sort_indices_temp_storage = candidate_sort_temp_storage;
        }

        CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, table_lowest_u32, sorted_table_lowest_u32,
            index, sorted_index, 1 << k, 0, k + 3, stream));

        if (alloc_for_sort)
        {
            CHECK_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
        }

        init_table_lookup_start_offset<<<(1 << (k - 5)), 32, 0, stream>>>(
            sorted_table_lowest_u32,
            start_offset,
            k);

        return cudaGetLastError();
    }

    cudaError_t logup_sum_input_inv(
        Bn254FrField *sum,
        Bn254FrField *input,
        Bn254FrField *temp,
        const Bn254FrField *beta,
        int init,
        int n,
        CUstream_st *stream)
    {
        int threads = 16;
        int blocks = 2048;

        int worker = threads * blocks;
        int size_per_worker = n / worker;
        assert(size_per_worker >= 1);

        logup_sum_input_inv_kernel<<<blocks, threads, 0, stream>>>(
            sum, input, temp, beta, init, size_per_worker);

        return cudaGetLastError();
    }

    cudaError_t eval_logup_z_pure(
        Bn254FrField *z,
        Bn254FrField *input,
        Bn254FrField *table,
        Bn254FrField *last_z,
        int last_z_index,
        int n,
        CUstream_st *stream)
    {
        int threads = 16;
        int blocks = 1024;
        int worker = threads * blocks;
        int size_per_worker = n / worker;
        _eval_logup_z_grand_sum_batch<<<blocks, threads, 0, stream>>>(
            z, input, size_per_worker);

        int mid_threads = 8;
        int mid_blocks = 16;
        int mid_size_per_worker = worker / mid_threads / mid_blocks;
        _eval_logup_z_grand_sum_batch_init<<<mid_blocks, mid_threads, 0, stream>>>(
            input, table, last_z, mid_size_per_worker);
        _eval_logup_z_grand_sum_single_spread<<<1, 1, 0, stream>>>(
            table, mid_threads * mid_blocks);
        _eval_logup_z_grand_sum_batch_spread<<<mid_blocks, mid_threads, 0, stream>>>(
            input, table, mid_size_per_worker);

        _eval_logup_z_grand_sum_batch_spread_skip<<<n / threads, threads, 0, stream>>>(
            z, input, last_z, last_z_index, size_per_worker);

        return cudaGetLastError();
    }

    cudaError_t eval_logup_z_compose(
        Bn254FrField *z,
        Bn254FrField *input,
        Bn254FrField *table,
        Bn254FrField *multiplicity,
        const Bn254FrField *beta,
        Bn254FrField *last_z,
        int last_z_index,
        int n,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;

        int worker = 16 * 1024;
        int size_per_worker = n / worker;
        batch_add_beta_invert_kernel<<<1024, 16, 0, stream>>>(
            table, z, beta, size_per_worker);

        _eval_logup_z_step2<<<blocks, threads, 0, stream>>>(
            z, input, table, multiplicity);

        eval_logup_z_pure(z, input, table, last_z, last_z_index, n, stream);

        return cudaGetLastError();
    }

    cudaError_t logup_eval_h_inputs_product_sum(
        Bn254FrField *product,
        Bn254FrField *product_sum,
        const Bn254FrField **input_set,
        int n_set,
        int n,
        CUstream_st *stream)
    {
        int threads = n >= 64 ? 64 : 1;
        int blocks = n / threads;

        _logup_eval_h_inputs_product_sum<<<blocks, threads, 0, stream>>>(
            product, product_sum, input_set, n_set, n);

        return cudaGetLastError();
    }
}
