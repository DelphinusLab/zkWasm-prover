use cuda_runtime_sys::{cudaError, CUstream_st};
use std::ffi::c_void;

#[link(name = "zkwasm_prover_kernel", kind = "static")]
extern "C" {
    pub fn ntt(
        buf: *mut c_void,
        tmp: *mut c_void,
        pq: *mut c_void,
        omega: *mut c_void,
        array_log: i32,
        max_deg: i32,
        swap: *mut c_void,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn field_op(
        res: *mut c_void,
        l: *mut c_void,
        l_rot: i32,
        l_c: *mut c_void,
        r: *mut c_void,
        r_rot: i32,
        r_c: *mut c_void,
        size: i32,
        op: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn extended_prepare(
        s: *mut c_void,
        coset_powers: *mut c_void,
        coset_powers_n: i32,
        size: i32,
        extended_size: i32,
        to_coset: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn permutation_eval_h_p1(
        res: *mut c_void,
        first_set: *mut c_void,
        last_set: *mut c_void,
        l0: *mut c_void,
        l_last: *mut c_void,
        y: *mut c_void,
        n: i32,
    ) -> cudaError;

    pub fn permutation_eval_h_p2(
        res: *mut c_void,
        set: *mut c_void,
        l0: *mut c_void,
        l_last: *mut c_void,
        y: *mut c_void,
        n_set: i32,
        rot: i32,
        n: i32,
    ) -> cudaError;

    pub fn permutation_eval_h_l(
        res: *mut c_void,
        beta: *mut c_void,
        gamma: *mut c_void,
        p: *mut c_void,
        n: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn field_sum(
        res: *mut c_void,
        v: *mut c_void,
        v_c: *mut c_void,
        v_rot: *mut c_void,
        omegas: *mut c_void,
        v_n: i32,
        n: i32,
    ) -> cudaError;

    pub fn field_op_batch_mul_sum(
        res: *mut c_void,
        v: *mut c_void, // coeff0, a00, a01, null, coeff1, a10, a11, null,
        rot: *mut c_void,
        v_n: i32,
        n: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn lookup_eval_h(
        res: *mut c_void,
        input: *mut c_void,
        table: *mut c_void,
        permuted_input: *mut c_void,
        permuted_table: *mut c_void,
        z: *mut c_void,
        l0: *mut c_void,
        l_last: *mut c_void,
        l_active_row: *mut c_void,
        y: *mut c_void,
        beta: *mut c_void,
        gamma: *mut c_void,
        rot: i32,
        n: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn shuffle_eval_h(
        res: *mut c_void,
        input: *mut c_void,
        table: *mut c_void,
        z: *mut c_void,
        l0: *mut c_void,
        l_last: *mut c_void,
        l_active_row: *mut c_void,
        y: *mut c_void,
        rot: i32,
        n: i32,
    ) -> cudaError;

    pub fn shuffle_eval_h_v2(
        res: *mut c_void,
        inputs: *mut c_void,
        tables: *mut c_void,
        betas: *mut c_void,
        group_len: i32,
        z: *mut c_void,
        l0: *mut c_void,
        l_last: *mut c_void,
        l_active_row: *mut c_void,
        y: *mut c_void,
        rot: i32,
        n: i32,
    ) -> cudaError;

    pub fn expand_omega_buffer(buf: *mut c_void, n: i32) -> cudaError;

    pub fn field_mul_zip(buf: *mut c_void, coeff: *mut c_void, coeff_n: i32, n: i32) -> cudaError;

    pub fn poly_eval(
        p: *mut c_void,
        res: *mut c_void,
        tmp: *mut c_void,
        x: *mut c_void,
        n: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn shplonk_h_x_merge(
        res: *mut c_void,
        v: *mut c_void,
        values: *mut c_void,
        omegas: *mut c_void,
        diff_points: *mut c_void,
        diff_points_n: i32,
        n: i32,
    ) -> cudaError;

    pub fn shplonk_h_x_div_points(
        values: *mut c_void,
        omegas: *mut c_void,
        points: *mut c_void,
        points_n: i32,
        n: i32,
    ) -> cudaError;

    pub fn eval_lookup_z(
        z: *mut c_void,
        input: *mut c_void,
        table: *mut c_void,
        permuted_input: *mut c_void,
        permuted_table: *mut c_void,
        beta_gamma: *mut c_void,
        n: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn batch_msm_collect(
        remain_indices: *mut c_void,
        remain_acc: *mut c_void,
        next_remain_indices: *mut c_void,
        next_remain_acc: *mut c_void,
        buckets: *mut c_void,
        workers: u32,
        window_bits: u32,
        windows: u32,
        batch_size: u32,
        stream: *mut CUstream_st,
    ) -> cudaError;

    pub fn msm(
        res: *mut c_void,
        p: *mut c_void,
        s: *mut c_void,
        sort_buf: *mut c_void,
        indices_buf: *mut c_void,
        array_len: i32,
        window_bits: i32,
        threads: i32,
        max_worker: i32,
        prepared_sort_indices_temp_storage_bytes: i32,
        stream: *mut CUstream_st,
    ) -> cudaError;
}
