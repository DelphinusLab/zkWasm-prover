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

#[test]
fn test_msm_v2() {
    use crate::CudaDevice;
    use ark_std::{end_timer, start_timer};
    use halo2_proofs::pairing::group::Curve;
    use halo2_proofs::{
        arithmetic::{BaseExt, CurveAffine, Field},
        pairing::bn256::{Fq, Fr, G1Affine, G1},
    };
    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::ParallelIterator;
    use std::{fs::File, path::Path};

    use crate::{cuda::msm::batch_msm, device::Device};

    {
        let mut allocator = crate::device::cuda::CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << 22) * core::mem::size_of::<Fr>(), 100);
    }

    const L: usize = 1 << 22;

    let timer = start_timer!(|| "gen points");
    let f = format!("./random{}.bin", L);
    let points = if !Path::new(&f).exists() {
        let mut fd = File::create(&f).unwrap();
        let points = vec![0; L]
            .par_iter()
            .map(|_| (G1::generator() * Fr::random(rand::thread_rng())).to_affine())
            .collect::<Vec<_>>();
        for p in points.iter() {
            p.x.write(&mut fd).unwrap();
            p.y.write(&mut fd).unwrap();
        }
        points
    } else {
        let mut fd = File::open(&f).unwrap();
        let mut points = vec![];
        for _ in 0..L {
            let x = Fq::read(&mut fd).unwrap();
            let y = Fq::read(&mut fd).unwrap();
            let p = G1Affine::from_xy(x, y).unwrap();
            points.push(p);
        }
        points
    };
    end_timer!(timer);

    let timer = start_timer!(|| "gen scalars");
    let scalars = vec![0; L]
        .par_iter()
        .map(|_| Fr::random(rand::thread_rng()))
        .collect::<Vec<_>>();
    end_timer!(timer);

    let device = CudaDevice::get_device(0).unwrap();
    let p_buf = device.alloc_device_buffer_from_slice(&points[..]).unwrap();
    for _ in 0..5 {
        let timer = start_timer!(|| "xxx");
        let res = batch_msm::<G1Affine>(
            &device,
            &p_buf,
            vec![
                &scalars, &scalars, &scalars, &scalars, &scalars, &scalars, &scalars, &scalars,
            ],
            L,
        )
        .unwrap();
        end_timer!(timer);
        println!("res is {:?}", res);
        for r in res {
            let succ: bool = r.is_on_curve().into();
            assert!(succ);
        }
    }
}
