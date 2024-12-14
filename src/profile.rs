#[test]
fn bench_eval_logup_z() {
    use crate::cuda::bn254_c::eval_logup_z;
    use crate::cuda::bn254_c::eval_logup_z_pure;
    use crate::device::cuda::CudaBuffer;
    use crate::device::cuda::CUDA_BUFFER_ALLOCATOR;
    use crate::device::{cuda::CudaDevice, Device};
    use crate::to_result;
    use ark_std::end_timer;
    use ark_std::start_timer;
    use halo2_proofs::arithmetic::Field;
    use halo2_proofs::pairing::bn256::Fr;
    use rand::thread_rng;
    use rayon::iter::*;

    let k = 22;
    let len = 1 << k;
    let unusable_rows_start = len - 17;

    let timer = start_timer!(|| "prepare scalars");
    let scalars0 = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    let scalars1 = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    let scalars2 = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    let scalars3 = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    end_timer!(timer);

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << k) * core::mem::size_of::<Fr>(), 64);
    }

    let device = CudaDevice::get_device(0).unwrap();

    let z_buf = device
        .alloc_device_buffer_from_slice(&scalars0[..])
        .unwrap();
    let input_buf = device
        .alloc_device_buffer_from_slice(&scalars1[..])
        .unwrap();
    let table_buf = device
        .alloc_device_buffer_from_slice(&scalars2[..])
        .unwrap();
    let m_buf = device
        .alloc_device_buffer_from_slice(&scalars3[..])
        .unwrap();

    let tmp1_buf = device.alloc_device_buffer::<Fr>(len).unwrap();
    let tmp2_buf = device.alloc_device_buffer::<Fr>(len).unwrap();
    let last_z_buf = device.alloc_device_buffer::<Fr>(1).unwrap();
    let beta_buf = device
        .alloc_device_buffer_from_slice(&[Fr::random(thread_rng())][..])
        .unwrap();

    for _ in 0..5 {
        let timer = start_timer!(|| "eval_logup_z_pure");
        unsafe {
            let err = eval_logup_z_pure(
                z_buf.ptr(),
                tmp1_buf.ptr(),
                tmp2_buf.ptr(),
                last_z_buf.ptr(),
                unusable_rows_start as i32,
                len as i32,
                0 as _,
            );

            to_result((), err, "failed to run eval_logup_z").unwrap();

            device.synchronize().unwrap();
        }
        end_timer!(timer);
    }

    for _ in 0..5 {
        let timer = start_timer!(|| "eval_logup_z");
        unsafe {
            let err = eval_logup_z(
                z_buf.ptr(),
                input_buf.ptr(),
                table_buf.ptr(),
                m_buf.ptr(),
                beta_buf.ptr(),
                last_z_buf.ptr(),
                unusable_rows_start as i32,
                len as i32,
                0 as _,
            );

            to_result((), err, "failed to run eval_logup_z").unwrap();

            device.synchronize().unwrap();
        }
        end_timer!(timer);
    }
}

#[test]
fn bench_logup_sum_input_inv() {
    use crate::device::cuda::CUDA_BUFFER_ALLOCATOR;
    use crate::device::{cuda::CudaDevice, Device};
    use crate::logup_sum_input_inv;
    use ark_std::end_timer;
    use ark_std::start_timer;
    use halo2_proofs::arithmetic::Field;
    use halo2_proofs::pairing::bn256::Fr;
    use rand::thread_rng;
    use rayon::iter::*;

    let k = 22;
    let len = 1 << k;

    let timer = start_timer!(|| "prepare scalars");
    let mut scalars0 = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    end_timer!(timer);

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << k) * core::mem::size_of::<Fr>(), 64);
    }

    let device = CudaDevice::get_device(0).unwrap();

    let input_buf = device
        .alloc_device_buffer_from_slice(&scalars0[..])
        .unwrap();

    let sum_buf = device.alloc_device_buffer::<Fr>(len).unwrap();
    let tmp1_buf = device.alloc_device_buffer::<Fr>(len).unwrap();

    let beta = Fr::random(thread_rng());
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..]).unwrap();

    let mut sum = vec![Fr::zero(); len];

    for i in 0..5 {
        let timer = start_timer!(|| "logup_sum_input_inv");
        logup_sum_input_inv(
            &device, &sum_buf, &input_buf, &tmp1_buf, &beta_buf, i, len, None,
        )
        .unwrap();

        device.synchronize().unwrap();
        end_timer!(timer);

        let check_len = 65536;
        let expect_out = (0..check_len)
            .into_iter()
            .map(|i| sum[i] + (scalars0[i] + beta).invert().unwrap())
            .collect::<Vec<Fr>>();

        device.copy_from_device_to_host(&mut sum, &sum_buf).unwrap();
        device.copy_from_device_to_host(&mut scalars0, &input_buf).unwrap();
        assert!(expect_out[0..check_len] == sum[0..check_len]);
    }
}
