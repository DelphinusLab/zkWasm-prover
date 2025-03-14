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
        device
            .copy_from_device_to_host(&mut scalars0, &input_buf)
            .unwrap();
        assert!(expect_out[0..check_len] == sum[0..check_len]);
    }
}

#[test]
fn test_bn254_ntt() {
    use crate::device::cuda::CudaDevice;
    use crate::device::Device;
    use crate::generate_ntt_buffers;
    use crate::ntt_raw;
    use crate::CUDA_BUFFER_ALLOCATOR;
    use ark_std::{end_timer, start_timer};
    use halo2_proofs::arithmetic::{best_fft_cpu, BaseExt, Field as _, FieldExt};
    use halo2_proofs::pairing::bn256::Fr;
    use halo2_proofs::pairing::group::ff::PrimeField as _;

    let device = CudaDevice::get_device(0).unwrap();
    let len_log = 22;
    let len = 1 << len_log;

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << len_log) * core::mem::size_of::<Fr>(), 16);
    }

    let mut omega = Fr::ROOT_OF_UNITY_INV.invert().unwrap();
    for _ in len_log..Fr::S {
        omega = omega.square();
    }

    let (omegas_buf, pq_buf) = generate_ntt_buffers(&device, omega, len_log as usize).unwrap();
    let (intt_omegas_buf, intt_pq_buf) =
        generate_ntt_buffers(&device, omega.invert().unwrap(), len_log as usize).unwrap();
    let divisor = Fr::from(1 << len_log).invert().unwrap();
    let divisor_buf = device
        .alloc_device_buffer_from_slice(&vec![divisor][..])
        .unwrap();

    for i in 0..128 {
        println!("test round {}", i);

        let mut s_vec = {
            let mut s_vec = vec![];
            s_vec.push(Fr::rand());
            for i in 1..len {
                s_vec.push(s_vec[i - 1] + Fr::from(i as u64));
            }
            s_vec
        };

        let mut expected_ntt_s = s_vec.clone();
        best_fft_cpu(&mut expected_ntt_s[..], omega, len_log);

        let s_origin = s_vec.clone();
        let s = &mut s_vec[..];
        device.pin_memory(s).unwrap();

        let mut a_buf = device.alloc_device_buffer_from_slice(s).unwrap();
        let mut b_buf = device.alloc_device_buffer_from_slice(s).unwrap();
        device.synchronize().unwrap();

        let timer = start_timer!(|| format!("NTT gpu costs for k {}, s[0] {:?}", len_log, &s_origin[0]));
        ntt_raw(
            &device,
            &mut a_buf,
            &mut b_buf,
            &pq_buf,
            &omegas_buf,
            len_log as usize,
            None,
            None,
        )
        .unwrap();
        device.synchronize().unwrap();
        end_timer!(timer);

        device.copy_from_device_to_host(s, &a_buf).unwrap();
        assert!(s == expected_ntt_s);

        device.unpin_memory(s).unwrap();

        let timer = start_timer!(|| format!("INTT gpu costs for k {}", len_log));
        ntt_raw(
            &device,
            &mut a_buf,
            &mut b_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            len_log as usize,
            Some(&divisor_buf),
            None,
        )
        .unwrap();
        device.synchronize().unwrap();
        end_timer!(timer);

        device.copy_from_device_to_host(s, &a_buf).unwrap();
        assert!(s == s_origin);
    }
}
