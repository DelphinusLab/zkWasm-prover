#[test]
fn bench_eval_logup_z_pure() {
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
    let scalars = vec![0; len]
        .into_par_iter()
        .map(|_| Fr::random(thread_rng()))
        .collect::<Vec<_>>();
    end_timer!(timer);

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << k) * core::mem::size_of::<Fr>(), 64);
    }

    let device = CudaDevice::get_device(0).unwrap();

    let z_buf = device.alloc_device_buffer_from_slice(&scalars[..]).unwrap();
    let sum_buf = device.alloc_device_buffer::<Fr>(len).unwrap();
    let table_buf = device.alloc_device_buffer::<Fr>(len).unwrap();
    let last_z_buf = device.alloc_device_buffer::<Fr>(1).unwrap();

    for _ in 0..5 {
        let timer = start_timer!(|| "eval_logup_z_pure");
        unsafe {
            let err = eval_logup_z_pure(
                z_buf.ptr(),
                sum_buf.ptr(),
                table_buf.ptr(),
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
