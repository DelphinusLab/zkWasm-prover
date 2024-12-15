use cuda_runtime_sys::cudaStream_t;
use halo2_proofs::arithmetic::FieldExt;

use super::bn254_c;
use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper};
use crate::device::Error;
use crate::device::{Device, DeviceResult};

#[derive(Debug, PartialEq)]
pub(crate) enum FieldOp {
    Add = 0,
    Mul = 1,
    UOp = 2,
    Sub = 3,
}

pub(crate) fn pick_from_buf<T>(
    device: &CudaDevice,
    buf: &CudaDeviceBufRaw,
    rot: isize,
    i: isize,
    size: usize,
) -> Result<T, Error> {
    let mut v: [T; 1] = unsafe { std::mem::zeroed() };
    device.acitve_ctx()?;
    unsafe {
        let err = cuda_runtime_sys::cudaMemcpy(
            v.as_mut_ptr() as _,
            buf.ptr().offset(
                ((rot + i + size as isize) & (size as isize - 1))
                    * core::mem::size_of::<T>() as isize,
            ),
            core::mem::size_of::<T>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        to_result((), err, "fail to pick_from_buf")?;
    }
    Ok(v.into_iter().next().unwrap())
}

pub(crate) fn field_op_v3(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    l: Option<&CudaDeviceBufRaw>,
    l_c: Option<&CudaDeviceBufRaw>,
    r: Option<&CudaDeviceBufRaw>,
    r_c: Option<&CudaDeviceBufRaw>,
    size: usize,
    op: FieldOp,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let err = bn254_c::field_op(
            res.ptr(),
            l.map_or(0usize as *mut _, |x| x.ptr()),
            0,
            l_c.as_ref().map_or(0usize as *mut _, |x| x.ptr()),
            r.map_or(0usize as *mut _, |x| x.ptr()),
            0,
            r_c.as_ref().map_or(0usize as *mut _, |x| x.ptr()),
            size as i32,
            op as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run field_op")?;
    }
    Ok(())
}

pub const MAX_DEG: usize = 8;

pub fn ntt_prepare<F: FieldExt>(
    device: &CudaDevice,
    omega: F,
    len_log: usize,
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw)> {
    let len = 1 << len_log;
    let omegas = vec![F::one(), omega];

    let max_deg = MAX_DEG.min(len_log);
    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
    let twiddle = omega.pow_vartime([(len >> max_deg) as u64]);
    pq[0] = F::one();
    if max_deg > 1 {
        pq[1] = twiddle;
        for i in 2..(1 << max_deg >> 1) {
            pq[i] = pq[i - 1];
            pq[i].mul_assign(&twiddle);
        }
    }

    let omegas_buf = device.alloc_device_buffer::<F>(1 << len_log)?;
    device.copy_from_host_to_device(&omegas_buf, &omegas[..])?;
    unsafe {
        let err =
            crate::cuda::bn254_c::expand_omega_buffer(omegas_buf.ptr(), (1 << len_log) as i32);
        to_result((), err, "fail to run expand_omega_buffer")?;
    }
    let pq_buf = device.alloc_device_buffer_from_slice(&pq[..])?;

    Ok((omegas_buf, pq_buf))
}

pub fn ntt_raw(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    len_log: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    let mut swap = false;
    unsafe {
        device.acitve_ctx()?;
        let err = crate::cuda::bn254::bn254_c::ntt(
            s_buf.ptr(),
            tmp_buf.ptr(),
            pq_buf.ptr(),
            omegas_buf.ptr(),
            len_log as i32,
            MAX_DEG as i32,
            &mut swap as *mut _ as _,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run ntt")?;
    }
    if swap {
        std::mem::swap(s_buf, tmp_buf);
    }
    Ok(())
}

pub fn intt_raw(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    divisor: &CudaDeviceBufRaw,
    len_log: usize,
) -> Result<(), Error> {
    intt_raw_async(
        device, s_buf, tmp_buf, pq_buf, omegas_buf, divisor, len_log, None,
    )
}

pub fn batch_intt_raw<F: FieldExt>(
    device: &CudaDevice,
    mut s_buf: Vec<CudaDeviceBufRaw>,
    value: Vec<&mut [F]>,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    divisor: &CudaDeviceBufRaw,
    len_log: usize,
) -> Result<(), Error> {
    const MAX_CONCURRENCY: usize = 3;

    let size = 1 << len_log;
    let mut streams = [None; MAX_CONCURRENCY];
    let mut t_buf = [0; MAX_CONCURRENCY].map(|_| device.alloc_device_buffer::<F>(size).unwrap());

    for (i, col) in value.into_iter().enumerate() {
        let idx = i % MAX_CONCURRENCY;
        let t_buf = &mut t_buf[idx];

        unsafe {
            if let Some(last_stream) = streams[idx] {
                cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                cuda_runtime_sys::cudaStreamDestroy(last_stream);
            }

            let mut stream = std::mem::zeroed();
            let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
            crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;
            intt_raw_async(
                &device,
                &mut s_buf[i],
                t_buf,
                &pq_buf,
                &omegas_buf,
                &divisor,
                len_log,
                Some(stream),
            )?;
            device.copy_from_device_to_host_async(&mut col[..], &s_buf[i], stream)?;
            streams[idx] = Some(stream);
        }
    }

    for idx in 0..MAX_CONCURRENCY {
        if let Some(last_stream) = streams[idx] {
            unsafe {
                cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                cuda_runtime_sys::cudaStreamDestroy(last_stream);
            }
        }
    }

    Ok(())
}

pub fn intt_raw_async(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    divisor: &CudaDeviceBufRaw,
    len_log: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    ntt_raw(device, s_buf, tmp_buf, pq_buf, omegas_buf, len_log, stream)?;
    unsafe {
        let err = bn254_c::field_op(
            s_buf.ptr(),
            s_buf.ptr(),
            0,
            0usize as *mut _,
            0usize as *mut _,
            0,
            divisor.ptr(),
            (1 << len_log) as i32,
            FieldOp::Mul as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run field_op in intt_raw")?;
    }
    Ok(())
}

pub fn ntt<F: FieldExt>(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    result: &mut [F],
    len_log: usize,
) -> Result<(), Error> {
    ntt_raw(device, s_buf, tmp_buf, pq_buf, omegas_buf, len_log, None)?;
    device.copy_from_device_to_host(result, s_buf)?;
    Ok(())
}

// plonk permutation
pub fn permutation_eval_h_p1(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    first_set: &CudaDeviceBufRaw,
    last_set: &CudaDeviceBufRaw,
    l0: &CudaDeviceBufRaw,
    l_last: &CudaDeviceBufRaw,
    y: &CudaDeviceBufRaw,
    n: usize,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let err = bn254_c::permutation_eval_h_p1(
            res.ptr(),
            first_set.ptr(),
            last_set.ptr(),
            l0.ptr(),
            l_last.ptr(),
            y.ptr(),
            n as i32,
        );
        to_result((), err, "fail to run permutation_eval_h_p1")?;
        device.synchronize()?;
    }
    Ok(())
}

pub fn permutation_eval_h_p2(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    set: &[CudaDeviceBufRaw],
    l0: &CudaDeviceBufRaw,
    l_last: &CudaDeviceBufRaw,
    y: &CudaDeviceBufRaw,
    rot: usize,
    n: usize,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let sets = device
            .alloc_device_buffer_from_slice(&set.iter().map(|x| x.ptr()).collect::<Vec<_>>()[..])?;
        let err = bn254_c::permutation_eval_h_p2(
            res.ptr(),
            sets.ptr(),
            l0.ptr(),
            l_last.ptr(),
            y.ptr(),
            set.len() as i32,
            rot as i32,
            n as i32,
        );
        to_result((), err, "fail to run permutation_eval_h_p2")?;
        device.synchronize()?;
    }
    Ok(())
}

pub fn permutation_eval_h_l(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    beta: &CudaDeviceBufRaw,
    gamma: &CudaDeviceBufRaw,
    p: &CudaDeviceBufRaw,
    n: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let err = bn254_c::permutation_eval_h_l(
            res.ptr(),
            beta.ptr(),
            gamma.ptr(),
            p.ptr(),
            n as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run permutation_eval_h_l")?;
        device.synchronize()?;
    }
    Ok(())
}

pub fn buffer_copy_with_shift<F: FieldExt>(
    device: &CudaDevice,
    dst: &CudaDeviceBufRaw,
    src: &CudaDeviceBufRaw,
    rot: isize,
    size: usize,
) -> Result<(), Error> {
    let (sw, stream) = CudaStreamWrapper::new_with_inner();
    if rot == 0 {
        device.copy_from_device_to_device_async::<F>(&dst, 0, src, 0, size, stream)?;
    } else if rot > 0 {
        let rot = rot as usize;
        let len = size - rot as usize;
        device.copy_from_device_to_device_async::<F>(&dst, 0, src, rot, len, stream)?;
        device.copy_from_device_to_device_async::<F>(&dst, len, src, 0, rot, stream)?;
    } else {
        let rot = -rot as usize;
        let len = size - rot;
        device.copy_from_device_to_device_async::<F>(&dst, 0, src, rot, len, stream)?;
        device.copy_from_device_to_device_async::<F>(&dst, len, src, 0, rot, stream)?;
    }
    sw.sync();
    Ok(())
}

pub fn logup_eval_h_z_set(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    set: &[CudaDeviceBufRaw],
    l0: &CudaDeviceBufRaw,
    l_last: &CudaDeviceBufRaw,
    y: &CudaDeviceBufRaw,
    rot: usize,
    n: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let sets = device
            .alloc_device_buffer_from_slice(&set.iter().map(|x| x.ptr()).collect::<Vec<_>>()[..])?;
        let err = bn254_c::logup_eval_h_z_set(
            res.ptr(),
            sets.ptr(),
            l0.ptr(),
            l_last.ptr(),
            y.ptr(),
            set.len() as i32,
            rot as i32,
            n as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run logup_eval_h_z_set")?;
    }
    Ok(())
}

pub fn logup_sum_input_inv(
    device: &CudaDevice,
    sum: &CudaDeviceBufRaw,
    input: &CudaDeviceBufRaw,
    temp: &CudaDeviceBufRaw,
    beta: &CudaDeviceBufRaw,
    init: usize,
    n: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;

        let err = bn254_c::logup_sum_input_inv(
            sum.ptr(),
            input.ptr(),
            temp.ptr(),
            beta.ptr(),
            init as i32,
            n as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run field_op")?;
    }
    Ok(())
}

pub fn logup_eval_h_inputs_product_sum(
    device: &CudaDevice,
    product: &CudaDeviceBufRaw,
    product_sum: &CudaDeviceBufRaw,
    set: &[CudaDeviceBufRaw],
    n: usize,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let sets = device
            .alloc_device_buffer_from_slice(&set.iter().map(|x| x.ptr()).collect::<Vec<_>>()[..])?;
        let err = bn254_c::logup_eval_h_inputs_product_sum(
            product.ptr(),
            product_sum.ptr(),
            sets.ptr(),
            set.len() as i32,
            n as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run permutation_eval_h_p2")?;
    }
    Ok(())
}
