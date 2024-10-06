use halo2_proofs::arithmetic::FieldExt;

use crate::cuda::bn254::FieldOp;
use crate::cuda::bn254_c;
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::Device;
use crate::device::DeviceResult;

pub const MAX_DEG: usize = 8;

pub(crate) fn generate_ntt_buffers<F: FieldExt>(
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

pub(crate) fn ntt_raw(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    len_log: usize,
    intt_divisor: Option<&CudaDeviceBufRaw>,
    stream: Option<&CudaStreamWrapper>,
) -> DeviceResult<()> {
    device.acitve_ctx()?;

    let mut swap = false;
    unsafe {
        let err = crate::cuda::bn254_c::ntt(
            s_buf.ptr(),
            tmp_buf.ptr(),
            pq_buf.ptr(),
            omegas_buf.ptr(),
            len_log as i32,
            MAX_DEG as i32,
            &mut swap as *mut _ as _,
            stream.map(|s| s.into()).unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run ntt")?;
    }

    if swap {
        std::mem::swap(s_buf, tmp_buf);
    }

    if let Some(divisor) = intt_divisor {
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
                stream.map(|s| s.into()).unwrap_or(0usize as _),
            );
            to_result((), err, "fail to run field_op in intt_raw")?;
        }
    }
    Ok(())
}

pub(crate) fn extended_prepare(
    device: &CudaDevice,
    s: &CudaDeviceBufRaw,
    coset_powers: &CudaDeviceBufRaw,
    coset_powers_n: usize,
    size: usize,
    extended_size: usize,
    to_coset: bool,
    stream: Option<&CudaStreamWrapper>,
) -> DeviceResult<()> {
    device.acitve_ctx()?;
    unsafe {
        let err = bn254_c::extended_prepare(
            s.ptr(),
            coset_powers.ptr(),
            coset_powers_n as i32,
            size as i32,
            extended_size as i32,
            if to_coset { 1 } else { 0 },
            stream.map(|s| s.into()).unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run extended_prepare")?;
        Ok(())
    }
}

pub(crate) fn _batch_ntt_raw<F: FieldExt>(
    device: &CudaDevice,
    value: Vec<(&mut [F], CudaDeviceBufRaw)>,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    len_log: usize,
    divisor: Option<&CudaDeviceBufRaw>,
) -> DeviceResult<()> {
    const MAX_CONCURRENCY: usize = 3;

    let size = 1 << len_log;
    let streams = [0; MAX_CONCURRENCY].map(|_| CudaStreamWrapper::new());
    let mut t_buf =
        [0; MAX_CONCURRENCY].map(|_| device.alloc_device_buffer_non_zeroed::<F>(size).unwrap());

    for (i, (col, mut s_buf)) in value.into_iter().enumerate() {
        let idx = i % MAX_CONCURRENCY;
        let t_buf = &mut t_buf[idx];

        ntt_raw(
            &device,
            &mut s_buf,
            t_buf,
            &pq_buf,
            &omegas_buf,
            len_log,
            divisor,
            Some(&streams[idx]),
        )?;
        device.copy_from_device_to_host_async(&mut col[..], &s_buf, (&streams[idx]).into())?;
    }

    drop(streams);
    Ok(())
}
