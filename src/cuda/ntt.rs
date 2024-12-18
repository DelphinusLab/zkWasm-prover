use halo2_proofs::arithmetic::FieldExt;

use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::Device;
use crate::device::DeviceResult;

use super::bn254_c;

pub const MAX_DEG: usize = 8;

pub(crate) fn generate_omega_buffers<F: FieldExt>(
    device: &CudaDevice,
    omega: F,
    len_log: usize,
    may_bit_reversed: bool,
) -> DeviceResult<CudaDeviceBufRaw> {
    let omegas = vec![F::one(), omega];
    let omegas_buf = device.alloc_device_buffer::<F>(1 << len_log)?;
    device.copy_from_host_to_device(&omegas_buf, &omegas[..])?;
    unsafe {
        let err = crate::cuda::bn254_c::expand_omega_buffer(
            omegas_buf.ptr(),
            len_log as i32,
            may_bit_reversed.into(),
        );
        to_result((), err, "fail to run expand_omega_buffer")?;
    }

    Ok(omegas_buf)
}

pub(crate) fn generate_ntt_buffers<F: FieldExt>(
    device: &CudaDevice,
    omega: F,
    len_log: usize,
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw)> {
    let len = 1 << len_log;

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
    let pq_buf = device.alloc_device_buffer_from_slice(&pq[..])?;

    let omegas_buf = generate_omega_buffers(device, omega, len_log, true)?;

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
            intt_divisor.map(|x| x.ptr()).unwrap_or(0 as _),
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
    Ok(())
}

pub fn ntt_sync<F: FieldExt>(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    result: &mut [F],
    len_log: usize,
) -> DeviceResult<()> {
    ntt_raw(
        device, s_buf, tmp_buf, pq_buf, omegas_buf, len_log, None, None,
    )?;
    device.copy_from_device_to_host(result, s_buf)?;
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
