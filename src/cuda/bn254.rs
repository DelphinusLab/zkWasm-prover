use cuda_runtime_sys::cudaStream_t;
use halo2_proofs::arithmetic::FieldExt;

use super::bn254_c;
use super::ntt::MAX_DEG;
use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper};
use crate::device::Error;
use crate::device::Device;

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
