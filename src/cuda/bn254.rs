use super::bn254_c;
use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw};
use crate::device::Device;
use crate::device::Error;

use cuda_runtime_sys::cudaStream_t;
use halo2_proofs::arithmetic::FieldExt;

#[derive(Debug, PartialEq)]
pub(crate) enum FieldOp {
    Add = 0,
    Mul = 1,
    UOp = 2,
    Sub = 3,
}

pub(crate) fn field_op_v2<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    l: Option<&CudaDeviceBufRaw>,
    l_c: Option<F>,
    r: Option<&CudaDeviceBufRaw>,
    r_c: Option<F>,
    size: usize,
    op: FieldOp,
) -> Result<(), Error> {
    field_op(device, res, l, 0, l_c, r, 0, r_c, size, op, None)?;

    Ok(())
}

pub(crate) fn field_sub<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    rhs: &CudaDeviceBufRaw,
    size: usize,
) -> Result<(), Error> {
    field_op_v2::<F>(
        device,
        res,
        Some(res),
        None,
        Some(rhs),
        None,
        size,
        FieldOp::Sub,
    )?;
    Ok(())
}

pub(crate) fn field_mul<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    rhs: &CudaDeviceBufRaw,
    size: usize,
) -> Result<(), Error> {
    field_op_v2::<F>(
        device,
        res,
        Some(res),
        None,
        Some(rhs),
        None,
        size,
        FieldOp::Mul,
    )?;
    Ok(())
}

pub(crate) fn pick_from_buf<F: FieldExt>(
    device: &CudaDevice,
    buf: &CudaDeviceBufRaw,
    rot: isize,
    i: isize,
    size: usize,
) -> Result<F, Error> {
    let mut v = [F::zero()];
    device.acitve_ctx()?;
    unsafe {
        let err = cuda_runtime_sys::cudaMemcpy(
            v.as_mut_ptr() as _,
            buf.ptr().offset(
                ((rot + i + size as isize) & (size as isize - 1))
                    * core::mem::size_of::<F>() as isize,
            ),
            core::mem::size_of::<F>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        to_result((), err, "fail to pick_from_buf")?;
    }
    Ok(v[0])
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

pub(crate) fn field_op<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    l: Option<&CudaDeviceBufRaw>,
    l_rot: i32,
    l_c: Option<F>,
    r: Option<&CudaDeviceBufRaw>,
    r_rot: i32,
    r_c: Option<F>,
    size: usize,
    op: FieldOp,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    let l_c = if l_c.is_none() {
        None
    } else {
        Some(device.alloc_device_buffer_from_slice([l_c.unwrap()].as_slice())?)
    };
    let r_c = if r_c.is_none() {
        None
    } else {
        Some(device.alloc_device_buffer_from_slice([r_c.unwrap()].as_slice())?)
    };

    unsafe {
        device.acitve_ctx()?;
        let err = bn254_c::field_op(
            res.ptr(),
            l.map_or(0usize as *mut _, |x| x.ptr()),
            l_rot,
            l_c.as_ref().map_or(0usize as *mut _, |x| x.ptr()),
            r.map_or(0usize as *mut _, |x| x.ptr()),
            r_rot,
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
    device.acitve_ctx()?;
    unsafe {
        let err = bn254_c::permutation_eval_h_l(
            res.ptr(),
            beta.ptr(),
            gamma.ptr(),
            p.ptr(),
            n as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run permutation_eval_h_l")?;
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
    if rot == 0 {
        device.copy_from_device_to_device::<F>(&dst, 0, src, 0, size)?;
        device.synchronize()?;
    } else if rot > 0 {
        let rot = rot as usize;
        let len = size - rot as usize;
        device.copy_from_device_to_device::<F>(&dst, 0, src, rot, len)?;
        device.synchronize()?;
        device.copy_from_device_to_device::<F>(&dst, len, src, 0, rot)?;
        device.synchronize()?;
    } else {
        let rot = -rot as usize;
        let len = size - rot;
        device.copy_from_device_to_device::<F>(&dst, 0, src, rot, len)?;
        device.synchronize()?;
        device.copy_from_device_to_device::<F>(&dst, len, src, 0, rot)?;
        device.synchronize()?;
    }
    Ok(())
}
