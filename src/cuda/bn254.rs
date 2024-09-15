use super::bn254_c;
use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw};
use crate::device::Device;
use crate::device::Error;

use cuda_runtime_sys::cudaStream_t;
use halo2_proofs::arithmetic::FieldExt;
use libc::c_void;

#[derive(Debug, PartialEq)]
pub(crate) enum FieldOp {
    Add = 0,
    Mul = 1,
    UOp = 2,
    Sub = 3,
}

pub(crate) trait FieldOpArgs<F: FieldExt> {
    fn ptr(&self) -> *mut c_void {
        0 as *mut _
    }
    fn rot(&self) -> i32 {
        0
    }
    fn constant(&self) -> Option<F> {
        None
    }
    fn constant_ptr(&self) -> *mut c_void {
        0 as *mut _
    }
}

impl<F: FieldExt> FieldOpArgs<F> for Option<F> {
    fn constant(&self) -> Option<F> {
        match self {
            Some(f) => Some(*f),
            None => None,
        }
    }
}

impl<'a, F: FieldExt> FieldOpArgs<F> for &'a CudaDeviceBufRaw {
    fn ptr(&self) -> *mut c_void {
        (*self).ptr()
    }
}

impl<'a, F: FieldExt> FieldOpArgs<F> for (&'a CudaDeviceBufRaw, i32) {
    fn ptr(&self) -> *mut c_void {
        (*self).0.ptr()
    }
    fn rot(&self) -> i32 {
        self.1
    }
}

impl<'a, F: FieldExt> FieldOpArgs<F> for (&'a CudaDeviceBufRaw, i32, F) {
    fn ptr(&self) -> *mut c_void {
        (*self).0.ptr()
    }
    fn rot(&self) -> i32 {
        self.1
    }
    fn constant(&self) -> Option<F> {
        Some(self.2)
    }
}

impl<'a, F: FieldExt> FieldOpArgs<F> for (&'a CudaDeviceBufRaw, &'a CudaDeviceBufRaw) {
    fn ptr(&self) -> *mut c_void {
        (*self).0.ptr()
    }
    fn constant_ptr(&self) -> *mut c_void {
        (*self).1.ptr()
    }
}

impl<'a, F: FieldExt> FieldOpArgs<F> for ((), &'a CudaDeviceBufRaw) {
    fn constant_ptr(&self) -> *mut c_void {
        (*self).1.ptr()
    }
}

pub(crate) fn field_op<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    l: impl FieldOpArgs<F>,
    r: impl FieldOpArgs<F>,
    size: usize,
    op: FieldOp,
    stream: Option<cudaStream_t>,
) -> Result<(), Error> {
    device.acitve_ctx()?;

    let l_c = match l.constant() {
        Some(c) => Some(device.alloc_device_buffer_from_slice([c].as_slice())?),
        None => None,
    };

    let r_c = match r.constant() {
        Some(c) => Some(device.alloc_device_buffer_from_slice([c].as_slice())?),
        None => None,
    };

    unsafe {
        let err = bn254_c::field_op(
            res.ptr(),
            l.ptr(),
            l.rot(),
            l_c.as_ref().map_or(l.constant_ptr(), |x| x.ptr()),
            r.ptr(),
            r.rot(),
            r_c.as_ref().map_or(r.constant_ptr(), |x| x.ptr()),
            size as i32,
            op as i32,
            stream.unwrap_or(0usize as _),
        );
        to_result((), err, "fail to run field_op")?;
    }
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
