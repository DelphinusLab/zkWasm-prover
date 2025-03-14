use super::bn254::FieldOp;
use super::bn254_c;
use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw};
use crate::device::Device;
use crate::device::Error;

use cuda_runtime_sys::cudaStream_t;
use halo2_proofs::arithmetic::FieldExt;
use libc::c_void;

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
