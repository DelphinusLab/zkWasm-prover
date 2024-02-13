use core::mem;
use core::{cell::RefCell, marker::PhantomData};
use std::ffi::c_void;
use std::mem::size_of;

use cuda_runtime_sys::cudaError;

use super::{Device, DeviceBuf, Error};
use crate::device::DeviceResult;

thread_local! {
    static ACITVE_CUDA_DEVICE: RefCell<i32> = RefCell::new(-1);
}

pub struct CudaDevice {
    device: i32,
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        ACITVE_CUDA_DEVICE.with(|x| {
            if *x.borrow() != self.device {
                *x.borrow_mut() = -1;
            }
        });
    }
}

impl CudaDevice {
    fn acitve_ctx(&self) -> DeviceResult<()> {
        ACITVE_CUDA_DEVICE.with(|x| {
            if *x.borrow() != self.device {
                *x.borrow_mut() = self.device
            }
        });

        unsafe {
            let res = cuda_runtime_sys::cudaSetDevice(self.device);
            to_result((), res, "fail to set device")
        }
    }
}

#[inline]
fn to_result<T>(value: T, res: cudaError, msg: &'static str) -> DeviceResult<T> {
    if res != cudaError::cudaSuccess {
        Err(Error::DeviceError(format!(
            "Cuda Error({:?}): {}",
            res, msg
        )))
    } else {
        Ok(value)
    }
}

type CudaDeviceBuf<T> = DeviceBuf<T, *mut c_void>;

impl Device<*mut c_void> for CudaDevice {
    fn get_device_count() -> DeviceResult<usize> {
        let mut count = 0;
        unsafe {
            let res = cuda_runtime_sys::cudaGetDeviceCount(&mut count);
            to_result(count as usize, res, "fail to get device count")
        }
    }

    fn get_device(idx: usize) -> DeviceResult<Self> {
        let count = Self::get_device_count()?;
        if idx < count {
            Ok(Self { device: idx as i32 })
        } else {
            Err(Error::DeviceError(format!(
                "Cuda Error(): Invalid device idx {}",
                idx
            )))
        }
    }

    fn alloc_device_buffer<T>(&self, size: usize) -> DeviceResult<CudaDeviceBuf<T>> {
        self.acitve_ctx()?;
        let mut ptr = 0 as *mut c_void;
        unsafe {
            let res = cuda_runtime_sys::cudaMalloc(&mut ptr, size * mem::size_of::<T>());
            to_result(
                DeviceBuf {
                    handler: ptr,
                    phantom: PhantomData,
                },
                res,
                "fail to alloc device memory",
            )
        }
    }

    fn free_device_buffer<T>(&self, buf: CudaDeviceBuf<T>) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaFree(buf.handler);
            to_result((), res, "fail to free device memory")
        }
    }

    fn copy_from_host_to_device<T>(&self, dst: &CudaDeviceBuf<T>, src: &[T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpy(
                dst.handler,
                src.as_ptr() as _,
                src.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            to_result((), res, "fail to copy memory from host to device")
        }
    }

    fn copy_from_device_to_host<T>(
        &self,
        dst: &mut [T],
        src: &CudaDeviceBuf<T>,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpy(
                dst.as_ptr() as _,
                src.handler,
                dst.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }

    fn synchronize(&self) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaDeviceSynchronize();
            to_result((), res, "fail to synchronize")
        }
    }

    fn pin_memory<T>(&self, dst: &mut [T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaHostRegister(
                dst.as_mut_ptr() as *mut _,
                dst.len() * size_of::<T>(),
                cuda_runtime_sys::cudaHostAllocMapped,
            );
            to_result((), res, "fail to synchronize")
        }
    }

    fn unpin_memory<T>(&self, dst: &mut [T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaHostUnregister(dst.as_mut_ptr() as *mut _);
            to_result((), res, "fail to synchronize")
        }
    }
}
