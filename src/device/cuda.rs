use core::mem;
use core::{cell::RefCell, marker::PhantomData};
use std::ffi::c_void;

use cuda_runtime_sys::cudaError;

use crate::device::DeviceResult;

use super::{Device, DeviceBuf, Error};

lazy_static! {
    static ref CUDA_INIT: () = {
        /*
        let res = unsafe { cuda_runtime_sys::cuInit(0) };

        if res != cudaError::cudaSuccess {
            panic!("cuda init failed.")
        }
         */
    };
}

thread_local! {
    static ACITVE_CUDA_DEVICE: RefCell<i32> = RefCell::new(-1);
}

pub struct CudaDevice {
    device: i32,
    //ctx: CUcontext,
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
        *CUDA_INIT;
        let mut count = 0;
        unsafe {
            let res = cuda_runtime_sys::cudaGetDeviceCount(&mut count);
            to_result(count as usize, res, "fail to get device count")
        }
    }

    fn get_device(idx: usize) -> DeviceResult<Self> {
        /*
        *CUDA_INIT;
        unsafe {
            let mut device = 0;
            let res = cuda_runtime_sys::cudaGetDevice(&mut device, idx as i32);
            let device = to_result(device, res, "fail to get device")?;

            let mut ctx = mem::zeroed();
            let res = cuda_driver_sys::cuCtxCreate_v2(
                &mut ctx,
                CUctx_flags::CU_CTX_MAP_HOST as u32
                    | CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC as u32,
                device,
            );
            let ctx = to_result(ctx, res, "fail to get device")?;

            Ok(Self { device, ctx })
        }
         */
        Ok(Self { device: idx as i32 })
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
}
