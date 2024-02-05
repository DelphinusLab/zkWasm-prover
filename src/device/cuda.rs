use core::mem;
use core::{cell::RefCell, marker::PhantomData};

use crate::device::DeviceResult;
use cuda_driver_sys::{CUcontext, CUctx_flags, CUdeviceptr, CUresult};

use super::{Device, DeviceBuf, Error};

lazy_static! {
    static ref CUDA_INIT: () = {
        let res = unsafe { cuda_driver_sys::cuInit(0) };

        if res != CUresult::CUDA_SUCCESS {
            panic!("cuda init failed.")
        }
    };
}

thread_local! {
    static ACITVE_CUDA_CTX: RefCell<Option<CUcontext>> =  RefCell::new(None);
}

pub struct CudaDevice {
    device: i32,
    ctx: CUcontext,
}

impl CudaDevice {
    fn acitve_ctx(&self) {
        match ACITVE_CUDA_CTX.with(|x| *x.borrow()) {
            Some(c) => {
                if c == self.ctx {
                    return;
                }
            }
            _ => {}
        }

        ACITVE_CUDA_CTX.with(|x| *x.borrow_mut() = Some(self.ctx));
        unsafe {
            cuda_driver_sys::cuCtxSetCurrent(self.ctx);
        }
    }
}

#[inline]
fn to_result<T>(value: T, res: CUresult, msg: &'static str) -> DeviceResult<T> {
    if res != CUresult::CUDA_SUCCESS {
        Err(Error::DeviceError(format!(
            "Cuda Error({:?}): {}",
            res, msg
        )))
    } else {
        Ok(value)
    }
}

type CudaDeviceBuf<T> = DeviceBuf<T, CUdeviceptr>;

impl Device<CUdeviceptr> for CudaDevice {
    fn get_device_count() -> DeviceResult<usize> {
        *CUDA_INIT;
        let mut count = 0;
        unsafe {
            let res = cuda_driver_sys::cuDeviceGetCount(&mut count);
            to_result(count as usize, res, "fail to get device count")
        }
    }

    fn get_device(idx: usize) -> DeviceResult<Self> {
        *CUDA_INIT;
        unsafe {
            let mut device = 0;
            let res = cuda_driver_sys::cuDeviceGet(&mut device, idx as i32);
            let device = to_result(device, res, "fail to get device")?;

            let mut ctx = mem::zeroed();
            let res = cuda_driver_sys::cuCtxCreate_v2(
                &mut ctx,
                CUctx_flags::CU_CTX_MAP_HOST as u32,
                device,
            );
            let ctx = to_result(ctx, res, "fail to get device")?;

            Ok(Self { device, ctx })
        }
    }

    fn alloc_device_buffer<T>(&self, size: usize) -> DeviceResult<CudaDeviceBuf<T>> {
        self.acitve_ctx();
        let mut ptr = 0 as CUdeviceptr;
        unsafe {
            let res = cuda_driver_sys::cuMemAlloc_v2(&mut ptr, size * mem::size_of::<T>());
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
        self.acitve_ctx();
        unsafe {
            let res = cuda_driver_sys::cuMemFree_v2(buf.handler);
            to_result((), res, "fail to free device memory")
        }
    }

    fn copy_from_host_to_device<T>(&self, dst: &CudaDeviceBuf<T>, src: &[T]) -> DeviceResult<()> {
        self.acitve_ctx();
        unsafe {
            let res = cuda_driver_sys::cuMemcpyHtoD_v2(
                dst.handler,
                src.as_ptr() as _,
                src.len() * mem::size_of::<T>(),
            );
            to_result((), res, "fail to copy memory from host to device")
        }
    }

    fn copy_from_device_to_host<T>(
        &self,
        dst: &mut [T],
        src: &CudaDeviceBuf<T>,
    ) -> DeviceResult<()> {
        self.acitve_ctx();
        unsafe {
            let res = cuda_driver_sys::cuMemcpyDtoH_v2(
                dst.as_ptr() as _,
                src.handler,
                dst.len() * mem::size_of::<T>(),
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }
}
