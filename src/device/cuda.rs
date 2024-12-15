use core::cell::RefCell;
use core::mem;
use cuda_runtime_sys::{cudaError, cudaStream_t, CUstream_st};
use log::info;
use std::collections::HashMap;
use std::mem::size_of;
use std::{ffi::c_void, sync::Mutex};

use super::{Device, DeviceBuf, Error};
use crate::device::cuda::buffer_allocator::CudaBufferAllocator;
use crate::device::DeviceResult;

mod buffer_allocator;

thread_local! {
    static ACITVE_CUDA_DEVICE: RefCell<i32> = RefCell::new(-1);
}

const HUGE_BUFFER_SIZE: usize = 1 << 27;

lazy_static! {
    pub(crate) static ref CUDA_BUFFER_CACHE: Mutex<HashMap::<(i32, usize), Vec<usize>>> =
        Mutex::new(HashMap::new());
    pub(crate) static ref CUDA_BUFFER_ALLOCATOR: Mutex<CudaBufferAllocator> =
        Mutex::new(CudaBufferAllocator::new());
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    device: i32,
}

impl Drop for CudaDevice {
    fn drop(&mut self) {}
}

impl CudaDevice {
    pub(crate) fn acitve_ctx(&self) -> DeviceResult<()> {
        ACITVE_CUDA_DEVICE.with(|x| {
            if *x.borrow() != self.device {
                *x.borrow_mut() = self.device;
                unsafe {
                    let res = cuda_runtime_sys::cudaSetDevice(self.device);
                    to_result((), res, "fail to set device")?
                }
            }
            Ok(())
        })
    }
}

#[inline]
pub(crate) fn to_result<T>(value: T, res: cudaError, msg: &'static str) -> DeviceResult<T> {
    if res != cudaError::cudaSuccess {
        Err(Error::DeviceError(format!(
            "Cuda Error({:?}): {}",
            res, msg
        )))
    } else {
        Ok(value)
    }
}

pub trait CudaBuffer {
    fn ptr(&self) -> *mut c_void;
    fn device<'a>(&'a self) -> &'a CudaDevice;
}

impl CudaBuffer for CudaDeviceBufRaw {
    fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn device<'a>(&'a self) -> &'a CudaDevice {
        &self.device
    }
}

#[derive(Debug)]
pub struct CudaDeviceBufRaw {
    pub(crate) ptr: *mut c_void,
    pub(crate) device: CudaDevice,
    pub(crate) size: usize,
}

extern "C" {
    pub fn cudaFreeAsync(ptr: *mut c_void, stream: cudaStream_t) -> cudaError;
}

impl Drop for CudaDeviceBufRaw {
    fn drop(&mut self) {
        if self.size < HUGE_BUFFER_SIZE {
            let mut cache = CUDA_BUFFER_CACHE.lock().unwrap();
            let arr = cache
                .entry((self.device.device, self.size))
                .or_insert(vec![]);
            assert!(!arr.contains(&(self.ptr() as usize)));
            arr.push(self.ptr() as usize);
        } else {
            let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
            allocator.free(self.ptr(), self.size);
        }
    }
}

impl DeviceBuf for CudaDeviceBufRaw {}

impl CudaDevice {
    pub fn copy_from_host_to_device_async<T>(
        &self,
        dst: &CudaDeviceBufRaw,
        src: &[T],
        stream: cudaStream_t,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpyAsync(
                dst.ptr(),
                src.as_ptr() as _,
                src.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            );
            to_result((), res, "fail to copy memory from host to device")
        }
    }

    pub fn copy_from_device_to_host_async<T>(
        &self,
        dst: &mut [T],
        src: &CudaDeviceBufRaw,
        stream: cudaStream_t,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpyAsync(
                dst.as_ptr() as _,
                src.ptr(),
                dst.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }

    pub fn copy_from_device_to_host_async_v2<T>(
        &self,
        dst: &mut [T],
        src: &CudaDeviceBufRaw,
        offset: isize,
        stream: Option<cudaStream_t>,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpyAsync(
                dst.as_ptr() as _,
                src.ptr().offset(offset * mem::size_of::<T>() as isize),
                dst.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream.unwrap_or(0usize as _),
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }

    fn _alloc_device_buffer<T>(
        &self,
        size: usize,
        zero: bool,
        stream: Option<cudaStream_t>,
    ) -> DeviceResult<CudaDeviceBufRaw> {
        //println!("alloc device memory {}", size * mem::size_of::<T>());
        //self.print_memory_info()?;
        unsafe {
            let size = size * mem::size_of::<T>();
            if size < HUGE_BUFFER_SIZE {
                let mut cache = CUDA_BUFFER_CACHE.lock().unwrap();
                let arr = cache.entry((self.device, size)).or_insert(vec![]);

                if arr.len() > 0 {
                    let ret = CudaDeviceBufRaw {
                        ptr: arr.pop().unwrap() as *mut c_void,
                        device: self.clone(),
                        size,
                    };
                    if zero {
                        cuda_runtime_sys::cudaMemsetAsync(
                            ret.ptr(),
                            0,
                            size,
                            stream.unwrap_or(0usize as _),
                        );
                    }
                    return Ok(ret);
                }

                self.acitve_ctx()?;
                let mut ptr = 0 as *mut c_void;
                let res = cuda_runtime_sys::cudaMalloc(&mut ptr, size);
                //self.print_memory_info()?;
                to_result(
                    CudaDeviceBufRaw {
                        ptr,
                        device: self.clone(),
                        size,
                    },
                    res,
                    "fail to alloc device memory",
                )
            } else {
                let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
                let ptr = allocator.alloc(size);
                if zero {
                    cuda_runtime_sys::cudaMemsetAsync(ptr, 0, size, stream.unwrap_or(0usize as _));
                }
                Ok(CudaDeviceBufRaw {
                    ptr,
                    device: self.clone(),
                    size,
                })
            }
        }
    }

    pub(crate) fn alloc_device_buffer_async<T>(
        &self,
        size: usize,
        stream: &CudaStreamWrapper,
    ) -> DeviceResult<CudaDeviceBufRaw> {
        self._alloc_device_buffer::<T>(size, true, Some(stream.into()))
    }

    pub fn copy_from_device_to_device_async<T>(
        &self,
        dst: &CudaDeviceBufRaw,
        dst_offset: usize,
        src: &CudaDeviceBufRaw,
        src_offset: usize,
        size: usize,
        stream: cudaStream_t,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpyAsync(
                (dst.ptr() as usize + dst_offset * size_of::<T>()) as _,
                (src.ptr() as usize + src_offset * size_of::<T>()) as _,
                size * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }

    pub fn alloc_device_buffer_from_slice_async<T>(
        &self,
        data: &[T],
        stream: cudaStream_t,
    ) -> DeviceResult<CudaDeviceBufRaw> {
        let buf = self._alloc_device_buffer::<T>(data.len(), false, None)?;
        self.copy_from_host_to_device_async(&buf, data, stream)?;
        Ok(buf)
    }

    pub fn alloc_device_buffer_non_zeroed<T>(&self, size: usize) -> DeviceResult<CudaDeviceBufRaw> {
        self._alloc_device_buffer::<T>(size, false, None)
    }
}

impl Device<CudaDeviceBufRaw> for CudaDevice {
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

    fn print_memory_info(&self) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let mut free = 0;
            let mut total = 0;
            cuda_runtime_sys::cudaMemGetInfo(&mut free, &mut total);
            info!("free is {}, total is {}", free, total);
        }
        Ok(())
    }

    fn alloc_device_buffer<T>(&self, size: usize) -> DeviceResult<CudaDeviceBufRaw> {
        self._alloc_device_buffer::<T>(size, true, None)
    }

    fn alloc_device_buffer_from_slice<T>(&self, data: &[T]) -> DeviceResult<CudaDeviceBufRaw> {
        let buf = self._alloc_device_buffer::<T>(data.len(), false, None)?;
        self.copy_from_host_to_device(&buf, data)?;
        Ok(buf)
    }

    fn copy_from_host_to_device<T>(&self, dst: &CudaDeviceBufRaw, src: &[T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpyAsync(
                dst.ptr(),
                src.as_ptr() as _,
                src.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
                0usize as *mut _,
            );
            to_result((), res, "fail to copy memory from host to device")
        }
    }

    fn copy_from_device_to_host<T>(
        &self,
        dst: &mut [T],
        src: &CudaDeviceBufRaw,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpy(
                dst.as_ptr() as _,
                src.ptr(),
                dst.len() * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            to_result((), res, "fail to copy memory from device to host")
        }
    }

    fn copy_from_device_to_device<T>(
        &self,
        dst: &CudaDeviceBufRaw,
        dst_offset: usize,
        src: &CudaDeviceBufRaw,
        src_offset: usize,
        len: usize,
    ) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaMemcpy(
                (dst.ptr()).offset((dst_offset * mem::size_of::<T>()) as isize),
                (src.ptr()).offset((src_offset * mem::size_of::<T>()) as isize),
                len * mem::size_of::<T>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            to_result((), res, "fail to copy memory from device to device")
        }
    }

    fn synchronize(&self) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaDeviceSynchronize();
            to_result((), res, "fail to synchronize")
        }
    }

    fn pin_memory<T>(&self, dst: &[T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res: cudaError = cuda_runtime_sys::cudaHostRegister(
                dst.as_ptr() as *mut _,
                dst.len() * size_of::<T>(),
                cuda_runtime_sys::cudaHostAllocMapped,
            );
            if res == cudaError::cudaErrorHostMemoryAlreadyRegistered {
                return Ok(());
            }
            to_result((), res, "fail to synchronize")
        }
    }

    fn unpin_memory<T>(&self, dst: &[T]) -> DeviceResult<()> {
        self.acitve_ctx()?;
        unsafe {
            let res = cuda_runtime_sys::cudaHostUnregister(dst.as_ptr() as *mut _);
            to_result((), res, "fail to synchronize")
        }
    }
}

pub(crate) struct CudaStreamWrapper(cudaStream_t);

impl CudaStreamWrapper {
    pub fn new() -> Self {
        unsafe {
            let mut stream = std::mem::zeroed();
            let _ = cuda_runtime_sys::cudaStreamCreate(&mut stream);
            Self(stream)
        }
    }

    pub fn new_with_inner() -> (Self, *mut CUstream_st) {
        unsafe {
            let mut stream = std::mem::zeroed();
            let _ = cuda_runtime_sys::cudaStreamCreate(&mut stream);
            (Self(stream), stream)
        }
    }

    pub fn sync(&self) {
        unsafe {
            let err = cuda_runtime_sys::cudaStreamSynchronize(self.into());
            to_result((), err, "fail to run cudaStreamSynchronize").unwrap();
        }
    }
}

impl From<&CudaStreamWrapper> for cudaStream_t {
    fn from(value: &CudaStreamWrapper) -> Self {
        value.0
    }
}

impl Drop for CudaStreamWrapper {
    fn drop(&mut self) {
        unsafe {
            let err = cuda_runtime_sys::cudaStreamSynchronize(self.0);
            to_result((), err, "fail to run cudaStreamSynchronize").unwrap();
            let err = cuda_runtime_sys::cudaStreamDestroy(self.0);
            to_result((), err, "fail to run cudaStreamDestroy").unwrap();
        }
    }
}
