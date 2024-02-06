use std::alloc::AllocError;
use std::alloc::Allocator;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

pub struct DeviceBuffer<T> {
    dptr: *mut c_void,
    phantom: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn new(dptr: *mut c_void) -> Self {
        Self {
            dptr,
            phantom: PhantomData,
        }
    }
}
/*
pub struct CudaHostAllocator {}

impl CudaHostAllocator {
    pub fn new(device_idx: usize) -> Self {
        unsafe {
            cuda_driver_sys::cuInit(0);

            let mut device = CUdevice::default();
            let mut ctx = mem::zeroed();
            let mut pctx = &mut ctx as *mut _;
            cuDeviceGet(&mut device, device_idx as i32);
            cuCtxCreate_v2(&mut pctx, 0, device);
        };

        Self {}
    }
}

unsafe impl Allocator for CudaHostAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let mut p = 0 as *mut ::std::os::raw::c_void;
            println!("cuda alloc size {}", layout.size());
            let res = cuda_driver_sys::cuMemAllocHost_v2(&mut p, layout.size());
            println!("cuda alloc ptr {:?} {:?}", p, res);
            Ok(NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
                p as _,
                layout.size(),
            )))
        }
    }
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        Ok(ptr)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        unsafe {
            cuda_driver_sys::cuMemFreeHost(ptr.as_ptr() as _);
        }
    }
}
 */
