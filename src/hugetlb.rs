use core::slice;
use libc::{
    c_void, mmap, MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE,
};
use std::{
    alloc::{AllocError, Allocator, Layout},
    collections::HashMap,
    ptr::{null_mut, NonNull},
    sync::Mutex,
};

use crate::device::{cuda::CudaDevice, Device};

lazy_static! {
    pub static ref PINNED_BUFFER_CACHE: Mutex<HashMap::<usize, Vec<usize>>> =
        Mutex::new(HashMap::new());
    pub static ref UNPINNED_BUFFER_CACHE: Mutex<HashMap::<usize, Vec<usize>>> =
        Mutex::new(HashMap::new());
}

const HUGEPAGE_SIZE: usize = 2 << 20;

#[derive(Clone)]
pub struct HugePageAllocator;

unsafe impl Allocator for HugePageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let aligned_layout = layout.align_to(HUGEPAGE_SIZE).unwrap();
        unsafe {
            let mut cache = PINNED_BUFFER_CACHE.lock().unwrap();
            let arr = cache.entry(aligned_layout.size()).or_insert(vec![]);
            let p = if arr.len() > 0 {
                arr.pop().unwrap() as *mut c_void
            } else {
                let p = mmap(
                    null_mut(),
                    aligned_layout.size(),
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                );
                let device = CudaDevice::get_device(0).unwrap();
                device
                    .pin_memory(slice::from_raw_parts_mut(p as *mut _, layout.size()))
                    .unwrap();
                p
            };

            if p == MAP_FAILED {
                return Err(AllocError {});
            }

            Ok(NonNull::new_unchecked(slice::from_raw_parts_mut(
                p as *mut _,
                layout.size(),
            )))
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        //munmap(ptr.as_ptr() as *mut c_void, layout.size());
        let mut cache = PINNED_BUFFER_CACHE.lock().unwrap();
        let arr = cache.entry(layout.size()).or_insert(vec![]);
        arr.push(ptr.as_ptr() as usize);
    }
}

#[derive(Clone)]
pub struct UnpinnedHugePageAllocator;

unsafe impl Allocator for UnpinnedHugePageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let aligned_layout = layout.align_to(HUGEPAGE_SIZE).unwrap();
        unsafe {
            let mut cache = UNPINNED_BUFFER_CACHE.lock().unwrap();
            let arr = cache.entry(aligned_layout.size()).or_insert(vec![]);
            let p = if arr.len() > 0 {
                arr.pop().unwrap() as *mut c_void
            } else {
                let p = mmap(
                    null_mut(),
                    aligned_layout.size(),
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                );
                p
            };

            if p == MAP_FAILED {
                return Err(AllocError {});
            }

            Ok(NonNull::new_unchecked(slice::from_raw_parts_mut(
                p as *mut _,
                layout.size(),
            )))
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        //munmap(ptr.as_ptr() as *mut c_void, layout.size());
        let mut cache = UNPINNED_BUFFER_CACHE.lock().unwrap();
        let arr = cache.entry(layout.size()).or_insert(vec![]);
        arr.push(ptr.as_ptr() as usize);
    }
}
