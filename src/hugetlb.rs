use core::slice;
use libc::{
    c_void, mmap, munmap, MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_PRIVATE, PROT_READ,
    PROT_WRITE,
};
use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::{null_mut, NonNull},
};

const HUGEPAGE_SIZE: usize = 2 << 20;

#[derive(Clone)]
pub struct HugePageAllocator;

unsafe impl Allocator for HugePageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let aligned_layout = layout.align_to(HUGEPAGE_SIZE).unwrap();
        unsafe {
            let p = mmap(
                null_mut(),
                aligned_layout.size(),
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            );

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
        munmap(ptr.as_ptr() as *mut c_void, layout.size());
    }
}
