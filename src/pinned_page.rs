use core::slice;
use cuda_runtime_sys::cudaError;
use libc::c_void;
use std::{
    alloc::{AllocError, Allocator, Layout},
    collections::HashMap,
    ptr::NonNull,
    sync::Mutex,
};

// Design for k22 and k23

const ALIGN_SIZE: usize = 1 << 27; // aligned to k22 buffer size
const ALLOC_SIZE: usize = 1 << 28;
const CHUNCKS: usize = 2;

#[derive(Debug)]
struct SimpleAllocator {
    buffers: Vec<(usize, [bool; CHUNCKS], usize)>, // base_addr, status bitmap, used chuncks
    full_free_buffers: Vec<usize>,                 // buffer_idx
    parti_free_buffers: Vec<usize>,                // buffer_idx
    alloc_records: HashMap<usize, (usize, bool)>,  // addr -> (buffer_idx, is_partial_alloc)
}

impl SimpleAllocator {
    fn new() -> Self {
        assert_eq!(ALIGN_SIZE * CHUNCKS, ALLOC_SIZE);
        Self {
            buffers: vec![],
            full_free_buffers: vec![],
            parti_free_buffers: vec![],
            alloc_records: HashMap::new(),
        }
    }

    fn ensure_full_buffer(&mut self) {
        if self.full_free_buffers.is_empty() {
            let mut p: *mut c_void = unsafe { std::mem::zeroed() };
            let res = unsafe { cuda_runtime_sys::cudaMallocHost(&mut p, ALLOC_SIZE) };
            assert_eq!(res, cudaError::cudaSuccess);
            self.full_free_buffers.push(self.buffers.len());
            self.buffers.push((p as usize, [false; CHUNCKS], 0));
        }
    }

    fn alloc_parital_buffer(&mut self) -> usize {
        if self.parti_free_buffers.is_empty() {
            self.ensure_full_buffer();
            let idx = self.full_free_buffers.pop().unwrap();
            self.parti_free_buffers.push(idx);
        }

        let buffer_idx = self.parti_free_buffers.last().cloned().unwrap();
        let (ptr, status, used) = &mut self.buffers[buffer_idx];
        let (shift, _) = status
            .iter()
            .enumerate()
            .find(|(_, is_in_use)| !**is_in_use)
            .unwrap();

        status[shift] = true;
        *used += 1;

        if *used == CHUNCKS {
            self.parti_free_buffers.pop();
        }

        let addr = *ptr + shift * ALIGN_SIZE;
        self.alloc_records.insert(addr, (buffer_idx, true));
        addr
    }

    fn alloc_full_buffer(&mut self) -> usize {
        self.ensure_full_buffer();
        let buffer_idx = self.full_free_buffers.pop().unwrap();

        let addr = self.buffers[buffer_idx].0;
        self.alloc_records.insert(addr, (buffer_idx, false));
        addr
    }

    fn free(&mut self, addr: usize) {
        let (buffer_idx, is_partial_alloc) = self.alloc_records.remove(&addr).unwrap();

        if !is_partial_alloc {
            self.full_free_buffers.push(buffer_idx)
        } else {
            let (ptr, status, used) = &mut self.buffers[buffer_idx];
            let shift = (addr - *ptr) / ALIGN_SIZE;
            status[shift] = false;
            *used -= 1;

            if *used == 0 {
                self.full_free_buffers.push(buffer_idx);
                let idx = self
                    .parti_free_buffers
                    .iter()
                    .enumerate()
                    .find(|(_, idx)| **idx == buffer_idx)
                    .unwrap()
                    .0;
                let last = self.parti_free_buffers.len() - 1;
                self.parti_free_buffers.swap(idx, last);
                self.parti_free_buffers.pop();
            }

            if *used == CHUNCKS - 1 {
                self.parti_free_buffers.push(buffer_idx);
            }
        }
    }
}

lazy_static! {
    static ref PINNED_BUFFER_CACHE: Mutex<SimpleAllocator> = Mutex::new(SimpleAllocator::new());
}

pub fn print_pinned_cache_info() {
    let pinned_buffer_cache = PINNED_BUFFER_CACHE.lock().unwrap();
    println!(
        "zkwasm-prover cached pinned memory size: {} MB",
        (pinned_buffer_cache.buffers.len() * ALLOC_SIZE) >> 20
    );
}

#[derive(Clone)]
pub struct PinnedPageAllocator;

unsafe impl Allocator for PinnedPageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let aligned_layout = layout.align_to(ALIGN_SIZE).unwrap();
        unsafe {
            let mut pinned_buffer_cache = PINNED_BUFFER_CACHE.lock().unwrap();
            assert!(aligned_layout.size() <= ALLOC_SIZE);
            let p = if aligned_layout.size() <= ALIGN_SIZE {
                pinned_buffer_cache.alloc_parital_buffer()
            } else {
                pinned_buffer_cache.alloc_full_buffer()
            };

            Ok(NonNull::new_unchecked(slice::from_raw_parts_mut(
                p as *mut _,
                layout.size(),
            )))
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, _layout: Layout) {
        let mut pinned_buffer_cache = PINNED_BUFFER_CACHE.lock().unwrap();
        pinned_buffer_cache.free(ptr.as_ptr() as usize)
    }
}
