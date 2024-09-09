use crate::device::cuda::c_void;
use crate::to_result;

pub(crate) struct CudaBufferAllocator {
    pub(crate) ptr_base: usize,
    pub(crate) bit_map: Vec<bool>,
    pub(crate) chunk_size: usize,
    pub(crate) total_size: usize,
}

impl CudaBufferAllocator {
    pub(crate) fn new() -> Self {
        Self {
            ptr_base: 0,
            chunk_size: 0,
            total_size: 0,
            bit_map: vec![],
        }
    }

    fn clear(&mut self) {
        if self.total_size > 0 {
            unsafe { cuda_runtime_sys::cudaFree(self.ptr_base as _) };
            self.ptr_base = 0;
            self.chunk_size = 0;
            self.total_size = 0;
            self.bit_map = vec![];
        }
    }

    pub(crate) fn reset(&mut self, chunk_size: usize, count: usize) {
        if chunk_size == self.chunk_size && count == self.bit_map.len() {
            return;
        }

        self.clear();

        let total_size = chunk_size * count;
        let err = unsafe {
            cuda_runtime_sys::cudaMalloc(&mut self.ptr_base as *mut _ as *mut _, total_size)
        };
        to_result((), err, "fail to alloc device memory").unwrap();

        self.chunk_size = chunk_size;
        self.total_size = total_size;
        self.bit_map = vec![false; count]
    }

    pub(crate) fn alloc(&mut self, size: usize) -> *mut c_void {
        let count = (size + self.chunk_size - 1) / self.chunk_size;

        let mut end = 0;
        let mut picked = 0;

        while picked < count && end < self.bit_map.len() {
            if self.bit_map[end] {
                // for alignment
                end = (end + count) / count * count;
                picked = 0;
            } else {
                end += 1;
                picked += 1;
            }
        }

        if picked == count {
            for i in (end - picked)..end {
                self.bit_map[i] = true;
            }
            let res = (self.ptr_base + (end - picked) * self.chunk_size) as _;
            return res;
        } else {
            let mut sum = 0;
            for used in self.bit_map.iter() {
                if *used {
                    sum += 1;
                }
            }
            println!("failed to alloc device buffer with {} bytes, count is {}, used chunks {}", size, count, sum);
            panic!("Cuda Device OOM");
        }
    }

    pub(crate) fn free(&mut self, ptr: *mut c_void, size: usize) {
        let offset = ptr as usize - self.ptr_base;
        assert!(offset < self.total_size);
        assert!(offset % self.chunk_size == 0);

        let offset = offset as usize / self.chunk_size;
        for i in 0..(size + self.chunk_size - 1) / self.chunk_size {
            self.bit_map[i + offset] = false;
        }
    }
}

impl Drop for CudaBufferAllocator {
    fn drop(&mut self) {
        self.clear()
    }
}
