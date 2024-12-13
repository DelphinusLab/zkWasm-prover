use std::collections::VecDeque;

use ark_std::{end_timer, start_timer};

use crate::device::{
    cuda::{CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper},
    DeviceResult,
};

pub(crate) struct AsyncCopyQueue {
    queue: VecDeque<(CudaDeviceBufRaw, CudaStreamWrapper)>,
    max: usize,
}

impl AsyncCopyQueue {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(max),
            max,
        }
    }

    fn push_raw(&mut self, buf: CudaDeviceBufRaw, sw: CudaStreamWrapper) -> DeviceResult<()> {
        if self.queue.len() == self.max {
            // let timer = start_timer!(|| "wait async copy queue");
            let (head, head_sw) = self.queue.pop_front().unwrap();
            head_sw.sync();
            drop(head);
            // end_timer!(timer);
        }

        self.queue.push_back((buf, sw));

        Ok(())
    }

    pub(crate) fn push_with_copy_buffer<T>(
        &mut self,
        device: &CudaDevice,
        dev_buf: &CudaDeviceBufRaw,
        host_buf: &mut [T],
        size: usize,
    ) -> DeviceResult<()> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let new_buf = device.alloc_device_buffer_non_zeroed::<T>(size)?;
        device.copy_from_device_to_device_async::<T>(&new_buf, 0, &dev_buf, 0, size, stream)?;
        sw.sync();
        device.copy_from_device_to_host_async(host_buf, &new_buf, stream)?;
        self.push_raw(new_buf, sw)?;
        Ok(())
    }

    pub(crate) fn push_with_new_buffer<T>(
        &mut self,
        device: &CudaDevice,
        dev_buf: &mut CudaDeviceBufRaw,
        host_buf: &mut [T],
        size: usize,
    ) -> DeviceResult<()> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let mut new_buf = device.alloc_device_buffer_non_zeroed::<T>(size)?;
        core::mem::swap(dev_buf, &mut new_buf);
        device.copy_from_device_to_host_async(host_buf, &new_buf, stream)?;
        self.push_raw(new_buf, sw)?;
        Ok(())
    }

    pub(crate) fn push<T>(
        &mut self,
        device: &CudaDevice,
        dev_buf: CudaDeviceBufRaw,
        host_buf: &mut [T],
    ) -> DeviceResult<()> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        device.copy_from_device_to_host_async(host_buf, &dev_buf, stream)?;
        self.push_raw(dev_buf, sw)?;
        Ok(())
    }

    pub(crate) fn sync_all(&mut self) {
        while let Some((buf, sw)) = self.queue.pop_front() {
            sw.sync();
            drop(buf);
        }
    }
}
