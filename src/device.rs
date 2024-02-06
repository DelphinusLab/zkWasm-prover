use std::marker::PhantomData;

pub mod cuda;

#[derive(Debug)]
pub enum Error {
    DeviceError(String),
}

type DeviceResult<T> = Result<T, Error>;

pub struct DeviceBuf<T, H: Sized> {
    pub handler: H,
    phantom: PhantomData<T>,
}

pub trait Device<H>: Sized {
    fn get_device_count() -> DeviceResult<usize>;
    fn get_device(idx: usize) -> DeviceResult<Self>;

    fn alloc_device_buffer<T>(&self, size: usize) -> DeviceResult<DeviceBuf<T, H>>;
    fn alloc_device_buffer_from_slice<T>(&self, data: &[T]) -> DeviceResult<DeviceBuf<T, H>> {
        let buf = self.alloc_device_buffer(data.len())?;
        self.copy_from_host_to_device(&buf, data)?;
        Ok(buf)
    }

    fn free_device_buffer<T>(&self, buf: DeviceBuf<T, H>) -> DeviceResult<()>;

    fn copy_from_host_to_device<T>(&self, dst: &DeviceBuf<T, H>, src: &[T]) -> DeviceResult<()>;
    fn copy_from_device_to_host<T>(&self, dst: &mut [T], dst: &DeviceBuf<T, H>)
        -> DeviceResult<()>;

    fn synchronize(&self) -> DeviceResult<()>;
}
