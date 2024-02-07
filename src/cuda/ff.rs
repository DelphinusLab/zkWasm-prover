use std::ffi::c_void;

#[link(name = "zkwasm_prover_kernel", kind = "static")]
extern "C" {
    pub fn test_int_add(blocks: i32, threads: i32, array: *mut c_void, array_len: i32);
}

#[test]
fn test_cuda() {
    use crate::device::cuda::CudaDevice;
    use crate::device::Device;

    let device = CudaDevice::get_device(0).unwrap();
    let a = vec![1, 2, 3];
    let c = vec![2, 3, 4];
    let mut b = vec![0, 0, 0];
    let buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
    device.copy_from_device_to_host(&mut b, &buf).unwrap();
    assert_eq!(a, b);

    unsafe {
        let res = test_int_add(1, b.len() as i32, buf.handler, b.len() as i32);
        println!("res is {:?}", res);
        device.synchronize().unwrap();
    }

    device.copy_from_device_to_host(&mut b, &buf).unwrap();
    assert_eq!(c, b);
}
