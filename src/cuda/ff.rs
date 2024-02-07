#[cfg(test)]
mod test {
    use std::ffi::c_void;

    use crate::device::cuda::CudaDevice;
    use crate::device::{Device, DeviceBuf};
    use ark_std::{end_timer, start_timer};
    use cuda_runtime_sys::cudaError;
    use halo2_proofs::arithmetic::BaseExt;
    use halo2_proofs::pairing::bn256::Fr;
    use halo2_proofs::pairing::group::ff::PrimeField;

    #[link(name = "zkwasm_prover_kernel", kind = "static")]
    extern "C" {
        pub fn test_bn254_fr_field_compare(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            bn254_field_array_b: *mut c_void,
            res: *mut c_void,
            array_len: i32,
        ) -> cudaError;
        pub fn test_bn254_fr_field_mul(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            bn254_field_array_b: *mut c_void,
            bn254_field_array_c: *mut c_void,
            array_len: i32,
        ) -> cudaError;
        pub fn test_bn254_fr_field_add(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            bn254_field_array_b: *mut c_void,
            bn254_field_array_c: *mut c_void,
            array_len: i32,
        ) -> cudaError;
        pub fn test_bn254_fr_field_sub(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            bn254_field_array_b: *mut c_void,
            bn254_field_array_c: *mut c_void,
            array_len: i32,
        ) -> cudaError;
        pub fn test_bn254_fr_field_mont(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            array_len: i32,
        ) -> cudaError;
        pub fn test_bn254_fr_field_unmont(
            blocks: i32,
            threads: i32,
            bn254_field_array_a: *mut c_void,
            array_len: i32,
        ) -> cudaError;
    }

    #[test]
    fn test_bn254_fr_field_unmont_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let len = 1024;
        let threads = if len >= 32 { 32 } else { len };
        let mut a = vec![];
        let mut b: Vec<[u8; 32]> = vec![];
        let mut c: Vec<[u8; 32]> = vec![[0u8; 32]; len];

        for _ in 0..len {
            let x = Fr::rand();
            a.push(x);
            b.push(x.to_repr());
        }

        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        unsafe {
            let err = test_bn254_fr_field_unmont(
                len as i32 / threads as i32,
                threads as i32,
                a_buf.handler,
                len as i32,
            );
            println!("error is {:?}", err);
            device
                .copy_from_device_to_host(
                    &mut c[..],
                    &*(&a_buf as *const DeviceBuf<Fr, *mut c_void> as *const _),
                )
                .unwrap();
        }
        assert_eq!(c, b);
    }

    #[test]
    fn test_bn254_fr_field_mont_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let len = 1;
        let threads = if len >= 32 { 32 } else { len };
        let mut a = vec![];
        let mut b = vec![];

        for _ in 0..len {
            let x = Fr::rand();
            a.push(x);
            b.push(x);
        }

        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        unsafe {
            let err = test_bn254_fr_field_mont(
                len as i32 / threads as i32,
                threads as i32,
                a_buf.handler,
                len as i32,
            );
            println!("error is {:?}", err);
            device
                .copy_from_device_to_host(
                    &mut b[..],
                    &*(&a_buf as *const DeviceBuf<Fr, *mut c_void> as *const _),
                )
                .unwrap();
        }
        assert_eq!(a, b);
    }

    #[test]
    fn test_bn254_fr_field_sub_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let mut a = vec![];
        let mut b = vec![];
        let mut c_expect = vec![];
        let len = 16;

        for _ in 0..len {
            let x = Fr::rand();
            let y = Fr::rand();

            a.push(x);
            b.push(y);
        }

        let timer = start_timer!(|| "cpu sub");
        for i in 0..len {
            c_expect.push(a[i] - b[i]);
        }
        end_timer!(timer);

        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();

        let timer = start_timer!(|| "gpu sub");
        let res = unsafe {
            test_bn254_fr_field_sub(
                (len / 16) as i32,
                16,
                a_buf.handler,
                b_buf.handler,
                a_buf.handler,
                len as i32,
            )
        };
        println!("error is {:?}", res);
        end_timer!(timer);

        let timer = start_timer!(|| "gpu copy");
        device.copy_from_device_to_host(&mut a[..], &a_buf).unwrap();
        end_timer!(timer);

        assert_eq!(a, c_expect);
    }

    #[test]
    fn test_bn254_fr_field_add_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let mut a = vec![];
        let mut b = vec![];
        let mut c_expect = vec![];
        let len = 32;

        for _ in 0..len {
            let x = Fr::rand();
            let y = Fr::rand();

            a.push(x);
            b.push(y);
        }

        let timer = start_timer!(|| "cpu add");
        for i in 0..len {
            c_expect.push(a[i] + b[i]);
        }
        end_timer!(timer);

        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();

        let timer = start_timer!(|| "gpu add");
        unsafe {
            test_bn254_fr_field_add(
                (len / 32) as i32,
                32,
                a_buf.handler,
                b_buf.handler,
                a_buf.handler,
                len as i32,
            );
        }
        end_timer!(timer);

        let timer = start_timer!(|| "gpu copy");
        device.copy_from_device_to_host(&mut a[..], &a_buf).unwrap();
        end_timer!(timer);

        assert_eq!(a, c_expect);
    }

    #[test]
    fn test_bn254_fr_field_mul_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let mut a = vec![];
        let mut b = vec![];
        let mut c_expect = vec![];
        let len = 32;

        for _ in 0..len {
            let x = Fr::rand();
            let y = Fr::rand();

            a.push(x);
            b.push(y);
        }

        let timer = start_timer!(|| "cpu add");
        for i in 0..len {
            c_expect.push(a[i] * b[i]);
        }
        end_timer!(timer);

        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();

        let timer = start_timer!(|| "gpu add");
        unsafe {
            test_bn254_fr_field_mul(
                (len / 32) as i32,
                32,
                a_buf.handler,
                b_buf.handler,
                a_buf.handler,
                len as i32,
            );
        }
        end_timer!(timer);

        let timer = start_timer!(|| "gpu copy");
        device.copy_from_device_to_host(&mut a[..], &a_buf).unwrap();
        end_timer!(timer);

        assert_eq!(a, c_expect);
    }

    #[test]
    fn test_bn254_fr_field_compare_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let mut a = vec![];
        let mut b = vec![];
        let mut c_expect = vec![];
        let len = 8;

        let gte = |l: &[u64; 4], r: &[u64; 4]| {
            for i in 0..4 {
                if l[3 - i] < r[3 - i] {
                    return false;
                }

                if l[3 - i] > r[3 - i] {
                    return true;
                }
            }

            return true;
        };

        for _ in 0..len {
            let x = Fr::rand();
            let y = Fr::rand();
            unsafe {
                println!(
                    "{:?}, {:?}",
                    core::mem::transmute::<_, &[u64; 4]>(&x),
                    core::mem::transmute::<_, &[u64; 4]>(&y)
                );
                c_expect.push(gte(
                    core::mem::transmute::<_, &[u64; 4]>(&x),
                    core::mem::transmute::<_, &[u64; 4]>(&y),
                ));
                a.push(x);
                b.push(y);
            }
        }

        let mut c = vec![false; len];
        let c_buf = device.alloc_device_buffer_from_slice(&c[..]).unwrap();
        let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
        let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();

        unsafe {
            test_bn254_fr_field_compare(
                1,
                len as i32,
                a_buf.handler,
                b_buf.handler,
                c_buf.handler,
                len as i32,
            );
        }
        device.copy_from_device_to_host(&mut c[..], &c_buf).unwrap();
        assert_eq!(c, c_expect);
    }
}
