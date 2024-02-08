#[cfg(test)]
mod test {
    use std::ffi::c_void;
    use std::mem;

    use crate::device::cuda::CudaDevice;
    use crate::device::Device;
    use ark_std::rand::rngs::OsRng;
    use ark_std::{end_timer, start_timer};
    use cuda_runtime_sys::cudaError;
    use halo2_proofs::arithmetic::{BaseExt, Field};
    use halo2_proofs::pairing::bn256::{Fq, Fr, G1Affine};
    use halo2_proofs::pairing::group::ff::PrimeField;
    use halo2_proofs::pairing::group::Curve;

    #[link(name = "zkwasm_prover_kernel", kind = "static")]
    extern "C" {
        pub fn test_bn254_fr_field(
            blocks: i32,
            threads: i32,
            a: *mut c_void,
            b: *mut c_void,
            exp: *mut c_void,
            add: *mut c_void,
            sub: *mut c_void,
            mul: *mut c_void,
            unmont: *mut c_void,
            sqr: *mut c_void,
            inv: *mut c_void,
            pow: *mut c_void,
            compare: *mut c_void,
            array_len: i32,
        ) -> cudaError;

        pub fn test_bn254_fp_field(
            blocks: i32,
            threads: i32,
            a: *mut c_void,
            b: *mut c_void,
            exp: *mut c_void,
            add: *mut c_void,
            sub: *mut c_void,
            mul: *mut c_void,
            unmont: *mut c_void,
            sqr: *mut c_void,
            inv: *mut c_void,
            pow: *mut c_void,
            compare: *mut c_void,
            array_len: i32,
        ) -> cudaError;

        pub fn test_bn254_ec(
            blocks: i32,
            threads: i32,
            a: *mut c_void,
            b: *mut c_void,
            add: *mut c_void,
            sub: *mut c_void,
            double: *mut c_void,
            array_len: i32,
        ) -> cudaError;
    }

    #[test]
    fn test_bn254_fr_field_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let len = 4096;
        let threads = if len >= 32 { 32 } else { len };
        let blocks = len / threads;

        let mut a = vec![];
        let mut b = vec![];
        let mut exp = vec![];

        let mut add_expect = vec![];
        let mut sub_expect = vec![];
        let mut mul_expect = vec![];
        let mut unmont_expect = vec![];
        let mut sqr_expect = vec![];
        let mut inv_expect = vec![];
        let mut pow_expect = vec![];
        let mut compare_expect = vec![];

        unsafe {
            for _ in 0..len {
                let x = Fr::rand();
                let y = Fr::rand();
                let m = *(&Fr::rand() as *const _ as *const u64);
                a.push(x);
                b.push(y);
                exp.push(m);
            }

            let timer = start_timer!(|| "cpu costs");
            for i in 0..len {
                let x = a[i];
                let y = b[i];
                let m = exp[i];
                add_expect.push(x + y);
                sub_expect.push(x - y);
                mul_expect.push(x * y);
                unmont_expect.push(x.to_repr());
                sqr_expect.push(x.square());
                inv_expect.push(x.invert().unwrap_or(Fr::zero()));
                pow_expect.push(x.pow_vartime([m]));
                compare_expect.push(x >= y);
            }
            end_timer!(timer);

            let mut tmp_bool_buffer = vec![false; len];

            let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();
            let exp_buf = device.alloc_device_buffer_from_slice(&exp[..]).unwrap();

            let add_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let sub_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let mul_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let sqr_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let inv_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let pow_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();

            let unmont_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let compare_buf = device
                .alloc_device_buffer_from_slice(&tmp_bool_buffer[..])
                .unwrap();

            let timer = start_timer!(|| "gpu costs");
            let res = test_bn254_fr_field(
                blocks as i32,
                threads as i32,
                a_buf.handler,
                b_buf.handler,
                exp_buf.handler,
                add_buf.handler,
                sub_buf.handler,
                mul_buf.handler,
                sqr_buf.handler,
                inv_buf.handler,
                pow_buf.handler,
                unmont_buf.handler,
                compare_buf.handler,
                len as i32,
            );
            end_timer!(timer);

            assert_eq!(res, cudaError::cudaSuccess);

            device.copy_from_device_to_host(&mut b, &add_buf).unwrap();
            assert_eq!(b, add_expect);

            device.copy_from_device_to_host(&mut b, &sub_buf).unwrap();
            assert_eq!(b, sub_expect);

            device.copy_from_device_to_host(&mut b, &mul_buf).unwrap();
            assert_eq!(b, mul_expect);

            device.copy_from_device_to_host(&mut b, &sqr_buf).unwrap();
            assert_eq!(b, sqr_expect);

            device.copy_from_device_to_host(&mut b, &inv_buf).unwrap();
            assert_eq!(b, inv_expect);

            device.copy_from_device_to_host(&mut b, &pow_buf).unwrap();
            assert_eq!(b, pow_expect);

            device
                .copy_from_device_to_host(&mut b, &unmont_buf)
                .unwrap();
            assert_eq!(mem::transmute::<_, &[[u8; 32]]>(&b[..]), &unmont_expect[..]);

            device
                .copy_from_device_to_host(&mut tmp_bool_buffer, &compare_buf)
                .unwrap();
            assert_eq!(tmp_bool_buffer, compare_expect);
        }
    }

    #[test]
    fn test_bn254_fp_field_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let len = 4096;
        let threads = if len >= 32 { 32 } else { len };
        let blocks = len / threads;

        let mut a = vec![];
        let mut b = vec![];
        let mut exp = vec![];

        let mut add_expect = vec![];
        let mut sub_expect = vec![];
        let mut mul_expect = vec![];
        let mut unmont_expect = vec![];
        let mut sqr_expect = vec![];
        let mut inv_expect = vec![];
        let mut pow_expect = vec![];
        let mut compare_expect = vec![];

        unsafe {
            for _ in 0..len {
                let x = Fq::rand();
                let y = Fq::rand();
                let m = *(&Fq::rand() as *const _ as *const u64);
                a.push(x);
                b.push(y);
                exp.push(m);
            }

            let timer = start_timer!(|| "cpu costs");
            for i in 0..len {
                let x = a[i];
                let y = b[i];
                let m = exp[i];
                add_expect.push(x + y);
                sub_expect.push(x - y);
                mul_expect.push(x * y);
                unmont_expect.push(x.to_repr());
                sqr_expect.push(x.square());
                inv_expect.push(x.invert().unwrap_or(Fq::zero()));
                pow_expect.push(x.pow_vartime([m]));
                compare_expect.push(x >= y);
            }
            end_timer!(timer);

            let mut tmp_bool_buffer = vec![false; len];

            let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();
            let exp_buf = device.alloc_device_buffer_from_slice(&exp[..]).unwrap();

            let add_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let sub_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let mul_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let sqr_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let inv_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let pow_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();

            let unmont_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let compare_buf = device
                .alloc_device_buffer_from_slice(&tmp_bool_buffer[..])
                .unwrap();

            let timer = start_timer!(|| "gpu costs");
            let res = test_bn254_fp_field(
                blocks as i32,
                threads as i32,
                a_buf.handler,
                b_buf.handler,
                exp_buf.handler,
                add_buf.handler,
                sub_buf.handler,
                mul_buf.handler,
                sqr_buf.handler,
                inv_buf.handler,
                pow_buf.handler,
                unmont_buf.handler,
                compare_buf.handler,
                len as i32,
            );
            end_timer!(timer);

            assert_eq!(res, cudaError::cudaSuccess);

            device.copy_from_device_to_host(&mut b, &add_buf).unwrap();
            assert_eq!(b, add_expect);

            device.copy_from_device_to_host(&mut b, &sub_buf).unwrap();
            assert_eq!(b, sub_expect);

            device.copy_from_device_to_host(&mut b, &mul_buf).unwrap();
            assert_eq!(b, mul_expect);

            device.copy_from_device_to_host(&mut b, &sqr_buf).unwrap();
            assert_eq!(b, sqr_expect);

            device.copy_from_device_to_host(&mut b, &inv_buf).unwrap();
            assert_eq!(b, inv_expect);

            device.copy_from_device_to_host(&mut b, &pow_buf).unwrap();
            assert_eq!(b, pow_expect);

            device
                .copy_from_device_to_host(&mut b, &unmont_buf)
                .unwrap();
            assert_eq!(mem::transmute::<_, &[[u8; 32]]>(&b[..]), &unmont_expect[..]);

            device
                .copy_from_device_to_host(&mut tmp_bool_buffer, &compare_buf)
                .unwrap();
            assert_eq!(tmp_bool_buffer, compare_expect);
        }
    }

    #[test]
    fn test_bn254_ec_cuda() {
        let device = CudaDevice::get_device(0).unwrap();
        let len = 4096;
        let threads = if len >= 32 { 32 } else { len };
        let blocks = len / threads;

        let mut a = vec![];
        let mut b = vec![];
        let mut add_expect = vec![];
        let mut sub_expect = vec![];
        let mut double_expect = vec![];

        for _ in 0..len {
            let x = G1Affine::random(OsRng);
            let y = G1Affine::random(OsRng);
            a.push(x);
            b.push(y);
        }

        let timer = start_timer!(|| "cpu costs");
        for i in 0..len {
            let x = a[i];
            let y = b[i];
            add_expect.push((x + y).to_affine());
            sub_expect.push((x - y).to_affine());
            double_expect.push((x + x).to_affine());
        }
        end_timer!(timer);

        unsafe {
            let a_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let b_buf = device.alloc_device_buffer_from_slice(&b[..]).unwrap();

            let add_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let sub_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
            let double_buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();

            let timer = start_timer!(|| "gpu costs");
            let res = test_bn254_ec(
                blocks as i32,
                threads as i32,
                a_buf.handler,
                b_buf.handler,
                add_buf.handler,
                sub_buf.handler,
                double_buf.handler,
                len as i32,
            );
            end_timer!(timer);

            assert_eq!(res, cudaError::cudaSuccess);

            device
                .copy_from_device_to_host(&mut b[..], &add_buf)
                .unwrap();
            assert_eq!(b, add_expect);

            device
                .copy_from_device_to_host(&mut b[..], &sub_buf)
                .unwrap();
            assert_eq!(b, sub_expect);

            device
                .copy_from_device_to_host(&mut b[..], &double_buf)
                .unwrap();
            assert_eq!(b, double_expect);
        }
    }
}
