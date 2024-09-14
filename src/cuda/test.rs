use super::ntt::generate_ntt_buffers;
use crate::cuda::ntt::ntt_raw;
use crate::device::cuda::CudaDevice;
use crate::device::Device;
use ark_std::{end_timer, start_timer};
use halo2_proofs::arithmetic::{best_fft_cpu, BaseExt, Field as _, FieldExt};
use halo2_proofs::pairing::bn256::Fr;
use halo2_proofs::pairing::group::ff::PrimeField as _;

#[test]
fn test_bn254_fft() {
    let device = CudaDevice::get_device(0).unwrap();
    let len_log = 20;
    let len = 1 << len_log;

    let mut omega = Fr::ROOT_OF_UNITY_INV.invert().unwrap();
    for _ in len_log..Fr::S {
        omega = omega.square();
    }

    let (omegas_buf, pq_buf) = generate_ntt_buffers(&device, omega, len_log as usize).unwrap();
    let (intt_omegas_buf, intt_pq_buf) =
        generate_ntt_buffers(&device, omega.invert().unwrap(), len_log as usize).unwrap();
    let divisor = Fr::from(1 << len_log).invert().unwrap();
    let divisor_buf = device
        .alloc_device_buffer_from_slice(&vec![divisor][..])
        .unwrap();

    for i in 0..4 {
        println!("test round {}", i);
        let random_nr = 256;
        let mut rands_s = vec![];
        for _ in 0..random_nr {
            let s = Fr::rand();
            rands_s.push(s);
        }

        #[cfg(feature = "hugetlb")]
        let mut s_vec = {
            let timer = start_timer!(|| "prepare buffer with hugetlb");
            let mut s_vec = Vec::new_in(unsafe { &HUGE_PAGE_ALLOCATOR });
            s_vec.reserve(len);
            for i in 0..len {
                let x = rands_s[i % random_nr];
                s_vec.push(x);
            }
            end_timer!(timer);
            s_vec
        };

        #[cfg(not(feature = "hugetlb"))]
        let mut s_vec = {
            let timer = start_timer!(|| "prepare buffer with");
            let mut s_vec = vec![];
            s_vec.reserve(len);
            for i in 0..len {
                let x = rands_s[i % random_nr];
                s_vec.push(x);
            }
            end_timer!(timer);
            s_vec
        };

        let timer = start_timer!(|| "st cpu cost");
        let mut expected_ntt_s = s_vec.clone();
        // println!("s is {:?}", expected_ntt_s);
        best_fft_cpu(&mut expected_ntt_s[..], omega, len_log);
        end_timer!(timer);

        let timer = start_timer!(|| "pin buffer");
        let s_origin = s_vec.clone();
        let s = &mut s_vec[..];
        device.pin_memory(s).unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "copy buffer");
        let mut a_buf = device.alloc_device_buffer_from_slice(s).unwrap();
        end_timer!(timer);
        let timer = start_timer!(|| "copy buffer");
        let mut b_buf = device.alloc_device_buffer_from_slice(s).unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "gpu costs");
        ntt_raw(
            &device,
            &mut a_buf,
            &mut b_buf,
            &pq_buf,
            &omegas_buf,
            len_log as usize,
            None,
            None,
        )
        .unwrap();
        device.synchronize().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "copy buffer back");
        device.copy_from_device_to_host(s, &a_buf).unwrap();
        end_timer!(timer);
        assert!(s == expected_ntt_s);

        let timer = start_timer!(|| "unpin buffer");
        device.unpin_memory(s).unwrap();
        end_timer!(timer);

        ntt_raw(
            &device,
            &mut a_buf,
            &mut b_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            len_log as usize,
            Some(&divisor_buf),
            None,
        )
        .unwrap();

        device.copy_from_device_to_host(s, &a_buf).unwrap();
        assert!(s == s_origin);
    }
}
