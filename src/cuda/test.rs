use super::bn254_c;
use crate::cuda::bn254::{intt_raw, ntt_raw};
use crate::device::cuda::{to_result, CudaBuffer as _, CudaDevice};
use crate::device::Device;
use ark_std::{end_timer, start_timer};
use halo2_proofs::arithmetic::{best_fft_cpu, BaseExt, CurveAffine, Field as _, FieldExt, Group};
use halo2_proofs::pairing::bn256::{Fq, Fr, G1Affine, G1};
use halo2_proofs::pairing::group::ff::PrimeField as _;
use halo2_proofs::pairing::group::Curve;
use rand::Rng;

fn batch_msm(p: &[G1Affine], s: &[&[Fr]], expect: Option<Vec<G1Affine>>) {
    const N: usize = 4;
    let device = CudaDevice::get_device(0).unwrap();
    let timer = start_timer!(|| "prepare batch msm");
    let mut tmp_buf = vec![];
    let mut s_buf = vec![];
    for _ in 0..N {
        tmp_buf.push(device.alloc_device_buffer::<Fq>((1 << 22) * 4).unwrap());
        s_buf.push(device.alloc_device_buffer::<Fr>(p.len()).unwrap());
    }
    end_timer!(timer);
    let p_buf = device.alloc_device_buffer_from_slice(&p[..]).unwrap();

    let mut streams = [None; N];
    let mut msm_results = vec![];

    let timer = start_timer!(|| "batch msm");
    for (i, s) in s.into_iter().enumerate() {
        unsafe {
            if streams[i % N].is_some() {
                cuda_runtime_sys::cudaStreamSynchronize(streams[i % N].unwrap());
                cuda_runtime_sys::cudaStreamDestroy(streams[i % N].unwrap());
                let mut res = [G1::group_zero()];
                device
                    .copy_from_device_to_host(&mut res[..], &tmp_buf[i % N])
                    .unwrap();
                let res = res[0].to_affine();
                let is_valid: bool = res.is_on_curve().into();
                assert!(is_valid);
                if expect.is_some() {
                    assert_eq!(res, expect.as_ref().unwrap()[i]);
                }
                msm_results.push(res);
                streams[i % N] = None;
            }

            let mut stream = std::mem::zeroed();
            cuda_runtime_sys::cudaStreamCreate(&mut stream);
            device
                .copy_from_host_to_device_async(&s_buf[i % N], &s[..], stream)
                .unwrap();
            let res = bn254_c::msm(
                tmp_buf[i % N].ptr(),
                p_buf.ptr(),
                s_buf[i % N].ptr(),
                p.len() as i32,
                stream,
            );
            to_result((), res, "batch msm").unwrap();
            streams[i % N] = Some(stream);
        }
    }

    for i in 0..N {
        if streams[i % N].is_some() {
            unsafe {
                cuda_runtime_sys::cudaStreamSynchronize(streams[i % N].unwrap());
                cuda_runtime_sys::cudaStreamDestroy(streams[i % N].unwrap());
            }
            if true {
                let mut res = [Fq::zero(); 4];
                device
                    .copy_from_device_to_host(&mut res[..], &tmp_buf[i % N])
                    .unwrap();

                let x = res[0];
                let y = res[1];
                let zz = res[2];
                let zzz_inv = res[3].invert().unwrap();
                let z_inv = zz * zzz_inv;
                let x = x * z_inv.square();
                let y = y * zzz_inv;

                let res = G1Affine::from_xy(x, y).unwrap();
                if expect.is_some() {
                    assert_eq!(res, expect.as_ref().unwrap()[i]);
                }
                msm_results.push(res);
            } else {
                let mut res = [G1::group_zero()];
                device
                    .copy_from_device_to_host(&mut res[..], &tmp_buf[i % N])
                    .unwrap();
                let res = res[0].to_affine();
                let is_valid: bool = res.is_on_curve().into();
                assert!(is_valid);
                if expect.is_some() {
                    assert_eq!(res, expect.as_ref().unwrap()[i]);
                }
                msm_results.push(res);
            }
            streams[i % N] = None;
        }
    }
    end_timer!(timer);
}

#[test]
fn test_bn254_msm() {
    let len = 1 << 22;

    for _ in 0..10 {
        let mut p = vec![];
        let mut s = vec![];

        //let timer = start_timer!(|| "prepare buffer");
        let random_nr = 1024;
        let mut rands_s = vec![];
        let mut rands_p = vec![];
        let mut rands_ps = vec![];

        for i in 0..random_nr {
            let s = if true {
                Fr::rand()
            } else {
                let mut rng = rand::thread_rng();
                let s: u64 = rng.gen();
                Fr::from(s)
            };

            rands_s.push(s);

            let ps = Fr::rand();
            rands_p.push((G1Affine::generator() * ps).to_affine());
            rands_ps.push(ps);
        }

        let mut acc = Fr::zero();
        for i in 0..len {
            let x = rands_p[i % random_nr];
            let y = rands_s[i % random_nr] + Fr::from(i as u64);
            p.push(x);
            s.push(y);
            acc += (rands_s[i % random_nr] + Fr::from(i as u64)) * rands_ps[i % random_nr];
        }
        //end_timer!(timer);

        //let timer = start_timer!(|| "cpu costs");
        let msm_res_expect = G1Affine::generator() * acc;
        //end_timer!(timer);

        batch_msm(
            &p[..],
            &[&s[..]; 1][..],
            Some(vec![msm_res_expect.to_affine(); 1]),
        );
    }
}

#[test]
fn test_bn254_fft() {
    let device = CudaDevice::get_device(0).unwrap();
    let len_log = 20;
    let len = 1 << len_log;

    let mut omega = Fr::ROOT_OF_UNITY_INV.invert().unwrap();
    for _ in len_log..Fr::S {
        omega = omega.square();
    }

    let (omegas_buf, pq_buf) = super::bn254::ntt_prepare(&device, omega, len_log as usize).unwrap();
    let (intt_omegas_buf, intt_pq_buf) =
        super::bn254::ntt_prepare(&device, omega.invert().unwrap(), len_log as usize).unwrap();
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

        intt_raw(
            &device,
            &mut a_buf,
            &mut b_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &divisor_buf,
            len_log as usize,
        )
        .unwrap();

        device.copy_from_device_to_host(s, &a_buf).unwrap();
        assert!(s == s_origin);
    }
}
