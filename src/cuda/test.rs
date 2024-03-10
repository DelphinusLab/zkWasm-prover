use super::bn254_c;
use crate::cuda::bn254::{intt_raw, ntt_raw};
use crate::device::cuda::{CudaBuffer as _, CudaDevice};
use crate::device::Device;
use ark_std::{end_timer, start_timer};
use cuda_runtime_sys::cudaError;
use halo2_proofs::arithmetic::{best_fft_cpu, BaseExt, Field as _, FieldExt, Group};
use halo2_proofs::pairing::bn256::{Fr, G1Affine, G1};
use halo2_proofs::pairing::group::cofactor::CofactorCurveAffine;
use halo2_proofs::pairing::group::ff::PrimeField as _;
use halo2_proofs::pairing::group::Curve;
use icicle_cuda_runtime::stream::CudaStream;

#[test]
fn test_bn254_msm() {
    let device = CudaDevice::get_device(0).unwrap();
    let len = 1 << 20;

    for _ in 0..10 {
        let mut p = vec![];
        let mut s = vec![];

        let timer = start_timer!(|| "prepare buffer");
        let random_nr = 1;
        let mut rands_s = vec![];
        let mut rands_p = vec![];
        let mut rands_ps = vec![];
        for _ in 0..random_nr {
            let s = Fr::rand();
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
        end_timer!(timer);

        let timer = start_timer!(|| "cpu costs");
        let msm_res_expect = G1Affine::generator() * acc;
        end_timer!(timer);

        let windows = 32;
        let windows_bits = 8;
        let msm_groups = 4;
        let threads = 128;
        let mut tmp = vec![];
        for _ in 0..windows * msm_groups {
            tmp.push(G1::group_zero());
        }

        unsafe {
            let timer = start_timer!(|| "copy buffer");
            let tmp_buf = device.alloc_device_buffer_from_slice(&tmp[..]).unwrap();
            let p_buf = device.alloc_device_buffer_from_slice(&p[..]).unwrap();
            let s_buf = device.alloc_device_buffer_from_slice(&s[..]).unwrap();
            end_timer!(timer);

            let timer = start_timer!(|| "gpu costs********");
            let res = bn254_c::msm(
                msm_groups as i32,
                threads,
                tmp_buf.ptr(),
                p_buf.ptr(),
                s_buf.ptr(),
                len as i32,
            );
            device.synchronize().unwrap();
            assert_eq!(res, cudaError::cudaSuccess);
            end_timer!(timer);

            let timer = start_timer!(|| "copy buffer back");
            device
                .copy_from_device_to_host(&mut tmp[..], &tmp_buf)
                .unwrap();
            end_timer!(timer);

            for i in 0..windows {
                for j in 1..msm_groups {
                    tmp[i] = tmp[i] + tmp[i + j * windows];
                }
            }

            let timer = start_timer!(|| "gpu msm merge");
            let mut msm_res = tmp[windows - 1];
            for i in 0..windows - 1 {
                for _ in 0..windows_bits {
                    msm_res = msm_res + msm_res;
                }
                msm_res = msm_res + tmp[windows - 2 - i];
            }
            end_timer!(timer);
            assert_eq!(msm_res.to_affine(), msm_res_expect.to_affine());

            {
                use icicle_bn254::curve::CurveCfg;
                use icicle_bn254::curve::G1Projective;
                use icicle_bn254::curve::ScalarField;
                use icicle_core::curve::Affine;
                use icicle_core::curve::Curve;
                use icicle_core::msm;
                use icicle_core::traits::FieldImpl;
                use icicle_cuda_runtime::memory::HostOrDeviceSlice;

                let mut msm_host_result = vec![G1Projective::zero(); 1];
                let mut msm_host_affine_result = vec![icicle_bn254::curve::G1Affine::zero(); 1];

                let points = HostOrDeviceSlice::Host(
                    std::mem::transmute::<_, &[Affine<_>]>(&p[..len]).to_vec(),
                );
                let scalars = HostOrDeviceSlice::Host(
                    std::mem::transmute::<_, &[ScalarField]>(&s[..len]).to_vec(),
                );
                let mut msm_results: HostOrDeviceSlice<'_, G1Projective> =
                    HostOrDeviceSlice::cuda_malloc(1).unwrap();
                let mut cfg = msm::MSMConfig::default();
                let stream = CudaStream::create().unwrap();
                cfg.ctx.stream = &stream;
                cfg.is_async = true;
                cfg.are_scalars_montgomery_form = true;
                cfg.are_points_montgomery_form = true;
                msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
                stream.synchronize().unwrap();
                msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
                let _res = [CurveCfg::to_affine(
                    &mut msm_host_result[0],
                    &mut msm_host_affine_result[0],
                )];
                let mut res = G1Affine::identity();
                res.x = halo2_proofs::pairing::bn256::Fq::from_repr(
                    msm_host_affine_result[0]
                        .x
                        .to_bytes_le()
                        .try_into()
                        .unwrap(),
                )
                .unwrap();
                res.y = halo2_proofs::pairing::bn256::Fq::from_repr(
                    msm_host_affine_result[0]
                        .y
                        .to_bytes_le()
                        .try_into()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(res, msm_res_expect.to_affine());
            }
        }
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
