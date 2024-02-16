use crate::device::cuda::{to_result, CudaBuffer, CudaDevice, CudaDeviceBufRaw};
use crate::device::Error;
use crate::device::{Device, DeviceResult};
use halo2_proofs::arithmetic::{CurveAffine, FieldExt};
use halo2_proofs::pairing::group::Curve;
use halo2_proofs::pairing::group::Group;

mod cuda_c {
    use cuda_runtime_sys::cudaError;
    use std::ffi::c_void;
    #[link(name = "zkwasm_prover_kernel", kind = "static")]
    extern "C" {
        pub fn msm(
            blocks: i32,
            threads: i32,
            res: *mut c_void,
            p: *mut c_void,
            s: *mut c_void,
            array_len: i32,
        ) -> cudaError;

        pub fn ntt(
            buf: *mut c_void,
            tmp: *mut c_void,
            pq: *mut c_void,
            omega: *mut c_void,
            array_log: i32,
            max_deg: i32,
            swap: *mut c_void,
        ) -> cudaError;

        pub fn field_op(
            res: *mut c_void,
            l: *mut c_void,
            l_rot: i32,
            l_c: *mut c_void,
            r: *mut c_void,
            r_rot: i32,
            r_c: *mut c_void,
            size: i32,
            op: i32,
        ) -> cudaError;

        pub fn extended_prepare(
            s: *mut c_void,
            coset_powers: *mut c_void,
            coset_powers_n: i32,
            size: i32,
            extended_size: i32,
        ) -> cudaError;
    }
}

pub(crate) fn extended_prepare(
    device: &CudaDevice,
    s: &CudaDeviceBufRaw,
    coset_powers: &CudaDeviceBufRaw,
    coset_powers_n: usize,
    size: usize,
    extended_size: usize,
) -> Result<(), Error> {
    unsafe {
        device.acitve_ctx()?;
        let err = cuda_c::extended_prepare(
            s.ptr(),
            coset_powers.ptr(),
            coset_powers_n as i32,
            size as i32,
            extended_size as i32,
        );
        to_result((), err, "fail to run extended_prepare")?;
        Ok(())
    }
}

pub(crate) enum FieldOp {
    Sum = 0,
    Mul = 1,
    Neg = 2,
}

pub(crate) fn field_op<F: FieldExt>(
    device: &CudaDevice,
    res: &CudaDeviceBufRaw,
    l: Option<&CudaDeviceBufRaw>,
    l_rot: i32,
    l_c: Option<F>,
    r: Option<&CudaDeviceBufRaw>,
    r_rot: i32,
    r_c: Option<F>,
    size: usize,
    op: FieldOp,
) -> Result<(), Error> {
    let l_c = if l_c.is_none() {
        0usize as *mut _
    } else {
        device
            .alloc_device_buffer_from_slice([l_c.unwrap()].as_slice())?
            .ptr()
    };
    let r_c = if r_c.is_none() {
        0usize as *mut _
    } else {
        device
            .alloc_device_buffer_from_slice([r_c.unwrap()].as_slice())?
            .ptr()
    };
    unsafe {
        device.acitve_ctx()?;
        let err = cuda_c::field_op(
            res.ptr(),
            l.map_or(0usize as *mut _, |x| x.ptr()),
            l_rot,
            l_c,
            r.map_or(0usize as *mut _, |x| x.ptr()),
            r_rot,
            r_c,
            size as i32,
            op as i32,
        );
        to_result((), err, "fail to run field_op")?;
        Ok(())
    }
}

pub fn msm<C: CurveAffine>(
    device: &CudaDevice,
    p_buf: &CudaDeviceBufRaw,
    s_buf: &CudaDeviceBufRaw,
    len: usize,
) -> Result<C, Error> {
    msm_with_groups(device, p_buf, s_buf, len, 4)
}

pub fn msm_with_groups<C: CurveAffine>(
    device: &CudaDevice,
    p_buf: &CudaDeviceBufRaw,
    s_buf: &CudaDeviceBufRaw,
    len: usize,
    msm_groups: usize,
) -> Result<C, Error> {
    let threads = 128;
    let windows = 32;
    let windows_bits = 8;
    let mut tmp = vec![C::Curve::identity(); msm_groups * windows];
    let res_buf = device.alloc_device_buffer_from_slice(&tmp[..])?;
    unsafe {
        device.acitve_ctx()?;
        let err = cuda_c::msm(
            msm_groups as i32,
            threads,
            res_buf.ptr(),
            p_buf.ptr(),
            s_buf.ptr(),
            len as i32,
        );
        to_result((), err, "fail to run msm")?;
    }
    device.copy_from_device_to_host(&mut tmp[..], &res_buf)?;

    for i in 0..windows {
        for j in 1..msm_groups {
            tmp[i] = tmp[i] + tmp[i + j * windows];
        }
    }

    let mut msm_res = tmp[windows - 1];
    for i in 0..windows - 1 {
        for _ in 0..windows_bits {
            msm_res = msm_res + msm_res;
        }
        msm_res = msm_res + tmp[windows - 2 - i];
    }

    Ok(msm_res.to_affine())
}

const MAX_DEG: usize = 8;

pub fn ntt_prepare<F: FieldExt>(
    device: &CudaDevice,
    len_log: usize,
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw)> {
    let mut omega = F::ROOT_OF_UNITY_INV.invert().unwrap();
    for _ in len_log..F::S as usize {
        omega = omega.square();
    }

    let len = 1 << len_log;
    let mut omegas = vec![F::one()];
    for _ in 1..len {
        omegas.push(*omegas.last().unwrap() * omega);
    }

    let max_deg = MAX_DEG.min(len_log);
    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
    let twiddle = omega.pow_vartime([(len >> max_deg) as u64]);
    pq[0] = F::one();
    if max_deg > 1 {
        pq[1] = twiddle;
        for i in 2..(1 << max_deg >> 1) {
            pq[i] = pq[i - 1];
            pq[i].mul_assign(&twiddle);
        }
    }

    let omegas_buf = device.alloc_device_buffer_from_slice(&omegas[..])?;
    let pq_buf = device.alloc_device_buffer_from_slice(&pq[..])?;

    Ok((omegas_buf, pq_buf))
}

pub fn ntt_raw(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    len_log: usize,
) -> Result<(), Error> {
    let mut swap = false;
    unsafe {
        let err = crate::cuda::bn254::cuda_c::ntt(
            s_buf.ptr(),
            tmp_buf.ptr(),
            pq_buf.ptr(),
            omegas_buf.ptr(),
            len_log as i32,
            MAX_DEG as i32,
            &mut swap as *mut _ as _,
        );
        to_result((), err, "fail to run ntt")?;
    }
    if swap {
        std::mem::swap(s_buf, tmp_buf);
    }
    Ok(())
}

pub fn ntt<F: FieldExt>(
    device: &CudaDevice,
    s_buf: &mut CudaDeviceBufRaw,
    tmp_buf: &mut CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    result: &mut [F],
    len_log: usize,
) -> Result<(), Error> {
    ntt_raw(device, s_buf, tmp_buf, pq_buf, omegas_buf, len_log)?;
    device.copy_from_device_to_host(result, s_buf)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::ffi::c_void;
    use std::mem;
    use std::ops::MulAssign as _;

    use crate::device::cuda::{to_result, CudaBuffer as _, CudaDevice};
    use crate::device::{Device, DeviceBuf};
    use ark_std::rand::rngs::OsRng;
    use ark_std::{end_timer, start_timer};
    use cuda_runtime_sys::cudaError;
    use halo2_proofs::arithmetic::{
        best_fft_cpu, BaseExt, CurveAffine, Field as _, FieldExt, Group,
    };
    use halo2_proofs::pairing::bn256::{Fr, G1Affine, G1};
    use halo2_proofs::pairing::group::ff::PrimeField as _;
    use halo2_proofs::pairing::group::{Curve, Group as _};

    use super::MAX_DEG;

    #[link(name = "zkwasm_prover_kernel", kind = "static")]
    extern "C" {
        #[cfg(features = "full-test")]
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

        #[cfg(features = "full-test")]
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

    #[cfg(features = "full-test")]
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
                a_buf.ptr(),
                b_buf.ptr(),
                exp_buf.ptr(),
                add_buf.ptr(),
                sub_buf.ptr(),
                mul_buf.ptr(),
                sqr_buf.ptr(),
                inv_buf.ptr(),
                pow_buf.ptr(),
                unmont_buf.ptr(),
                compare_buf.ptr(),
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

    #[cfg(features = "full-test")]
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
                a_buf.ptr(),
                b_buf.ptr(),
                exp_buf.ptr(),
                add_buf.ptr(),
                sub_buf.ptr(),
                mul_buf.ptr(),
                sqr_buf.ptr(),
                inv_buf.ptr(),
                pow_buf.ptr(),
                unmont_buf.ptr(),
                compare_buf.ptr(),
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

        let x = G1Affine::generator();
        let y = G1Affine::generator();
        a.push(x);
        b.push(y);

        for _ in 1..len {
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
                a_buf.ptr(),
                b_buf.ptr(),
                add_buf.ptr(),
                sub_buf.ptr(),
                double_buf.ptr(),
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
                let a_buf = device.alloc_device_buffer_from_slice(&p[..]).unwrap();
                let b_buf = device.alloc_device_buffer_from_slice(&s[..]).unwrap();
                end_timer!(timer);

                let timer = start_timer!(|| "gpu costs********");
                let res = crate::cuda::bn254::cuda_c::msm(
                    msm_groups as i32,
                    threads,
                    tmp_buf.ptr(),
                    a_buf.ptr(),
                    b_buf.ptr(),
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

        /*
        let mut omegas = vec![omega];
        for _ in 1..32 {
            omegas.push(omegas.last().unwrap().square());
        }
        */
        let mut omegas = vec![Fr::one()];
        for _ in 1..len {
            omegas.push(omegas.last().unwrap() * omega);
        }

        let max_deg = MAX_DEG.min(len_log as usize);
        let mut pq = vec![Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow_vartime([(len >> max_deg) as u64]);
        pq[0] = Fr::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }

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
            let s = &mut s_vec[..];
            device.pin_memory(s).unwrap();
            end_timer!(timer);

            unsafe {
                let timer = start_timer!(|| "copy buffer");
                let a_buf = device.alloc_device_buffer_from_slice(s).unwrap();
                end_timer!(timer);
                let timer = start_timer!(|| "copy buffer");
                let b_buf = device.alloc_device_buffer_from_slice(s).unwrap();
                end_timer!(timer);
                let timer = start_timer!(|| "copy buffer");
                let omegas_buf = device.alloc_device_buffer_from_slice(&omegas[..]).unwrap();
                end_timer!(timer);
                let timer = start_timer!(|| "copy buffer");
                let pq_buf = device.alloc_device_buffer_from_slice(&pq[..]).unwrap();
                end_timer!(timer);

                let mut swap = false;
                let timer = start_timer!(|| "gpu costs");
                let res = crate::cuda::bn254::cuda_c::ntt(
                    a_buf.ptr(),
                    b_buf.ptr(),
                    pq_buf.ptr(),
                    omegas_buf.ptr(),
                    len_log as i32,
                    max_deg as i32,
                    &mut swap as *mut _ as _,
                );
                assert_eq!(res, cudaError::cudaSuccess);
                device.synchronize().unwrap();
                end_timer!(timer);

                let timer = start_timer!(|| "copy buffer back");
                if swap {
                    device.copy_from_device_to_host(s, &b_buf).unwrap();
                } else {
                    device.copy_from_device_to_host(s, &a_buf).unwrap();
                }
                end_timer!(timer);
                assert!(s == expected_ntt_s);
            }

            let timer = start_timer!(|| "unpin buffer");
            device.unpin_memory(s).unwrap();
            end_timer!(timer);
        }
    }
}
