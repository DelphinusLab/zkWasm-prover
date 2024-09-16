use std::collections::{HashMap, HashSet};
use std::mem::ManuallyDrop;

use crate::cuda::ntt::ntt_raw;
use crate::device::cuda::{CudaBuffer, CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper};
use crate::device::Error;
use crate::device::{Device, DeviceResult};
use crate::ADD_RANDOM;

use halo2_proofs::arithmetic::{CurveAffine, FieldExt};
use icicle_bn254::curve::BaseField;
use icicle_bn254::curve::CurveCfg;
use icicle_bn254::curve::G1Projective;
use icicle_bn254::curve::ScalarCfg;
use icicle_core::curve::Projective;
use icicle_core::field::Field;
use icicle_core::msm;
use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::stream::CudaStream;

pub(crate) trait CopyToCudaHost<'a>: Sized {
    type F;

    fn to_icicle_device_buf(
        &self,
        _: &CudaDevice,
        _: usize,
        _: &CudaStream,
    ) -> ManuallyDrop<HostOrDeviceSlice<'a, Field<4, ScalarCfg>>> {
        unreachable!()
    }

    fn to_cuda_device_buffer(
        &self,
        _: &CudaDevice,
        _: usize,
        _: &CudaStream,
    ) -> DeviceResult<Option<CudaDeviceBufRaw>> {
        Ok(None)
    }

    fn to_cuda_device_buffer_ref(
        &self,
        _: &CudaDevice,
        _: usize,
        _: &CudaStream,
    ) -> &'a CudaDeviceBufRaw {
        unimplemented!()
    }

    fn to_icicle_host_slice(&self) -> &'a [Field<4, ScalarCfg>];

    fn to_mut_slice(self) -> &'a mut [Self::F] {
        unreachable!()
    }
}

impl<'a> CopyToCudaHost<'a> for &'a CudaDeviceBufRaw {
    type F = ();

    fn to_icicle_host_slice(&self) -> &'a [Field<4, ScalarCfg>] {
        unreachable!()
    }

    fn to_icicle_device_buf(
        &self,
        _: &CudaDevice,
        len: usize,
        _: &CudaStream,
    ) -> ManuallyDrop<HostOrDeviceSlice<'a, Field<4, ScalarCfg>>> {
        unsafe {
            ManuallyDrop::new(HostOrDeviceSlice::Device(
                std::slice::from_raw_parts_mut(self.ptr() as _, len),
                0,
            ))
        }
    }

    fn to_cuda_device_buffer_ref(
        &self,
        _: &CudaDevice,
        _: usize,
        _: &CudaStream,
    ) -> &'a CudaDeviceBufRaw {
        self
    }
}

impl<'a, F: FieldExt> CopyToCudaHost<'a> for &'a [F] {
    type F = F;

    fn to_cuda_device_buffer(
        &self,
        device: &CudaDevice,
        _: usize,
        _stream: &CudaStream,
    ) -> DeviceResult<Option<CudaDeviceBufRaw>> {
        let buf = device.alloc_device_buffer_from_slice::<F>(self)?;
        Ok(Some(buf))
    }

    fn to_icicle_host_slice(&self) -> &'a [Field<4, ScalarCfg>] {
        unsafe { core::mem::transmute::<_, _>(*self) }
    }
}

impl<'a, F: FieldExt> CopyToCudaHost<'a> for &'a mut [F] {
    type F = F;

    fn to_cuda_device_buffer(
        &self,
        device: &CudaDevice,
        _: usize,
        _stream: &CudaStream,
    ) -> DeviceResult<Option<CudaDeviceBufRaw>> {
        let buf = device.alloc_device_buffer_from_slice::<F>(self)?;
        Ok(Some(buf))
    }

    fn to_icicle_host_slice(&self) -> &'a [Field<4, ScalarCfg>] {
        unsafe { core::mem::transmute::<_, _>(&**self) }
    }

    fn to_mut_slice(self) -> &'a mut [F] {
        &mut *self
    }
}

// msm sometimes return bad point, retry to make it correct
fn to_affine<C: CurveAffine>(g: &icicle_bn254::curve::G1Projective) -> Option<C> {
    if g.z == BaseField::zero() {
        Some(C::identity())
    } else {
        use halo2_proofs::arithmetic::BaseExt;
        use halo2_proofs::arithmetic::Field;

        let mut t: Vec<_> = g.x.to_bytes_le();
        t.resize(64, 0u8);
        let x = C::Base::from_bytes_wide(&t.try_into().unwrap());

        let mut t: Vec<_> = g.y.to_bytes_le();
        t.resize(64, 0u8);
        let y = C::Base::from_bytes_wide(&t.try_into().unwrap());

        let mut t: Vec<_> = g.z.to_bytes_le();
        t.resize(64, 0u8);
        let z = C::Base::from_bytes_wide(&t.try_into().unwrap());

        let z_inv = z.invert().unwrap();
        C::from_xy(x * z_inv, y * z_inv).into()
    }
}

fn copy_and_to_affine<C: CurveAffine>(
    msm_result: &HostOrDeviceSlice<'_, Projective<CurveCfg>>,
) -> DeviceResult<C> {
    let retry_limit = 3;

    for i in 0..retry_limit {
        let mut msm_host_result = [G1Projective::zero()];
        msm_result.copy_to_host(&mut msm_host_result[..]).unwrap();

        let res = to_affine(&msm_host_result[0]);
        if res.is_some() {
            if !ADD_RANDOM {
                println!("commitment is {:?}", res);
            }
            return Ok(res.unwrap());
        }

        println!("bad msm result at round {} is {:?}", i, msm_host_result);
    }

    Err(Error::MsmError)
}

pub(crate) fn batch_msm_pure<'a, C: CurveAffine, T: CopyToCudaHost<'a> + 'a>(
    device: &CudaDevice,
    points_buf: &CudaDeviceBufRaw,
    scalars_buf: &mut [T],
    len: usize,
) -> DeviceResult<(Vec<C>, HashMap<usize, CudaDeviceBufRaw>)> {
    device.synchronize()?;

    const MSM_STREAMS_NR: usize = 4;

    let streams = [0; MSM_STREAMS_NR].map(|_| CudaStream::create().unwrap());
    let mut msm_results_buf: Vec<HostOrDeviceSlice<'_, Projective<_>>> = scalars_buf
        .iter()
        .map(|_| HostOrDeviceSlice::cuda_malloc(1).unwrap())
        .collect::<Vec<_>>();

    let points = {
        unsafe {
            ManuallyDrop::new(HostOrDeviceSlice::Device(
                std::slice::from_raw_parts_mut(points_buf.ptr() as _, len),
                0,
            ))
        }
    };

    let mut s_buf_queue = HashMap::new();

    for (idx, value) in scalars_buf.into_iter().enumerate() {
        let inner_idx = idx % MSM_STREAMS_NR;
        let stream = &streams[inner_idx];

        stream.synchronize().unwrap();
        s_buf_queue.remove(&inner_idx);

        let cuda_device_buffer = value.to_cuda_device_buffer(device, len, stream)?;

        // MSM
        {
            let scalars = if let Some(buffer) = cuda_device_buffer {
                s_buf_queue.insert(inner_idx, buffer);
                let cuda_buf = s_buf_queue.get(&inner_idx).unwrap();
                let mut buf = cuda_buf.to_icicle_device_buf(device, len, stream);

                buf.copy_from_host(value.to_icicle_host_slice()).unwrap();
                buf
            } else {
                value.to_icicle_device_buf(device, len, stream)
            };
            let mut cfg = msm::MSMConfig::default();
            cfg.ctx.stream = stream;
            cfg.is_async = true;
            cfg.are_scalars_montgomery_form = true;
            cfg.are_points_montgomery_form = true;
            msm::msm(&scalars, &points, &cfg, &mut msm_results_buf[idx]).unwrap();
        }
    }

    for stream in streams {
        stream.synchronize().unwrap();
    }

    let res_vec = msm_results_buf
        .into_iter()
        .map(|x| copy_and_to_affine(&x))
        .collect::<DeviceResult<_>>()?;

    Ok((res_vec, HashMap::new()))
}

/*
pub(crate) fn batch_msm_pure<'a, C: CurveAffine>(
    points_buf: &'a [C],
    scalars_buf: Vec<&'a [C::Scalar]>,
    len: usize,
) -> Result<Vec<C>, Error> {
    let mut res_vec = vec![];

    for (_, value) in scalars_buf.into_iter().enumerate() {
        assert_eq!(points_buf.len(), value.len());
        let stream = CudaStream::create().unwrap();
        let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
        let points = HostSlice::from_slice(unsafe { std::mem::transmute(&points_buf[..len]) });
        let scalars = HostSlice::from_slice(unsafe { std::mem::transmute(&value[..len]) });
        let mut cfg = msm::MSMConfig::default();
        cfg.ctx.stream = &stream;
        cfg.is_async = true;
        cfg.are_scalars_montgomery_form = true;
        cfg.are_points_montgomery_form = true;
        msm::msm(scalars, points, &cfg, &mut msm_results[..]).unwrap();
        stream.synchronize().unwrap();
        res_vec.push(copy_and_to_affine(&msm_results).unwrap());
    }

    Ok(res_vec)
}
 */

pub(crate) fn batch_msm_pure_robust<'a, C: CurveAffine, T: CopyToCudaHost<'a> + 'a>(
    device: &CudaDevice,
    points_buf: &CudaDeviceBufRaw,
    scalars_buf: &mut [T],
    len: usize,
) -> Result<(Vec<C>, HashMap<usize, CudaDeviceBufRaw>), Error> {
    for _ in 0..100 {
        let res = batch_msm_pure(device, points_buf, &mut *scalars_buf, len);

        if res.is_ok() {
            return res;
        }
    }

    unreachable!()
}

pub(crate) fn batch_msm<'a, C: CurveAffine, T: CopyToCudaHost<'a> + 'a>(
    device: &CudaDevice,
    points_buf: &CudaDeviceBufRaw,
    mut scalars_buf: Vec<T>,
    // pq_buf, omegas_buf, divisor_buf, len_log, intt_map, skips
    intt_args: Option<(
        &CudaDeviceBufRaw,
        &CudaDeviceBufRaw,
        &CudaDeviceBufRaw,
        usize,
        Option<&HashSet<usize>>,
        usize,
    )>,
    len: usize,
) -> Result<(Vec<C>, HashMap<usize, CudaDeviceBufRaw>), Error> {
    let res = batch_msm_pure_robust(device, points_buf, &mut scalars_buf[..], len)?;

    // INTT
    let mut buffer_map = HashMap::new();

    if let Some((pq_buf, omegas_buf, divisor_buf, len_log, intt_map, skips)) = intt_args {
        const MSM_STREAMS_NR: usize = 4;
        let streams = [0; MSM_STREAMS_NR].map(|_| CudaStream::create().unwrap());

        let mut s_buf_queue = HashMap::new();
        let mut t_buf_queue = HashMap::new();

        for (idx, value) in scalars_buf.into_iter().enumerate() {
            let inner_idx = idx % MSM_STREAMS_NR;
            let stream = &streams[inner_idx];

            stream.synchronize().unwrap();
            s_buf_queue.remove(&inner_idx);
            t_buf_queue.remove(&inner_idx);

            let mut s_buf = value.to_cuda_device_buffer(device, len, stream)?.unwrap();
            if idx >= skips {
                if intt_map.map(|m| m.contains(&(idx - skips))).unwrap_or(true) {
                    let stream =
                        unsafe { std::mem::transmute::<&CudaStream, &CudaStreamWrapper>(stream) };
                    let mut t_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?;
                    ntt_raw(
                        device,
                        &mut s_buf,
                        &mut t_buf,
                        pq_buf,
                        omegas_buf,
                        len_log,
                        Some(divisor_buf),
                        Some(stream),
                    )?;

                    device.copy_from_device_to_host_async(
                        value.to_mut_slice(),
                        &s_buf,
                        stream.into(),
                    )?;
                    s_buf_queue.insert(inner_idx, s_buf);
                    t_buf_queue.insert(inner_idx, t_buf);
                } else {
                    buffer_map.insert(idx - skips, s_buf);
                }
            }
        }
    }

    Ok((res.0, buffer_map))
}

// keep for vanishment
pub fn batch_msm_v2<C: CurveAffine>(
    p_buf: &CudaDeviceBufRaw,
    values: Vec<&CudaDeviceBufRaw>,
    len: usize,
) -> Result<Vec<C>, Error> {
    for _ in 0..100 {
        let res = batch_msm_core_v2(p_buf, values.clone(), len);

        if res.is_ok() {
            return res;
        }
    }

    unreachable!()
}

fn batch_msm_core_v2<C: CurveAffine>(
    p_buf: &CudaDeviceBufRaw,
    values: Vec<&CudaDeviceBufRaw>,
    len: usize,
) -> Result<Vec<C>, Error> {
    unsafe {
        cuda_runtime_sys::cudaDeviceSynchronize();
    }

    const STREAMS_NR: usize = 1;
    let streams = [0; STREAMS_NR].map(|_| CudaStream::create().unwrap());
    let mut msm_results_buf = values
        .iter()
        .map(|_| HostOrDeviceSlice::cuda_malloc(1).unwrap())
        .collect::<Vec<_>>();

    let points = {
        unsafe {
            ManuallyDrop::new(HostOrDeviceSlice::Device(
                std::slice::from_raw_parts_mut(
                    p_buf.ptr() as *mut icicle_bn254::curve::G1Affine,
                    len,
                ),
                0,
            ))
        }
    };

    for (idx, value) in values.into_iter().enumerate() {
        let scalars = {
            unsafe {
                ManuallyDrop::new(HostOrDeviceSlice::Device(
                    std::slice::from_raw_parts_mut(value.ptr() as _, len),
                    0,
                ))
            }
        };
        let stream = &streams[idx % STREAMS_NR];
        let mut cfg = msm::MSMConfig::default();
        cfg.ctx.stream = &stream;
        cfg.is_async = true;
        cfg.are_scalars_montgomery_form = true;
        cfg.are_points_montgomery_form = true;
        msm::msm(&scalars, &points, &cfg, &mut msm_results_buf[idx]).unwrap();
    }

    for stream in streams {
        stream.synchronize().unwrap();
    }

    let res_vec = msm_results_buf
        .into_iter()
        .map(|x| copy_and_to_affine(&x).unwrap())
        .collect();

    Ok(res_vec)
}
