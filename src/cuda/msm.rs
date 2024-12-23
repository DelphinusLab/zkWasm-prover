use std::collections::HashMap;
use std::mem;

use ark_std::log2;
use cuda_runtime_sys::cudaError;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::pairing::group::Curve;

use crate::cuda::bn254_c::batch_msm_collect;
use crate::cuda::bn254_c::msm;
use crate::cuda::ntt::ntt_raw;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::{CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper};
use crate::device::Device;
use crate::device::DeviceResult;

pub(crate) trait ToDevBuffer {
    fn to_dev_buf<'a>(
        &'a self,
        device: &CudaDevice,
        buf: &'a CudaDeviceBufRaw,
        sw: &CudaStreamWrapper,
    ) -> DeviceResult<&'a CudaDeviceBufRaw>;
}

impl ToDevBuffer for &CudaDeviceBufRaw {
    fn to_dev_buf<'a>(
        &'a self,
        _: &CudaDevice,
        _: &'a CudaDeviceBufRaw,
        _: &CudaStreamWrapper,
    ) -> DeviceResult<&'a CudaDeviceBufRaw> {
        Ok(self)
    }
}

impl<T> ToDevBuffer for &[T] {
    fn to_dev_buf<'a>(
        &'a self,
        device: &CudaDevice,
        buf: &'a CudaDeviceBufRaw,
        sw: &CudaStreamWrapper,
    ) -> DeviceResult<&'a CudaDeviceBufRaw> {
        device
            .copy_from_host_to_device_async(buf, self, sw.into())
            .unwrap();
        Ok(buf)
    }
}

impl<T> ToDevBuffer for &Vec<T> {
    fn to_dev_buf<'a>(
        &'a self,
        device: &CudaDevice,
        buf: &'a CudaDeviceBufRaw,
        sw: &CudaStreamWrapper,
    ) -> DeviceResult<&'a CudaDeviceBufRaw> {
        device
            .copy_from_host_to_device_async(buf, &self[..], sw.into())
            .unwrap();
        Ok(buf)
    }
}

pub(crate) struct InttArgs<'a> {
    pub(crate) pq_buf: &'a CudaDeviceBufRaw,
    pub(crate) omegas_buf: &'a CudaDeviceBufRaw,
    pub(crate) divisor_buf: &'a CudaDeviceBufRaw,
    pub(crate) len_log: usize,
    pub(crate) selector: &'a dyn Fn(usize) -> bool,
}

pub(crate) fn batch_msm_and_intt_ext<'a, C: CurveAffine>(
    device: &CudaDevice,
    points_dev_buf: &CudaDeviceBufRaw,
    mut scalar_buf: Vec<&mut [C::Scalar]>,
    intt_args: InttArgs<'a>,
    cache_buffer_selector: &'a dyn Fn(usize) -> bool,
    before_final_round: &'a mut dyn FnMut() -> (),
    len: usize,
    skip_zero: bool,
) -> DeviceResult<(Vec<C>, HashMap<usize, CudaDeviceBufRaw>)> {
    if scalar_buf.len() == 0 {
        before_final_round();
        return Ok((vec![], HashMap::new()));
    }

    let threads = 64;
    let bits = 254;

    let msm_count = scalar_buf.len();

    // Allocated at first to make device memory more continuous.
    let mut cache_bufs = vec![];
    for i in 0..msm_count {
        if (cache_buffer_selector)(i) {
            let buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?;
            cache_bufs.push(buf);
        }
    }

    let mut cached_buffer = HashMap::new();

    // k22, 8, 13bits is best for RTX4090
    let window_bits = log2(msm_count).min(3) as usize + 10 + log2(len).max(22) as usize - 22;
    let windows = (bits + window_bits - 1) / window_bits;
    let bucket_size = windows << window_bits;
    let max_worker = 128 * 512;
    let worker = if len < max_worker {
        (len + threads - 1) / threads * threads
    } else {
        max_worker
    };

    let streams = [
        CudaStreamWrapper::new_with_inner_and_priority(1),
        CudaStreamWrapper::new_with_inner_and_priority(1),
    ];

    let mut scalar_dev_bufs = [
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
    ];

    let mut intt_tmp_bufs = [
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
    ];

    // About 30MB per MSM
    // layout: | bucktes | msm worker remain | collect worker remain |
    let curve_buf_size = bucket_size + worker * 2;
    let curve_buf =
        device.alloc_device_buffer_non_zeroed::<C::Curve>(msm_count * curve_buf_size)?;
    let single_sort_indices_size = len * windows;
    let sort_temp_storage_size = single_sort_indices_size * 3; // ~2N = 2 * windows * len, pick 32 as upper bound

    let sort_indices_temp_storage_buf =
        device.alloc_device_buffer_non_zeroed::<u32>(sort_temp_storage_size)?;

    let indices_buf = [0; 4].map(|_| {
        device
            .alloc_device_buffer_non_zeroed::<u32>(single_sort_indices_size)
            .unwrap()
    });

    let acc_indices_buf = (0..msm_count)
        .into_iter()
        .map(|_| device.alloc_device_buffer_non_zeroed::<u32>(worker * 2))
        .collect::<DeviceResult<Vec<_>>>()?;

    let mut pending_copy_queue = vec![];

    for i in 0..msm_count {
        let idx = i & 1;
        let scalar_dev_buf = &scalar_dev_bufs[idx];
        let stream = &streams[idx];
        let last_stream = &streams[1 - idx];
        stream.0.sync();

        device.copy_from_host_to_device_async(scalar_dev_buf, &scalar_buf[i][..], stream.1)?;
        last_stream.0.sync(); // sync to reuse sort_indices_buf

        let scalar_dev_buf = &scalar_dev_bufs[idx];
        unsafe {
            let err = msm(
                (curve_buf.ptr() as usize + i * curve_buf_size * std::mem::size_of::<C::Curve>())
                    as _,
                points_dev_buf.ptr(),
                scalar_dev_buf.ptr(),
                indices_buf[0].ptr(),
                indices_buf[1].ptr(),
                indices_buf[2].ptr(),
                indices_buf[3].ptr(),
                sort_indices_temp_storage_buf.ptr(),
                acc_indices_buf[i].ptr(),
                len as i32,
                windows as i32,
                window_bits as i32,
                threads as i32,
                worker as i32,
                (sort_temp_storage_size * mem::size_of::<u32>()) as i32,
                skip_zero.into(),
                stream.1,
            );
            assert_eq!(err, cudaError::cudaSuccess);
        }

        if (intt_args.selector)(i) {
            assert!(!(cache_buffer_selector)(i));
            ntt_raw(
                device,
                &mut scalar_dev_bufs[idx],
                &mut intt_tmp_bufs[idx],
                intt_args.pq_buf,
                intt_args.omegas_buf,
                intt_args.len_log,
                Some(intt_args.divisor_buf),
                Some(&stream.0),
            )?;

            let mut buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?;
            mem::swap(&mut scalar_dev_bufs[idx], &mut buf);
            pending_copy_queue.push((i, buf));
        }

        if (cache_buffer_selector)(i) {
            let mut buf = cache_bufs.pop().unwrap();
            mem::swap(&mut scalar_dev_bufs[idx], &mut buf);
            cached_buffer.insert(i, buf);
        }
    }

    drop(streams);

    let (sw, stream) = CudaStreamWrapper::new_with_inner();
    for (i, buffer) in pending_copy_queue.iter() {
        device.copy_from_device_to_host_async(&mut scalar_buf[*i][..], &buffer, stream)?;
    }

    (before_final_round)();

    let res = batch_msm_acc(
        device,
        &curve_buf,
        &acc_indices_buf,
        windows,
        window_bits,
        worker,
        bucket_size,
        curve_buf_size,
    )?;

    sw.sync();

    Ok((res, cached_buffer))
}

pub(crate) fn batch_msm<C: CurveAffine, B: ToDevBuffer>(
    device: &CudaDevice,
    points_dev_buf: &CudaDeviceBufRaw,
    scalar_buf: Vec<B>,
    len: usize,
    skip_zero: bool,
) -> DeviceResult<Vec<C>> {
    batch_msm_ext(
        device,
        points_dev_buf,
        scalar_buf,
        &mut || {},
        len,
        skip_zero,
    )
}

pub(crate) fn batch_msm_ext<C: CurveAffine, B: ToDevBuffer>(
    device: &CudaDevice,
    points_dev_buf: &CudaDeviceBufRaw,
    scalar_buf: Vec<B>,
    before_final_round: &mut dyn FnMut() -> (),
    len: usize,
    skip_zero: bool,
) -> DeviceResult<Vec<C>> {
    if scalar_buf.len() == 0 {
        before_final_round();
        return Ok(vec![]);
    }

    let threads = 64;
    let bits = 254;

    let msm_count = scalar_buf.len();

    // k22, 8, 13bits is best for RTX4090
    let window_bits = log2(msm_count).min(3) as usize + 10 + log2(len).max(22) as usize - 22;
    let windows = (bits + window_bits - 1) / window_bits;
    let bucket_size = windows << window_bits;
    let max_worker = 128 * 512;
    let worker = if len < max_worker {
        (len + threads - 1) / threads * threads
    } else {
        max_worker
    };

    let streams = [
        CudaStreamWrapper::new_with_inner_and_priority(1),
        CudaStreamWrapper::new_with_inner_and_priority(1),
    ];

    let scalar_dev_bufs = [
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
        device.alloc_device_buffer_non_zeroed::<C::Scalar>(len)?,
    ];

    // About 30MB per MSM
    // layout: | bucktes | msm worker remain | collect worker remain |
    let curve_buf_size = bucket_size + worker * 2;
    let curve_buf =
        device.alloc_device_buffer_non_zeroed::<C::Curve>(msm_count * curve_buf_size)?;

    let single_sort_indices_size = len * windows;
    let sort_temp_storage_size = single_sort_indices_size * 3; // ~2N = 2 * windows * len, pick 32 as upper bound

    let sort_indices_temp_storage_buf =
        device.alloc_device_buffer_non_zeroed::<u32>(sort_temp_storage_size)?;

    let indices_buf = [0; 4].map(|_| {
        device
            .alloc_device_buffer_non_zeroed::<u32>(single_sort_indices_size)
            .unwrap()
    });

    let acc_indices_buf = (0..msm_count)
        .into_iter()
        .map(|_| device.alloc_device_buffer_non_zeroed::<u32>(worker * 2))
        .collect::<DeviceResult<Vec<_>>>()
        .unwrap();

    for (i, buf) in scalar_buf.into_iter().enumerate() {
        let idx = i & 1;
        let scalar_dev_buf = &scalar_dev_bufs[idx];
        let stream = &streams[idx];
        let last_stream = &streams[1 - idx];
        stream.0.sync();
        let scalar_dev_buf = buf.to_dev_buf(device, scalar_dev_buf, &stream.0).unwrap();
        last_stream.0.sync(); // sync to reuse sort_indices_buf
        unsafe {
            let err = msm(
                (curve_buf.ptr() as usize + i * curve_buf_size * std::mem::size_of::<C::Curve>())
                    as _,
                points_dev_buf.ptr(),
                scalar_dev_buf.ptr(),
                indices_buf[0].ptr(),
                indices_buf[1].ptr(),
                indices_buf[2].ptr(),
                indices_buf[3].ptr(),
                sort_indices_temp_storage_buf.ptr(),
                acc_indices_buf[i].ptr(),
                len as i32,
                windows as i32,
                window_bits as i32,
                threads as i32,
                worker as i32,
                (sort_temp_storage_size * mem::size_of::<u32>()) as i32,
                skip_zero.into(),
                stream.1,
            );
            assert_eq!(err, cudaError::cudaSuccess);
        }
    }
    drop(streams);

    (before_final_round)();

    let res = batch_msm_acc(
        device,
        &curve_buf,
        &acc_indices_buf,
        windows,
        window_bits,
        worker,
        bucket_size,
        curve_buf_size,
    )?;

    Ok(res)
}

fn batch_msm_acc<C: CurveAffine>(
    device: &CudaDevice,
    curve_buf: &CudaDeviceBufRaw,
    acc_indices_buf: &Vec<CudaDeviceBufRaw>,
    windows: usize,
    window_bits: usize,
    worker: usize,
    bucket_size: usize,
    curve_buf_size: usize,
) -> DeviceResult<Vec<C>> {
    let msm_count = acc_indices_buf.len();
    let (sw, stream) = CudaStreamWrapper::new_with_inner();

    let mut remain_indices_ptr = vec![];
    let mut remain_acc_ptr = vec![];
    let mut next_remain_indices_ptr = vec![];
    let mut next_remain_acc_ptr = vec![];
    let mut buckets_ptr = vec![];

    let bucket_base = curve_buf.ptr() as usize;
    for i in 0..msm_count {
        let indices_base = acc_indices_buf[i].ptr() as usize;
        let curr_bucket_offset = curve_buf_size * i;
        let curve_size = std::mem::size_of::<C::Curve>();

        remain_indices_ptr.push(indices_base);
        next_remain_indices_ptr.push(indices_base + worker * std::mem::size_of::<u32>());

        let ptr = bucket_base + curr_bucket_offset * curve_size;
        buckets_ptr.push(ptr);

        let ptr = ptr + bucket_size * curve_size;
        remain_acc_ptr.push(ptr);

        let ptr = ptr + worker * curve_size;
        next_remain_acc_ptr.push(ptr);
    }

    let remain_indices_buf = device
        .alloc_device_buffer_from_slice_async(&remain_indices_ptr[..], stream)
        .unwrap();
    let remain_acc_ptr_buf = device
        .alloc_device_buffer_from_slice_async(&remain_acc_ptr[..], stream)
        .unwrap();
    let next_remain_indices_ptr_buf = device
        .alloc_device_buffer_from_slice_async(&next_remain_indices_ptr[..], stream)
        .unwrap();
    let next_remain_acc_ptr_buf = device
        .alloc_device_buffer_from_slice_async(&next_remain_acc_ptr[..], stream)
        .unwrap();
    let buckets_ptr_buf = device
        .alloc_device_buffer_from_slice_async(&buckets_ptr[..], stream)
        .unwrap();

    unsafe {
        let err = batch_msm_collect(
            remain_indices_buf.ptr(),
            remain_acc_ptr_buf.ptr(),
            next_remain_indices_ptr_buf.ptr(),
            next_remain_acc_ptr_buf.ptr(),
            buckets_ptr_buf.ptr(),
            worker as u32,
            windows as u32,
            window_bits as u32,
            msm_count as u32,
            stream,
        );
        assert_eq!(err, cudaError::cudaSuccess);
    }

    sw.sync();

    let mut res = vec![C::identity().to_curve(); msm_count];
    for i in 0..msm_count {
        device
            .copy_from_device_to_host_async_v2(
                &mut res[i..i + 1],
                &curve_buf,
                (curve_buf_size * i) as isize,
                None,
            )
            .unwrap();
    }
    device.synchronize().unwrap();

    Ok(res.into_iter().map(|c| c.to_affine()).collect())
}

#[test]
fn test_msm() {
    use crate::CudaDevice;
    use crate::{cuda::msm::batch_msm, device::Device};
    use ark_std::{end_timer, start_timer};
    use halo2_proofs::arithmetic::Field;
    use halo2_proofs::pairing::group::Curve;
    use halo2_proofs::{
        arithmetic::BaseExt,
        pairing::bn256::{Fr, G1Affine, G1},
    };
    use rand::thread_rng;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;

    {
        let mut allocator = crate::device::cuda::CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset((1 << 22) * core::mem::size_of::<Fr>(), 100);
    }

    const CANDIDATES: usize = 1 << 8;
    let mut candidate_scalars = vec![];
    let mut candidate_points = vec![];
    for i in 0..CANDIDATES {
        candidate_scalars.push(Fr::rand());
        candidate_points.push((G1::generator() * candidate_scalars[i]).to_affine());
    }

    let len_deg_start = 22;
    let len_deg_end = 22;
    let batch_deg_start = 0;
    let batch_deg_end = 7;
    let rounds = 4;

    for deg in len_deg_start..=len_deg_end {
        let len = 1 << deg;
        let mut points = vec![];
        let timer = start_timer!(|| "prepare point");
        for i in 0..len {
            points.push(candidate_points[i % CANDIDATES]);
        }
        end_timer!(timer);

        let timer = start_timer!(|| "prepare scalars");
        let scalars = vec![0; len]
            .into_par_iter()
            .map(|_| Fr::random(thread_rng()))
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "calc expect");
        let sum = scalars.iter().enumerate().fold(Fr::zero(), |acc, (i, x)| {
            acc + x * candidate_scalars[i % CANDIDATES]
        });
        let expected_res = (G1::generator() * sum).to_affine();
        end_timer!(timer);

        let device = CudaDevice::get_device(0).unwrap();
        let p_buf = device.alloc_device_buffer_from_slice(&points[..]).unwrap();
        for msm_count_deg in batch_deg_start..=batch_deg_end {
            for round in 0..rounds {
                let timer = start_timer!(|| format!(
                    "run msm, len_deg {}, msm_count_deg {}, round {}",
                    deg, msm_count_deg, round
                ));
                let res = batch_msm::<G1Affine, _>(
                    &device,
                    &p_buf,
                    vec![&scalars; 1 << msm_count_deg],
                    len,
                    round & 1 == 1,
                )
                .unwrap();
                end_timer!(timer);
                for c in res {
                    assert_eq!(c, expected_res);
                }
            }
        }
    }
}
