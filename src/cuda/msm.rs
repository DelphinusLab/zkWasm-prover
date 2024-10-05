use std::mem;

use ark_std::log2;
use cuda_runtime_sys::cudaError;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::pairing::group::Curve;
use halo2_proofs::pairing::group::Group;

use crate::cuda::bn254_c::batch_msm_collect;
use crate::cuda::bn254_c::msm;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::{CudaDevice, CudaDeviceBufRaw, CudaStreamWrapper};
use crate::device::Device;
use crate::device::DeviceResult;

const DEBUG: bool = false;
const GPU_MEMORY_PROFILING: bool = DEBUG || false;

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

pub(crate) fn batch_msm<C: CurveAffine, B: ToDevBuffer>(
    device: &CudaDevice,
    points_dev_buf: &CudaDeviceBufRaw,
    scalar_buf: Vec<B>,
    len: usize,
) -> DeviceResult<Vec<C>> {
    let threads = 64;
    let bits = 254;

    let msm_count = scalar_buf.len();

    // k22, 8, 13bits is best for RTX4090
    let window_bits = log2(msm_count).min(3) as usize + 10 + log2(len).max(22) as usize - 22;
    if GPU_MEMORY_PROFILING {
        println!(
            "msm_count is {}, log is {}, window_bits is {}",
            msm_count,
            log2(msm_count),
            window_bits
        );
    }

    let windows = (bits + window_bits - 1) / window_bits;
    let bucket_size = windows << window_bits;
    let max_worker = 64 * 512;
    let worker = if len < max_worker {
        (len + threads - 1) / threads * threads
    } else {
        max_worker
    };

    if GPU_MEMORY_PROFILING {
        println!(
            "len {}, windows {}, window_bits {}, worker {}",
            len, windows, window_bits, worker
        );
    }

    let streams = [
        CudaStreamWrapper::new_with_inner(),
        CudaStreamWrapper::new_with_inner(),
    ];

    if GPU_MEMORY_PROFILING {
        println!(
            "scalar_dev_bufs size is {} MB",
            (len * mem::size_of::<C::Scalar>() * 2) >> 20
        );
    }
    let scalar_dev_bufs = [
        device.alloc_device_buffer::<C::Scalar>(len).unwrap(),
        device.alloc_device_buffer::<C::Scalar>(len).unwrap(),
    ];

    // About 30MB per MSM
    // layout: | bucktes | msm worker remain | collect worker remain |
    let curve_buf_size = bucket_size + worker * 2;
    if GPU_MEMORY_PROFILING {
        println!(
            "curve_buf size is {} MB",
            (msm_count * curve_buf_size * mem::size_of::<C::Curve>()) >> 20
        );
    }
    let curve_buf = device
        .alloc_device_buffer_non_zeroed::<C::Curve>(msm_count * curve_buf_size)
        .unwrap();

    let single_sort_indices_size = len * windows;
    let sort_temp_storage_size = single_sort_indices_size * 3; // ~2N = 2 * windows * len, pick 32 as upper bound
    let total_sort_indices_size = single_sort_indices_size * 4 + sort_temp_storage_size;
    if GPU_MEMORY_PROFILING {
        println!(
            "sort_indices_buf size is {} MB",
            (total_sort_indices_size * mem::size_of::<u32>()) >> 20
        );
    }
    let sort_indices_buf = device
        .alloc_device_buffer_non_zeroed::<u32>(total_sort_indices_size)
        .unwrap();

    // About 64KB per MSM
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
                sort_indices_buf.ptr(),
                acc_indices_buf[i].ptr(),
                len as i32,
                windows as i32,
                window_bits as i32,
                threads as i32,
                worker as i32,
                (sort_temp_storage_size * mem::size_of::<u32>()) as i32,
                stream.1,
            );
            assert_eq!(err, cudaError::cudaSuccess);
        }
    }
    drop(streams);

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

        if DEBUG {
            let mut i_buf = vec![0u32; windows * len];
            let mut pi_buf = vec![0u32; windows * len];
            device
                .copy_from_device_to_host_async_v2(
                    &mut i_buf[..],
                    &sort_indices_buf,
                    (2 * len * windows) as isize,
                    None,
                )
                .unwrap();
            device
                .copy_from_device_to_host_async_v2(
                    &mut pi_buf[..],
                    &sort_indices_buf,
                    (3 * len * windows) as isize,
                    None,
                )
                .unwrap();
            for j in 0..windows * len {
                if i_buf[j] != 0 {
                    println!(
                        "sorted indices {} has window {} bucket offset {} p index {}",
                        j,
                        i_buf[j] >> 16,
                        i_buf[j] & (0xffff),
                        pi_buf[j]
                    );
                }
            }

            let mut buf = vec![C::CurveExt::identity(); bucket_size + worker];
            device
                .copy_from_device_to_host_async_v2(
                    &mut buf[..],
                    &curve_buf,
                    (curve_buf_size * i) as isize,
                    None,
                )
                .unwrap();

            for w in 0..windows {
                for j in 0..1 << window_bits {
                    let p = buf[(w << window_bits) + j];
                    if p.is_identity().into() {
                    } else {
                        println!(
                            "round 1 msm {} window {} bucket {} has p {:?}",
                            i,
                            w,
                            j,
                            p.to_affine()
                        );
                    }
                }
            }

            let mut i_buf = vec![0u32; worker];
            device
                .copy_from_device_to_host_async_v2(
                    &mut i_buf[..],
                    &acc_indices_buf[0],
                    0 as isize,
                    None,
                )
                .unwrap();

            for w in 0..worker {
                let p = buf[bucket_size + w];
                if p.is_identity().into() {
                } else {
                    println!(
                        "round 1 msm_id {} worker_id {} windex {} bucket offset {} point_acc {:?} ",
                        i,
                        w,
                        i_buf[w] >> 16,
                        i_buf[w] & 0xffff,
                        p.to_affine(),
                    )
                };
            }
        }
    }

    unsafe {
        let err = batch_msm_collect(
            device
                .alloc_device_buffer_from_slice(&remain_indices_ptr[..])
                .unwrap()
                .ptr(),
            device
                .alloc_device_buffer_from_slice(&remain_acc_ptr[..])
                .unwrap()
                .ptr(),
            device
                .alloc_device_buffer_from_slice(&next_remain_indices_ptr[..])
                .unwrap()
                .ptr(),
            device
                .alloc_device_buffer_from_slice(&next_remain_acc_ptr[..])
                .unwrap()
                .ptr(),
            device
                .alloc_device_buffer_from_slice(&buckets_ptr[..])
                .unwrap()
                .ptr(),
            worker as u32,
            windows as u32,
            window_bits as u32,
            msm_count as u32,
            0 as _,
        );

        assert_eq!(err, cudaError::cudaSuccess);
    }

    for i in 0..msm_count {
        if DEBUG {
            if worker > 128 {
                let mut i_buf = vec![0u32; 128];
                device
                    .copy_from_device_to_host_async_v2(
                        &mut i_buf[..],
                        &acc_indices_buf[i],
                        worker as isize,
                        None,
                    )
                    .unwrap();
                for j in 0..128 {
                    if i_buf[j] != 0 {
                        println!(
                            "round 2 indices at worker {} has window {} bucket offset {}",
                            j,
                            i_buf[j] >> 16,
                            i_buf[j] & (0xffff),
                        );
                    }
                }
            }

            let mut buf = vec![C::Curve::identity(); curve_buf_size];
            device
                .copy_from_device_to_host_async_v2::<C::Curve>(
                    &mut buf[..],
                    &curve_buf,
                    (curve_buf_size * i) as isize,
                    None,
                )
                .unwrap();

            if worker > 128 {
                for w in 0..worker {
                    let p = buf[w + worker + bucket_size];
                    if p.is_identity().into() {
                    } else {
                        println!("msm {} round2 worker {} value {:?}", i, w, p.to_affine())
                    };
                }
            }

            for w in 0..windows {
                for j in 0..1 << window_bits {
                    let p = buf[(w << window_bits) + j];
                    if p.is_identity().into() {
                    } else {
                        println!(
                            "msm {} window {} bucket {} has p {:?}",
                            i,
                            w,
                            j,
                            p.to_affine()
                        );
                    }
                }
            }
        }
    }

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
