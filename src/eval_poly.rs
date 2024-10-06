use std::{
    collections::{BTreeMap, HashMap},
    mem::{self, ManuallyDrop},
    rc::Rc,
};

use halo2_proofs::arithmetic::FieldExt;

use crate::device::{
    cuda::{to_result, CudaDevice, CudaDeviceBufRaw},
    Device as _, DeviceResult,
};
use crate::{
    cuda::bn254_c::poly_eval,
    device::cuda::{CudaBuffer, CudaStreamWrapper},
};

pub(crate) fn batch_poly_eval<F: FieldExt>(
    device: &CudaDevice,
    evals: Vec<(&[F], usize, F)>,
    k: usize,
    cache_count: usize,
) -> DeviceResult<(
    BTreeMap<usize, CudaDeviceBufRaw>,
    BTreeMap<(usize, F), F>,
    Vec<F>,
)> {
    let size = 1 << k;

    // 1. Merge evals into Map from buffer_ptr to its (slice, priority, eval_point_vec)
    // 2. Collect all eval_points
    let mut collection: HashMap<usize, (&[F], usize, Vec<(usize, &F)>)> = HashMap::new();
    let mut x_sets = vec![];
    for (idx, (slice, prio, x)) in evals.iter().enumerate() {
        let addr = slice.as_ptr() as usize;
        x_sets.push(x);
        collection
            .entry(addr)
            .and_modify(|(_, _, arr)| arr.push((idx, x)))
            .or_insert((slice, *prio, vec![(idx, x)]));
    }

    // 3. For each x, we prepare its 1st..kth square for GPU eval
    x_sets.sort_unstable();
    x_sets.dedup();
    let mut x_extend_sets = vec![];
    for x in x_sets.iter() {
        x_extend_sets.push(**x);
        for _ in 1..k {
            x_extend_sets.push(x_extend_sets.last().unwrap().square());
        }
    }

    let x_buf = device
        .alloc_device_buffer_from_slice(&x_extend_sets)
        .unwrap();
    let mut x_map = BTreeMap::new();
    for (i, x) in x_sets.into_iter().enumerate() {
        x_map.insert(
            x,
            ManuallyDrop::new(CudaDeviceBufRaw {
                ptr: unsafe { x_buf.ptr().offset((i * k * mem::size_of::<F>()) as isize) },
                device: device.clone(),
                size: core::mem::size_of::<F>(),
            }),
        );
    }

    // 4. Sort Collection for better parallel
    let mut collection = collection.into_values().collect::<Vec<_>>();
    collection.sort_by(|(_, a_prio, a_points), (_, b_prio, b_points)| {
        a_prio
            .cmp(b_prio)
            .then(a_points.len().cmp(&b_points.len()).reverse())
    });

    let mut evals = vec![F::zero(); evals.len()];
    let mut eval_map = BTreeMap::new();
    let mut poly_buf_cache = BTreeMap::new();

    let (copy_sw, copy_stream) = CudaStreamWrapper::new_with_inner();
    let (eval_sw, eval_stream) = CudaStreamWrapper::new_with_inner();

    let res_buf = device.alloc_device_buffer_non_zeroed::<F>(size)?;
    let eval_buf = device.alloc_device_buffer_non_zeroed::<F>(evals.len())?;

    let unpinned_buffers = collection.iter().filter(|x| x.1 == 0).count();
    let mut pending_tasks = vec![];
    let mut pending_bufs = vec![];
    assert!(unpinned_buffers < cache_count);

    for (i, (slice, _, points)) in collection.into_iter().enumerate() {
        let buf = Rc::new(
            device
                .alloc_device_buffer_from_slice_async::<F>(slice, copy_stream)
                .unwrap(),
        );

        if pending_bufs.len() >= 1 {
            eval_sw.sync();
            pending_bufs.clear();
        }

        for (idx, x) in points {
            pending_tasks.push((slice, buf.clone(), idx, x))
        }

        // Start to issue tasks after finish unpinned buffer copy.
        if i >= unpinned_buffers - 1 {
            copy_sw.sync();

            for (slice, buf, idx, x) in pending_tasks {
                unsafe {
                    let err = poly_eval(
                        buf.ptr(),
                        res_buf.ptr(),
                        x_map.get(x).unwrap().ptr(),
                        size as i32,
                        eval_stream,
                    );
                    to_result((), err, "fail to run poly_eval")?;

                    device.copy_from_device_to_device_async::<F>(
                        &eval_buf,
                        idx,
                        &res_buf,
                        0,
                        1,
                        eval_stream,
                    )?;
                    eval_map.insert((slice.as_ptr() as usize, *x), idx);
                }
            }
            pending_tasks = vec![];
        }

        if poly_buf_cache.len() < cache_count {
            poly_buf_cache.insert(slice.as_ptr() as usize, buf);
        } else {
            // buf can be released after next eval_sw.sync().
            pending_bufs.push(buf);
        }
    }
    copy_sw.sync();
    eval_sw.sync();

    device
        .copy_from_device_to_host(&mut evals, &eval_buf)
        .unwrap();

    let eval_map = eval_map
        .into_iter()
        .map(|(k, eval_idx)| (k, evals[eval_idx]))
        .collect::<BTreeMap<(usize, F), F>>();

    let poly_buf_cache = poly_buf_cache
        .into_iter()
        .map(|(k, v)| (k, Rc::try_unwrap(v).unwrap()))
        .collect();

    // Skip commit first eval, it is for prover calculation only.
    Ok((poly_buf_cache, eval_map, evals))
}
