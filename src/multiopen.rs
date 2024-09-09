use std::iter;

use crate::hugetlb::HugePageAllocator;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::Rotation;

#[derive(Debug, Clone, Copy)]
pub struct ProverQuery<'a, F: FieldExt> {
    pub point: F,
    pub rotation: Rotation,
    pub poly: &'a [F],
}

pub(crate) mod gwc {
    use ark_std::end_timer;
    use ark_std::start_timer;

    use halo2_proofs::arithmetic::CurveAffine;
    use halo2_proofs::arithmetic::Field;
    use halo2_proofs::arithmetic::FieldExt;
    use halo2_proofs::poly::Rotation;
    use halo2_proofs::transcript::EncodedChallenge;
    use halo2_proofs::transcript::TranscriptWrite;
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;
    use std::collections::BTreeMap;

    use crate::cuda::bn254::batch_msm;
    use crate::cuda::bn254::field_op_v3;
    use crate::cuda::bn254::FieldOp;
    use crate::device::cuda::CudaDevice;
    use crate::device::cuda::CudaDeviceBufRaw;
    use crate::device::Device as _;
    use crate::device::DeviceResult;
    use crate::hugetlb::HugePageAllocator;
    use crate::multiopen::ProverQuery;

    pub struct CommitmentData<'a, F: FieldExt> {
        queries: Vec<ProverQuery<'a, F>>,
        point: F,
    }

    fn construct_intermediate_sets<'a, F: FieldExt, I>(queries: I) -> Vec<CommitmentData<'a, F>>
    where
        I: IntoIterator<Item = ProverQuery<'a, F>>,
    {
        let mut point_query_map: BTreeMap<Rotation, Vec<_>> = BTreeMap::new();
        for query in queries {
            point_query_map
                .entry(query.rotation)
                .and_modify(|x| x.push(query))
                .or_insert(vec![query]);
        }

        point_query_map
            .into_iter()
            .map(|(_, queries)| {
                let point = queries[0].point;
                CommitmentData { queries, point }
            })
            .collect()
    }

    pub(crate) fn multiopen<
        'a,
        I,
        C: CurveAffine,
        E: EncodedChallenge<C>,
        T: TranscriptWrite<C, E>,
    >(
        device: &CudaDevice,
        g_buf: &CudaDeviceBufRaw,
        queries: I,
        size: usize,
        s_buf: [&CudaDeviceBufRaw; 2],
        eval_map: BTreeMap<(usize, C::Scalar), C::Scalar>,
        transcript: &mut T,
    ) -> DeviceResult<()>
    where
        I: IntoIterator<Item = ProverQuery<'a, C::Scalar>>,
    {
        let v: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let commitment_data = construct_intermediate_sets(queries);

        let mut eval_batch = vec![C::Scalar::zero(); commitment_data.len()];

        let mut collection = BTreeMap::new();
        let mut bufs = vec![];
        for (rot_idx, data) in commitment_data.iter().enumerate() {
            bufs.push(device.alloc_device_buffer::<C::Scalar>(size)?);
            let len = data.queries.len();
            for (inner_idx, q) in data.queries.iter().enumerate() {
                collection
                    .entry(q.poly.as_ptr() as usize)
                    .and_modify(|x: &mut (_, Vec<_>)| {
                        x.1.push((data.point, rot_idx, len - 1 - inner_idx))
                    })
                    .or_insert((q.poly, vec![(data.point, rot_idx, len - 1 - inner_idx)]));
            }
        }

        let mut vs = vec![C::Scalar::one(), v];
        let tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let v_buf = device.alloc_device_buffer::<C::Scalar>(1)?;
        for (_, (poly, assoc)) in collection {
            device.copy_from_host_to_device_async(&tmp_buf, poly, 0usize as _)?;
            for (x, rot_idx, inner_idx) in assoc {
                for _ in vs.len()..=inner_idx {
                    vs.push(*vs.last().unwrap() * v);
                }
                device.copy_from_host_to_device_async(&v_buf, &[vs[inner_idx]][..], 0usize as _)?;
                field_op_v3(
                    device,
                    &bufs[rot_idx],
                    Some(&bufs[rot_idx]),
                    None,
                    Some(&tmp_buf),
                    Some(&v_buf),
                    size,
                    FieldOp::Add,
                    None,
                )?;

                let eval = eval_map.get(&(poly.as_ptr() as usize, x));
                eval_batch[rot_idx] += eval.cloned().unwrap() * vs[inner_idx];
            }
        }

        let mut ws = commitment_data
            .par_iter()
            .map(|_| {
                let mut poly_batch = Vec::new_in(HugePageAllocator);
                poly_batch.resize(size, C::Scalar::zero());
                poly_batch
            })
            .collect::<Vec<_>>();

        for i in 0..ws.len() {
            device.copy_from_device_to_host(&mut ws[i][..], &bufs[i])?;
        }

        ws.par_iter_mut()
            .zip(commitment_data.par_iter().zip(eval_batch.par_iter()))
            .for_each(|(poly_batch, (commitment_at_a_point, eval_batch))| {
                let z = commitment_at_a_point.point;
                poly_batch[0] -= eval_batch;

                let mut tmp = *poly_batch.last().unwrap();
                *poly_batch.last_mut().unwrap() = C::Scalar::zero();
                for i in (1..poly_batch.len() - 1).rev() {
                    let p = poly_batch[i] + tmp * z;
                    poly_batch[i] = tmp;
                    tmp = p;
                }
                poly_batch[0] = tmp;
            });

        let timer = start_timer!(|| "msm");

        let commitments = batch_msm(&g_buf, s_buf, ws.iter().map(|x| &x[..]).collect(), size)?;
        for commitment in commitments {
            transcript.write_point(commitment).unwrap();
        }

        end_timer!(timer);

        Ok(())
    }
}

pub mod shplonk {
    use halo2_proofs::arithmetic::eval_polynomial_st;
    use halo2_proofs::arithmetic::lagrange_interpolate;
    use halo2_proofs::arithmetic::CurveAffine;
    use halo2_proofs::arithmetic::Field;
    use halo2_proofs::arithmetic::FieldExt;
    use halo2_proofs::plonk::ProvingKey;
    use halo2_proofs::poly::Rotation;
    use halo2_proofs::transcript::EncodedChallenge;
    use halo2_proofs::transcript::TranscriptWrite;
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    use crate::cuda::bn254::batch_msm;
    use crate::cuda::bn254::batch_msm_v2;
    use crate::cuda::bn254::field_op_v3;
    use crate::cuda::bn254::FieldOp;
    use crate::device::cuda::CudaBuffer;
    use crate::device::cuda::CudaDevice;
    use crate::device::cuda::CudaDeviceBufRaw;
    use crate::device::cuda::CudaStreamWrapper;
    use crate::device::Device as _;
    use crate::device::DeviceResult;
    use crate::hugetlb::HugePageAllocator;
    use crate::multiopen::ProverQuery;

    fn construct_intermediate_sets<'a, F: FieldExt, I>(
        queries: I,
        eval_map: BTreeMap<(usize, F), F>,
    ) -> (Vec<(Vec<(&'a [F], Vec<F>)>, Vec<F>)>, Vec<F>)
    where
        I: IntoIterator<Item = ProverQuery<'a, F>>,
    {
        let queries = queries.into_iter().collect::<Vec<_>>();

        let mut rotation_point_map = BTreeMap::new();
        for query in queries.clone() {
            rotation_point_map
                .entry(query.rotation)
                .or_insert_with(|| query.point);
        }

        let super_point_set: Vec<F> = rotation_point_map.values().cloned().collect();

        let mut poly_rotation_set_map: Vec<(&[F], BTreeSet<Rotation>)> = vec![];
        for query in queries.clone() {
            let rotation = query.rotation;
            if let Some(pos) = poly_rotation_set_map
                .iter()
                .position(|(poly, _)| (*poly).as_ptr() == query.poly.as_ptr())
            {
                let (_, rotation_set) = &mut poly_rotation_set_map[pos];
                rotation_set.insert(rotation);
            } else {
                let rotation_set = BTreeSet::from([rotation]);
                poly_rotation_set_map.push((query.poly, rotation_set));
            };
        }

        let mut rotation_set_poly_map = BTreeMap::<BTreeSet<_>, Vec<_>>::new();
        for (commitment, rotation_set) in poly_rotation_set_map.iter() {
            let commitments = rotation_set_poly_map
                .entry(rotation_set.clone())
                .or_insert_with(Vec::new);
            commitments.push(*commitment);
        }

        let rotation_sets = rotation_set_poly_map
            .into_iter()
            .enumerate()
            .map(|(i, (rotation_set, polys))| {
                let rotations: Vec<Rotation> = rotation_set.iter().cloned().collect();
                let points: Vec<_> = rotations
                    .iter()
                    .map(|rotation| *rotation_point_map.get(rotation).unwrap())
                    .collect();

                let polys: Vec<_> = polys
                    .iter()
                    .map(|poly| {
                        let evals: Vec<F> = points
                            .iter()
                            .map(|x| {
                                let eval = eval_map.get(&(poly.as_ptr() as usize, *x)).cloned();
                                if eval.is_none() {
                                    println!("miss eval on {:?} on set {}", *x, i);
                                }
                                eval.unwrap()
                            })
                            .collect();
                        (*poly, evals)
                    })
                    .collect();

                (polys, points)
            })
            .collect::<Vec<_>>();

        (rotation_sets, super_point_set)
    }

    pub(crate) fn multiopen<
        'a,
        I,
        C: CurveAffine,
        E: EncodedChallenge<C>,
        T: TranscriptWrite<C, E>,
    >(
        pk: &ProvingKey<C>,
        device: &CudaDevice,
        g_buf: &CudaDeviceBufRaw,
        queries: I,
        size: usize,
        s_buf: [&CudaDeviceBufRaw; 2],
        eval_map: BTreeMap<(usize, C::Scalar), C::Scalar>,
        mut poly_cache: BTreeMap<usize, CudaDeviceBufRaw>,
        transcript: &mut T,
    ) -> DeviceResult<()>
    where
        I: IntoIterator<Item = ProverQuery<'a, C::Scalar>>,
    {
        let y: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let y_buf = device.alloc_device_buffer_from_slice(&[y][..])?;

        let v: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let (rotation_sets, super_point_set) = construct_intermediate_sets(queries, eval_map);

        let mut streams = vec![];
        let rotation_sets: Vec<(_, _, Vec<_>)> = rotation_sets
            .iter()
            .map(|(queries, points)| -> DeviceResult<(_, _, Vec<_>)> {
                let v_buf = if let Some(buf) = poly_cache.remove(&(queries[0].0.as_ptr() as usize))
                {
                    buf
                } else {
                    let (stream_wrapper, stream) = CudaStreamWrapper::new_with_inner();
                    let buf =
                        device.alloc_device_buffer_from_slice_async(&queries[0].0[..], stream)?;
                    streams.push(stream_wrapper);
                    buf
                };

                let mut evals_acc = lagrange_interpolate(&points[..], &queries[0].1[..]);

                if queries.len() > 1 {
                    if let Some(stream) = streams.last() {
                        stream.sync();
                    }
                    let mut buf = None;
                    let calc_streams = CudaStreamWrapper::new_with_inner();
                    for (poly, evals) in queries.iter().skip(1) {
                        let poly_buf =
                            if let Some(buf) = poly_cache.remove(&(poly.as_ptr() as usize)) {
                                buf
                            } else {
                                let (stream_wrapper, stream) = CudaStreamWrapper::new_with_inner();
                                let buf = device
                                    .alloc_device_buffer_from_slice_async(&poly[..], stream)?;
                                drop(stream_wrapper);
                                buf
                            };

                        calc_streams.0.sync();

                        field_op_v3(
                            device,
                            &v_buf,
                            Some(&v_buf),
                            Some(&y_buf),
                            Some(&poly_buf),
                            None,
                            size,
                            FieldOp::Add,
                            Some(calc_streams.1),
                        )?;

                        // Can't release poly_buf before calc_streams sync.
                        buf = Some(poly_buf);

                        let evals = lagrange_interpolate(&points[..], &evals[..]);
                        for i in 0..evals_acc.len() {
                            evals_acc[i] = evals_acc[i] * y + evals[i];
                        }
                    }
                    drop(calc_streams);
                    drop(buf);
                }

                Ok((points, v_buf, evals_acc))
            })
            .collect::<DeviceResult<Vec<_>>>()?;
        drop(streams);
        drop(poly_cache);

        let k = pk.vk.domain.k as usize;
        let (ntt_omegas_buf, ntt_pq_buf) =
            crate::ntt_prepare(&device, pk.get_vk().domain.get_omega(), k)?;
        let (intt_omegas_buf, intt_pq_buf) =
            crate::ntt_prepare(&device, pk.get_vk().domain.get_omega_inv(), k)?;
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])?;

        let mut hx_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let mut poly_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let mut tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let point_buf = device.alloc_device_buffer::<C::Scalar>(super_point_set.len())?;
        let v_buf = device.alloc_device_buffer_from_slice(&[v][..])?;

        for (_, (points, poly, evals)) in rotation_sets.iter().enumerate() {
            device.copy_from_device_to_device::<C::Scalar>(&poly_buf, 0, poly, 0, size)?;
            device.copy_from_host_to_device(&point_buf, &evals[..])?;

            crate::cuda::bn254::field_op_v2::<C::ScalarExt>(
                &device,
                &poly_buf,
                Some(&poly),
                None,
                Some(&point_buf),
                None,
                evals.len(),
                FieldOp::Sub,
            )?;

            let diffs: Vec<C::Scalar> = super_point_set
                .iter()
                .filter(|point| !points.contains(point))
                .copied()
                .collect();

            crate::cuda::bn254::ntt_raw(
                &device,
                &mut poly_buf,
                &mut tmp_buf,
                &ntt_pq_buf,
                &ntt_omegas_buf,
                k,
                None,
            )?;

            device.copy_from_host_to_device(&point_buf, &diffs[..])?;

            unsafe {
                let err = crate::cuda::bn254_c::shplonk_h_x_merge(
                    hx_buf.ptr(),
                    v_buf.ptr(),
                    poly_buf.ptr(),
                    ntt_omegas_buf.ptr(),
                    point_buf.ptr(),
                    diffs.len() as i32,
                    size as i32,
                );

                crate::device::cuda::to_result((), err, "failed to run shplonk_h_x_merge")?;
            }
        }

        device.copy_from_host_to_device(&point_buf, &super_point_set[..])?;
        unsafe {
            let err = crate::cuda::bn254_c::shplonk_h_x_div_points(
                hx_buf.ptr(),
                ntt_omegas_buf.ptr(),
                point_buf.ptr(),
                super_point_set.len() as i32,
                size as i32,
            );

            crate::device::cuda::to_result((), err, "failed to run shplonk_h_x_div_points")?;
        }

        crate::intt_raw(
            &device,
            &mut hx_buf,
            &mut tmp_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;

        let commitment = batch_msm_v2::<C>(&g_buf, vec![&hx_buf], size)?;
        transcript.write_point(commitment[0]).unwrap();

        let u: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let zt_eval = super_point_set
            .iter()
            .map(|root| u - root)
            .reduce(|a, b| a * b)
            .unwrap();

        let lx_parts = rotation_sets
            .into_iter()
            .map(|(points, poly, evals)| -> DeviceResult<_> {
                let eval = eval_polynomial_st(&evals[..], u);
                let coeff =
                    crate::cuda::bn254::pick_from_buf::<C::Scalar>(device, &poly, 0, 0, size)?
                        - eval;
                device.copy_from_host_to_device(&poly, &[coeff])?;

                let diffs: Vec<C::Scalar> = super_point_set
                    .iter()
                    .filter(|point| !points.contains(point))
                    .copied()
                    .collect();

                assert_eq!(diffs.len() + points.len(), super_point_set.len());

                let z_i = diffs
                    .iter()
                    .map(|root| u - root)
                    .reduce(|a, b| a * b)
                    .unwrap();
                Ok((poly, z_i))
            })
            .collect::<DeviceResult<Vec<_>>>()?;

        let z_diff_0_inv = lx_parts[0].1.invert().unwrap();

        let fz_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let z_buf = device.alloc_device_buffer::<C::Scalar>(1)?;
        for (_, (poly_buf, z_i)) in lx_parts.into_iter().enumerate() {
            device.copy_from_host_to_device(&z_buf, &[z_i][..])?;
            field_op_v3(
                device,
                &fz_buf,
                Some(&fz_buf),
                Some(&v_buf),
                Some(&poly_buf),
                Some(&z_buf),
                size,
                FieldOp::Add,
                None,
            )?;
        }

        let zt_eval_buf = v_buf;
        device.copy_from_host_to_device(&zt_eval_buf, &[zt_eval][..])?;
        field_op_v3(
            device,
            &fz_buf,
            Some(&fz_buf),
            None,
            Some(&hx_buf),
            Some(&zt_eval_buf),
            size,
            FieldOp::Sub,
            None,
        )?;
        device.synchronize()?;

        let z_diff_0_inv_buf = zt_eval_buf;
        device.copy_from_host_to_device(&z_diff_0_inv_buf, &[z_diff_0_inv][..])?;
        field_op_v3(
            device,
            &fz_buf,
            Some(&fz_buf),
            Some(&z_diff_0_inv_buf),
            None,
            None,
            size,
            FieldOp::UOp,
            None,
        )?;

        let mut lx = Vec::new_in(HugePageAllocator);
        lx.resize(size, C::Scalar::zero());

        device.copy_from_device_to_host(&mut lx[..], &fz_buf)?;

        {
            let z = u;
            let mut tmp = *lx.last().unwrap();
            *lx.last_mut().unwrap() = C::Scalar::zero();
            for i in (1..lx.len() - 1).rev() {
                let p = lx[i] + tmp * z;
                lx[i] = tmp;
                tmp = p;
            }
            lx[0] = tmp;
        }

        let commitments = batch_msm(&g_buf, s_buf, vec![&lx[..]], size)?;
        for commitment in commitments {
            transcript.write_point(commitment).unwrap();
        }

        Ok(())
    }
}

pub(crate) fn permutation_product_open<'a, C: CurveAffine>(
    pk: &'a ProvingKey<C>,
    products: &'a [Vec<C::Scalar, HugePageAllocator>],
    x: C::Scalar,
) -> impl Iterator<Item = ProverQuery<'a, C::Scalar>> + Clone {
    let blinding_factors = pk.vk.cs.blinding_factors();
    let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());
    let x_last = pk
        .vk
        .domain
        .rotate_omega(x, Rotation(-((blinding_factors + 1) as i32)));

    iter::empty()
        .chain(products.iter().flat_map(move |product| {
            iter::empty()
                .chain(Some(ProverQuery {
                    point: x,
                    rotation: Rotation::cur(),
                    poly: &product,
                }))
                .chain(Some(ProverQuery {
                    point: x_next,
                    rotation: Rotation::next(),
                    poly: &product,
                }))
        }))
        .chain(products.iter().rev().skip(1).flat_map(move |product| {
            Some(ProverQuery {
                point: x_last,
                rotation: Rotation(-((blinding_factors + 1) as i32)),
                poly: &product,
            })
        }))
}

pub(crate) fn lookup_open<'a, C: CurveAffine>(
    pk: &'a ProvingKey<C>,
    lookup: (&'a [C::Scalar], &'a [C::Scalar], &'a [C::Scalar]),
    x: C::Scalar,
) -> impl Iterator<Item = ProverQuery<'a, C::Scalar>> + Clone {
    let x_inv = pk.vk.domain.rotate_omega(x, Rotation::prev());
    let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());

    let (permuted_input, permuted_table, z) = lookup;

    iter::empty()
        // Open lookup product commitments at x
        .chain(Some(ProverQuery {
            point: x,
            rotation: Rotation::cur(),
            poly: z,
        }))
        // Open lookup input commitments at x
        .chain(Some(ProverQuery {
            point: x,
            rotation: Rotation::cur(),
            poly: permuted_input,
        }))
        // Open lookup table commitments at x
        .chain(Some(ProverQuery {
            point: x,
            rotation: Rotation::cur(),
            poly: permuted_table,
        }))
        // Open lookup input commitments at x_inv
        .chain(Some(ProverQuery {
            point: x_inv,
            rotation: Rotation::prev(),
            poly: permuted_input,
        }))
        // Open lookup product commitments at x_next
        .chain(Some(ProverQuery {
            point: x_next,
            rotation: Rotation::next(),
            poly: z,
        }))
}

pub(crate) fn shuffle_open<'a, C: CurveAffine>(
    pk: &'a ProvingKey<C>,
    shuffle_z: &'a [C::Scalar],
    x: C::Scalar,
) -> impl Iterator<Item = ProverQuery<'a, C::Scalar>> + Clone {
    let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());

    iter::empty()
        // Open lookup product commitments at x
        .chain(Some(ProverQuery {
            point: x,
            rotation: Rotation::cur(),
            poly: shuffle_z,
        }))
        // Open lookup product commitments at x_next
        .chain(Some(ProverQuery {
            point: x_next,
            rotation: Rotation::next(),
            poly: shuffle_z,
        }))
}
