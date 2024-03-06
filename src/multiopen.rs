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

pub struct CommitmentData<'a, F: FieldExt> {
    queries: Vec<ProverQuery<'a, F>>,
    point: F,
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

    use crate::cuda::bn254::field_op_v3;
    use crate::cuda::bn254::FieldOp;
    use crate::device::cuda::CudaDevice;
    use crate::device::cuda::CudaDeviceBufRaw;
    use crate::device::Device as _;
    use crate::device::DeviceResult;
    use crate::hugetlb::HugePageAllocator;
    use crate::multiopen::CommitmentData;
    use crate::multiopen::ProverQuery;

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
                    FieldOp::Sum,
                    None,
                )?;

                let eval = eval_map.get(&(poly.as_ptr() as usize, x));
                eval_batch[rot_idx] += eval.cloned().unwrap_or(C::Scalar::zero()) * vs[inner_idx];
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

        let commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_buf,
            s_buf,
            ws.iter().map(|x| &x[..]).collect(),
            size,
        )?;
        for commitment in commitments {
            transcript.write_point(commitment).unwrap();
        }

        end_timer!(timer);

        Ok(())
    }
}

pub mod shplonk {
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
    use std::collections::BTreeSet;

    use crate::cuda::bn254::field_op_v3;
    use crate::cuda::bn254::FieldOp;
    use crate::device::cuda::CudaDevice;
    use crate::device::cuda::CudaDeviceBufRaw;
    use crate::device::Device as _;
    use crate::device::DeviceResult;
    use crate::hugetlb::HugePageAllocator;
    use crate::multiopen::CommitmentData;
    use crate::multiopen::ProverQuery;
    /*
       fn construct_intermediate_sets<'a, F: FieldExt, I>(queries: I) -> Vec<CommitmentData<'a, F>>
       where
           I: IntoIterator<Item = ProverQuery<'a, F>>,
       {
           let queries = queries.into_iter().collect::<Vec<_>>();

       // Find evaluation of a commitment at a rotation
       let get_eval = |commitment: Q::Commitment, rotation: Rotation| -> F {
           queries
               .iter()
               .find(|query| query.get_commitment() == commitment && query.get_rotation() == rotation)
               .unwrap()
               .get_eval()
       };

       // Order points according to their rotation
       let mut rotation_point_map = BTreeMap::new();
       for query in queries.clone() {
           let point = rotation_point_map
               .entry(query.get_rotation())
               .or_insert_with(|| query.get_point());

           // Assert rotation point matching consistency
           assert_eq!(*point, query.get_point());
       }
       // All points appear in queries
       let super_point_set: Vec<F> = rotation_point_map.values().cloned().collect();

       // Collect rotation sets for each commitment
       // Example elements in the vector:
       // (C_0, {r_5}),
       // (C_1, {r_1, r_2, r_3}),
       // (C_2, {r_2, r_3, r_4}),
       // (C_3, {r_2, r_3, r_4}),
       // ...
       let mut commitment_rotation_set_map: Vec<(Q::Commitment, BTreeSet<Rotation>)> = vec![];
       for query in queries.clone() {
           let rotation = query.get_rotation();
           if let Some(pos) = commitment_rotation_set_map
               .iter()
               .position(|(commitment, _)| *commitment == query.get_commitment())
           {
               let (_, rotation_set) = &mut commitment_rotation_set_map[pos];
               rotation_set.insert(rotation);
           } else {
               let rotation_set = BTreeSet::from([rotation]);
               commitment_rotation_set_map.push((query.get_commitment(), rotation_set));
           };
       }

       // Flatten rotation sets and collect commitments that opens against each commitment set
       // Example elements in the vector:
       // {r_5}: [C_0],
       // {r_1, r_2, r_3} : [C_1]
       // {r_2, r_3, r_4} : [C_2, C_3],
       // ...
       let mut rotation_set_commitment_map = BTreeMap::<BTreeSet<_>, Vec<Q::Commitment>>::new();
       for (commitment, rotation_set) in commitment_rotation_set_map.iter() {
           let commitments = rotation_set_commitment_map
               .entry(rotation_set.clone())
               .or_insert_with(Vec::new);
           if !commitments.contains(commitment) {
               commitments.push(commitment.clone());
           }
       }

       let rotation_sets = rotation_set_commitment_map
           .into_iter()
           .map(|(rotation_set, commitments)| {
               let rotations: Vec<Rotation> = rotation_set.iter().cloned().collect();

               let commitments: Vec<Commitment<F, Q::Commitment>> = commitments
                   .iter()
                   .map(|commitment| {
                       let evals: Vec<F> = rotations
                           .iter()
                           .map(|rotation| get_eval(commitment.clone(), *rotation))
                           .collect();
                       Commitment((commitment.clone(), evals))
                   })
                   .collect();

               RotationSet {
                   commitments,
                   points: rotations
                       .iter()
                       .map(|rotation| *rotation_point_map.get(rotation).unwrap())
                       .collect(),
               }
           })
           .collect::<Vec<RotationSet<_, _>>>();

       IntermediateSets {
           rotation_sets,
           super_point_set,
       }
       }
    */
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
