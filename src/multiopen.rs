use std::iter;

use ark_std::end_timer;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator as _;
use std::collections::BTreeMap;

use crate::cuda::bn254::field_op_v3;
use crate::cuda::bn254::FieldOp;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::Device as _;
use crate::device::DeviceResult;
use crate::hugetlb::HugePageAllocator;

#[derive(Debug, Clone, Copy)]
pub struct ProverQuery<'a, F: FieldExt> {
    pub point: F,
    pub rotation: Rotation,
    pub poly: &'a [F],
}

struct CommitmentData<'a, F: FieldExt> {
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

pub(crate) fn multiopen<'a, I, C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    device: &CudaDevice,
    g_buf: &CudaDeviceBufRaw,
    queries: I,
    size: usize,
    s_buf: [&CudaDeviceBufRaw; 2],
    transcript: &mut T,
) -> DeviceResult<()>
where
    I: IntoIterator<Item = ProverQuery<'a, C::Scalar>>,
{
    let v: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
    let commitment_data = construct_intermediate_sets(queries);

    let mut collection = BTreeMap::new();
    let mut bufs = vec![];
    for (rot_idx, data) in commitment_data.iter().enumerate() {
        bufs.push(device.alloc_device_buffer::<C::ScalarExt>(size)?);
        let len = data.queries.len();
        for (inner_idx, q) in data.queries.iter().enumerate() {
            collection
                .entry(q.poly.as_ptr() as usize)
                .and_modify(|x: &mut (_, Vec<_>)| x.1.push((rot_idx, len - 1 - inner_idx)))
                .or_insert((q.poly, vec![(rot_idx, len - 1 - inner_idx)]));
        }
    }

    println!("collection has len {}", collection.len());

    let mut vs = vec![C::ScalarExt::one(), v];
    let tmp_buf = device.alloc_device_buffer::<C::ScalarExt>(size)?;
    let v_buf = device.alloc_device_buffer::<C::ScalarExt>(1)?;
    for (_, (poly, assoc)) in collection {
        device.copy_from_host_to_device_async(&tmp_buf, poly, 0usize as _)?;
        for (rot_idx, inner_idx) in assoc {
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

    ws.par_iter_mut().zip(commitment_data.par_iter()).for_each(
        |(poly_batch, commitment_at_a_point)| {
            let z = commitment_at_a_point.point;
            let eval_batch = halo2_proofs::arithmetic::eval_polynomial_st(&poly_batch, z);
            poly_batch[0] -= eval_batch;

            let mut tmp = *poly_batch.last().unwrap();
            *poly_batch.last_mut().unwrap() = C::ScalarExt::zero();
            for i in (1..poly_batch.len() - 1).rev() {
                let p = poly_batch[i] + tmp * z;
                poly_batch[i] = tmp;
                tmp = p;
            }
            poly_batch[0] = tmp;
        },
    );

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
