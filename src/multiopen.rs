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
use rayon::iter::IntoParallelIterator as _;
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

    let ws = commitment_data
        .into_par_iter()
        .map(|commitment_at_a_point| -> DeviceResult<_> {
            //println!("queries {}", commitment_at_a_point.queries.len());
            let poly_batch_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let c_buf = device.alloc_device_buffer_from_slice(&[v][..])?;

            let mut poly_batch = Vec::new_in(HugePageAllocator);
            poly_batch.resize(size, C::Scalar::zero());

            let z = commitment_at_a_point.point;

            device
                .copy_from_host_to_device(&poly_batch_buf, commitment_at_a_point.queries[0].poly)?;
            for query in commitment_at_a_point.queries.iter().skip(1) {
                assert_eq!(query.point, z);
                device.copy_from_host_to_device(&tmp_buf, query.poly)?;

                field_op_v3(
                    device,
                    &poly_batch_buf,
                    Some(&poly_batch_buf),
                    Some(&c_buf),
                    Some(&tmp_buf),
                    None,
                    size,
                    FieldOp::Sum,
                    None,
                )?;
            }

            device.copy_from_device_to_host(&mut poly_batch[..], &poly_batch_buf)?;
            let eval_batch = halo2_proofs::arithmetic::eval_polynomial_st(
                &poly_batch,
                commitment_at_a_point.queries[0].point,
            );
            poly_batch[0] -= eval_batch;

            let mut tmp = *poly_batch.last().unwrap();
            *poly_batch.last_mut().unwrap() = C::ScalarExt::zero();
            for i in (1..poly_batch.len() - 1).rev() {
                let p = poly_batch[i] + tmp * z;
                poly_batch[i] = tmp;
                tmp = p;
            }
            poly_batch[0] = tmp;

            Ok(poly_batch)
        })
        .collect::<DeviceResult<Vec<_>>>()?;

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
