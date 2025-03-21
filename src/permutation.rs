use std::sync::Arc;

use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::pairing::group::ff::BatchInvert;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::Column;
use halo2_proofs::plonk::ProvingKey;
use rayon::iter::*;
use rayon::prelude::*;

use crate::fill_random;
use crate::hugetlb::HugePageAllocator;

pub(crate) fn generate_permutation_product<C: CurveAffine>(
    pk: &ProvingKey<C>,
    instances: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    permutations: Vec<Vec<C::Scalar, HugePageAllocator>>,
    beta: C::Scalar,
    gamma: C::Scalar,
    size: usize,
    unusable_rows_start: usize,
) -> Vec<Vec<<C as CurveAffine>::ScalarExt, HugePageAllocator>> {
    let chunk_size = size >> 8;

    let chunk_len = &pk.vk.cs.degree() - 2;
    let omega = pk.get_vk().domain.get_omega();

    let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

    let f = |column: Column<Any>| match column.column_type() {
        Any::Advice => advice_ref,
        Any::Fixed => fixed_ref,
        Any::Instance => instance_ref,
    };

    let mut p_z = pk
        .vk
        .cs
        .permutation
        .columns
        .chunks(chunk_len)
        .zip((&pk).permutation.permutations.chunks(chunk_len))
        .zip(permutations)
        .enumerate()
        .map(|(i, ((columns, permutations), mut modified_values))| {
            let mut delta_omega = C::Scalar::DELTA.pow_vartime([i as u64 * chunk_len as u64]);
            // Iterate over each column of the permutation
            for (j, (&column, permuted_column_values)) in
                columns.iter().zip(permutations.iter()).enumerate()
            {
                let values = f(column);
                modified_values
                    .par_chunks_mut(chunk_size)
                    .zip(permuted_column_values.par_chunks(chunk_size))
                    .zip(values[column.index()].par_chunks(chunk_size))
                    .for_each(|((res, p), v)| {
                        for i in 0..chunk_size {
                            if j == 0 {
                                res[i] = beta * p[i] + &gamma + v[i];
                            } else {
                                res[i] *= &(beta * p[i] + &gamma + v[i]);
                            }
                        }
                    });
            }

            // Invert to obtain the denominator for the permutation product polynomial
            modified_values.par_chunks_mut(chunk_size).for_each(|x| {
                x.iter_mut().batch_invert();
            });

            // Iterate over each column again, this time finishing the computation
            // of the entire fraction by computing the numerators
            for &column in columns.iter() {
                let values = f(column);
                modified_values
                    .par_chunks_mut(chunk_size)
                    .zip(values[column.index()].par_chunks(chunk_size))
                    .enumerate()
                    .for_each(|(idx, (res, v))| {
                        let mut delta_omega =
                            delta_omega * omega.pow_vartime([(idx * chunk_size) as u64]) * &beta;
                        for i in 0..chunk_size {
                            res[i] *= &(delta_omega + &gamma + v[i]);
                            delta_omega *= &omega;
                        }
                    });

                delta_omega *= &C::Scalar::DELTA;
            }

            modified_values
        })
        .collect::<Vec<_>>();

    let mut tails: Vec<_> = p_z
        .iter_mut()
        .map(|z| {
            let mut tails = z
                .par_chunks_mut(chunk_size)
                .map(|z| {
                    let mut tmp = C::Scalar::one();
                    for v in z {
                        std::mem::swap(&mut tmp, v);
                        tmp = tmp * *v;
                    }
                    tmp
                })
                .collect::<Vec<_>>();

            for i in 1..tails.len() {
                tails[i] = tails[i] * tails[i - 1];
            }

            z.par_chunks_mut(chunk_size)
                .skip(1)
                .zip(tails.par_iter())
                .map(|(z, tail)| {
                    for v in z {
                        *v = *v * tail;
                    }
                })
                .collect::<Vec<_>>();

            fill_random(&mut z[unusable_rows_start + 1..]);
            z[unusable_rows_start]
        })
        .collect();

    for i in 1..tails.len() {
        tails[i] = tails[i] * tails[i - 1];
    }

    p_z.iter_mut().skip(1).enumerate().for_each(|(i, z)| {
        z[0..=unusable_rows_start]
            .par_chunks_mut(chunk_size)
            .for_each(|z| {
                for row in z {
                    *row = *row * tails[i];
                }
            });
    });

    p_z
}
