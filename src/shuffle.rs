use std::sync::Arc;

use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::pairing::group::ff::BatchInvert;
use halo2_proofs::plonk::Expression;
use halo2_proofs::plonk::ProvingKey;
use rayon::iter::*;
use rayon::prelude::*;

use crate::evaluate_exprs;
use crate::fill_random;
use crate::hugetlb::HugePageAllocator;
use crate::is_expression_pure_unit;

pub(crate) fn generate_shuffle_product<C: CurveAffine>(
    pk: &ProvingKey<C>,
    instances: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    shuffles: Vec<Vec<C::Scalar, HugePageAllocator>>,
    beta: C::Scalar,
    theta: C::Scalar,
    size: usize,
    unusable_rows_start: usize,
) -> Vec<Vec<<C as CurveAffine>::ScalarExt, HugePageAllocator>> {
    let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

    let shuffle_groups = pk.vk.cs.shuffles.group(pk.vk.cs.degree());
    let mut more_buffer_groups = shuffle_groups
        .par_iter()
        .map(|group| {
            group
                .0
                .iter()
                .map(|elements| {
                    let input_buffer = if elements.input_expressions.len() == 1
                        && is_expression_pure_unit(&elements.input_expressions[0])
                    {
                        None
                    } else {
                        let mut buffer = Vec::new_in(HugePageAllocator);
                        buffer.resize(size, C::Scalar::zero());
                        Some(buffer)
                    };

                    let shuffle_buffer = if elements.shuffle_expressions.len() == 1
                        && is_expression_pure_unit(&elements.input_expressions[0])
                    {
                        None
                    } else {
                        let mut buffer = Vec::new_in(HugePageAllocator);
                        buffer.resize(size, C::Scalar::zero());
                        Some(buffer)
                    };
                    (input_buffer, shuffle_buffer)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let f = |expr: &Expression<_>| {
        if let Some(_) = expr.is_constant() {
            unreachable!()
        } else if let Some(idx) = expr.is_pure_fixed() {
            &fixed_ref[idx][..]
        } else if let Some(idx) = expr.is_pure_instance() {
            &instance_ref[idx][..]
        } else if let Some(idx) = expr.is_pure_advice() {
            &advice_ref[idx][..]
        } else {
            unreachable!()
        }
    };

    let buffer_groups = shuffle_groups
        .par_iter()
        .zip(more_buffer_groups.par_iter_mut())
        .map(|(group, buffers)| {
            group
                .0
                .par_iter()
                .zip(buffers.par_iter_mut())
                .map(|(elements, buffer)| {
                    let mut tuple_expres = vec![];
                    if !(elements.input_expressions.len() == 1
                        && is_expression_pure_unit(&elements.input_expressions[0]))
                    {
                        tuple_expres
                            .push((&elements.input_expressions[..], buffer.0.as_mut().unwrap()));
                    }
                    if !(elements.shuffle_expressions.len() == 1
                        && is_expression_pure_unit(&elements.shuffle_expressions[0]))
                    {
                        tuple_expres.push((
                            &elements.shuffle_expressions[..],
                            buffer.1.as_mut().unwrap(),
                        ))
                    }
                    tuple_expres.into_par_iter().for_each(|(expr, buffer)| {
                        evaluate_exprs(
                            expr,
                            size,
                            1,
                            fixed_ref,
                            advice_ref,
                            instance_ref,
                            theta,
                            buffer,
                        )
                    });

                    let input_buffer_ref = if elements.input_expressions.len() == 1
                        && is_expression_pure_unit(&elements.input_expressions[0])
                    {
                        f(&elements.input_expressions[0])
                    } else {
                        buffer.0.as_ref().unwrap()
                    };
                    let shuffle_buffer_ref = if elements.shuffle_expressions.len() == 1
                        && is_expression_pure_unit(&elements.shuffle_expressions[0])
                    {
                        f(&elements.shuffle_expressions[0])
                    } else {
                        buffer.1.as_ref().unwrap()
                    };
                    (input_buffer_ref, shuffle_buffer_ref)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut p_z = buffer_groups
        .par_iter()
        .zip(shuffles.into_par_iter())
        .map(|(group, mut modified_values)| {
            let beta_pows: Vec<C::Scalar> = (0..group.len())
                .map(|i| beta.pow_vartime([1 + i as u64, 0, 0, 0]))
                .collect();

            let chunk_size = size >> 3;
            modified_values
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunck_id, values)| {
                    for row in 0..values.len() {
                        for (i, (_, shuffle)) in group.iter().enumerate() {
                            if i == 0 {
                                values[row] = shuffle[chunck_id * chunk_size + row] + beta_pows[i];
                            } else {
                                values[row] *= shuffle[chunck_id * chunk_size + row] + beta_pows[i];
                            }
                        }
                    }
                });

            modified_values.par_chunks_mut(chunk_size).for_each(|x| {
                x.iter_mut().batch_invert();
            });

            modified_values
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunck_id, values)| {
                    for row in 0..values.len() {
                        for (i, (input, _)) in group.iter().enumerate() {
                            values[row] *= input[chunck_id * chunk_size + row] + beta_pows[i];
                        }
                    }
                });

            modified_values
        })
        .collect::<Vec<_>>();

    p_z.par_iter_mut().for_each(|z| {
        let chunks = 4;
        let chunk_size = (size + chunks - 1) / chunks;

        let mut tails = z
            .par_chunks_mut(chunk_size)
            .map(|z| {
                let mut tmp = C::Scalar::one();
                for i in 0..z.len() {
                    std::mem::swap(&mut tmp, &mut z[i]);
                    tmp = tmp * z[i];
                }
                tmp
            })
            .collect::<Vec<_>>();

        for i in 1..tails.len() {
            tails[i] = tails[i] * tails[i - 1];
        }

        z.par_chunks_mut(chunk_size)
            .skip(1)
            .zip(tails.into_par_iter())
            .for_each(|(z, tail)| {
                for x in z.iter_mut() {
                    *x = *x * tail;
                }
            });

        fill_random(&mut z[unusable_rows_start + 1..]);
    });
    p_z
}
