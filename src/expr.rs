use crate::cuda::bn254_c::field_op_batch_mul_sum;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::DeviceResult;
use crate::to_result;
use crate::CudaDevice;
use crate::CudaDeviceBufRaw;

use ark_std::iterable::Iterable;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::evaluation_gpu::Bop;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Expression;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

pub(crate) fn is_expression_pure_unit<F: FieldExt>(x: &Expression<F>) -> bool {
    x.is_constant().is_some()
        || x.is_pure_fixed().is_some()
        || x.is_pure_advice().is_some()
        || x.is_pure_instance().is_some()
}

fn flatten_prove_expression<F: FieldExt>(
    prove_expression: ProveExpression<F>,
) -> Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)> {
    prove_expression
        .flatten()
        .into_iter()
        .map(|(prove_expr_units, coeff)| {
            let mut units_exp_map = BTreeMap::new();
            for unit in prove_expr_units {
                units_exp_map
                    .entry(unit)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
            (units_exp_map, coeff)
        })
        .collect::<Vec<_, _>>()
}

fn compose_tuple_expressions<F: FieldExt>(
    exprs: &[Expression<F>],
    challenge: Option<F>,
    theta: F,
) -> ProveExpression<F> {
    let mut composed_expr = ProveExpression::<F>::from_expr(&exprs[0]);

    for expr in exprs.iter().skip(1) {
        composed_expr = ProveExpression::Scale(
            Box::new(composed_expr),
            BTreeMap::from_iter([(0, theta)].into_iter()),
        );
        composed_expr = ProveExpression::Op(
            Box::new(composed_expr),
            Box::new(ProveExpression::<F>::from_expr(expr)),
            Bop::Sum,
        );
    }

    if let Some(challenge) = challenge {
        composed_expr = ProveExpression::Op(
            Box::new(composed_expr),
            Box::new(ProveExpression::Y(BTreeMap::from_iter(
                [(0, challenge)].into_iter(),
            ))),
            Bop::Sum,
        );
    }

    composed_expr
}

pub(crate) fn flatten_tuple_expressions<F: FieldExt>(
    expr: &[Expression<F>],
    challenge: Option<F>,
    theta: F,
) -> Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)> {
    let composed_expr = compose_tuple_expressions(expr, challenge, theta);
    flatten_prove_expression(composed_expr)
}

pub(crate) fn flatten_lookup_input_expression<F: FieldExt>(
    inputs_sets: &Vec<Vec<Vec<Expression<F>>>>,
    beta: F,
    theta: F,
) -> [Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>>; 2] {
    let input_sets_expr = inputs_sets
        .iter()
        .map(|set| {
            set.iter()
                .map(|input| compose_tuple_expressions(input, Some(beta), theta))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let expr_input_product = input_sets_expr
        .iter()
        .map(|set| {
            let product = set.iter().skip(1).fold(set[0].clone(), |acc, v| {
                ProveExpression::Op(Box::new(acc.clone()), Box::new(v.clone()), Bop::Product)
            });
            flatten_prove_expression(product)
        })
        .collect::<Vec<_>>();

    let expr_input_product_sum = input_sets_expr
        .iter()
        .map(|set| {
            let product_sum = if set.len() > 1 {
                (0..set.len())
                    .map(|i| {
                        set.iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, v)| v.clone())
                            .reduce(|acc, v| {
                                ProveExpression::Op(
                                    Box::new(acc.clone()),
                                    Box::new(v.clone()),
                                    Bop::Product,
                                )
                            })
                            .unwrap()
                    })
                    .reduce(|acc, v| {
                        ProveExpression::Op(Box::new(acc.clone()), Box::new(v.clone()), Bop::Sum)
                    })
                    .unwrap()
            } else {
                ProveExpression::Y(BTreeMap::from_iter([(0, F::one())].into_iter()))
            };

            flatten_prove_expression(product_sum)
        })
        .collect::<Vec<_>>();
    [expr_input_product, expr_input_product_sum]
}

pub(crate) fn flatten_lookup_expression<F: FieldExt>(
    inputs_sets: &Vec<Vec<Vec<Expression<F>>>>,
    table: &Vec<Expression<F>>,
    beta: F,
    theta: F,
) -> (
    Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>>,
    Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>>,
    Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>,
) {
    let flatten_table = flatten_tuple_expressions(&table[..], Some(beta), theta);
    let [flatten_input_product, flatten_input_product_sum] =
        flatten_lookup_input_expression(inputs_sets, beta, theta);

    (
        flatten_input_product,
        flatten_input_product_sum,
        flatten_table,
    )
}

pub(crate) fn flatten_shuffle_expression<F: FieldExt>(
    inputs: &Vec<Vec<Expression<F>>>,
    tables: &Vec<Vec<Expression<F>>>,
    beta: F,
    theta: F,
) -> [Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>; 2] {
    let flatten_expr = |exprs: &Vec<Vec<Expression<F>>>| {
        let composed_expr = exprs
            .iter()
            .enumerate()
            .map(|(i, exprs)| {
                compose_tuple_expressions(exprs, Some(beta.pow_vartime([1 + i as u64])), theta)
            })
            .reduce(|acc, new| {
                ProveExpression::Op(Box::new(acc.clone()), Box::new(new.clone()), Bop::Product)
            })
            .unwrap();

        flatten_prove_expression(composed_expr)
    };

    let expr_input = flatten_expr(inputs);
    let expr_table = flatten_expr(tables);

    [expr_input, expr_table]
}

/// Simple evaluation of an expression
pub(crate) fn evaluate_exprs<F: FieldExt>(
    expressions: &[Expression<F>],
    size: usize,
    rot_scale: i32,
    fixed: &[&[F]],
    advice: &[&[F]],
    instance: &[&[F]],
    theta: F,
    res: &mut [F],
) {
    let isize = size as i32;
    let get_rotation_idx = |idx: usize, rot: i32, rot_scale: i32, isize: i32| -> usize {
        (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
    };

    let chunks = 8;
    let chunk_size = size / chunks;

    res.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, res_chunck)| {
            for (i, value) in res_chunck.into_iter().enumerate() {
                let idx = chunk_idx * chunk_size + i;
                for (i, expression) in expressions.iter().enumerate() {
                    let v = expression.evaluate(
                        &|scalar| scalar,
                        &|_| panic!("virtual selectors are removed during optimization"),
                        &|_, column_index, rotation| {
                            fixed[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                        },
                        &|_, column_index, rotation| {
                            advice[column_index]
                                [get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                        },
                        &|_, column_index, rotation| {
                            instance[column_index]
                                [get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                        },
                        &|a| -a,
                        &|a, b| a + &b,
                        &|a, b| {
                            let a = a();
                            if a == F::zero() {
                                a
                            } else {
                                a * b()
                            }
                        },
                        &|a, scalar| a * scalar,
                    );

                    if i > 0 && *value != F::zero() {
                        *value = *value * theta;
                        *value += v;
                    } else {
                        *value = v;
                    }
                }
            }
        });
}

pub(crate) fn evaluate_exprs_in_gpu_pure<F: FieldExt>(
    device: &CudaDevice,
    advice_and_instance_buffers: &HashMap<usize, CudaDeviceBufRaw>,
    fixed_buffers: &mut HashMap<usize, CudaDeviceBufRaw>,
    exprs: &[Expression<F>],
    fixed: &[&[F]],
    theta: F,
    advices_len: usize,
    size: usize,
) -> DeviceResult<CudaDeviceBufRaw> {
    let (sw, stream) = CudaStreamWrapper::new_with_inner();
    let expr = flatten_tuple_expressions(exprs, None, theta);

    let mut coeffs = vec![];
    for (_, y_map) in expr.iter() {
        assert_eq!(y_map.len(), 1);
        for (k, v) in y_map {
            assert_eq!(*k, 0);
            coeffs.push(*v);
        }
    }

    let coeffs_buf = device.alloc_device_buffer_from_slice_async(&coeffs[..], stream)?;

    let mut terms = vec![]; // Array of polynomial terms [coeff0, x0, y0, ..., nil, coeff1, x1, y1, ...]
    let mut rots = vec![];
    for (j, (units, _)) in expr.into_iter().enumerate() {
        terms.push(unsafe {
            coeffs_buf
                .ptr()
                .offset((j * core::mem::size_of::<F>()) as isize)
        });

        for (unit, exp) in units {
            let (buf, rot) = match unit {
                ProveExpressionUnit::Advice {
                    column_index,
                    rotation,
                } => (
                    advice_and_instance_buffers.get(&column_index).unwrap(),
                    rotation.0,
                ),
                ProveExpressionUnit::Instance {
                    column_index,
                    rotation,
                } => (
                    advice_and_instance_buffers
                        .get(&(column_index + &advices_len))
                        .unwrap(),
                    rotation.0,
                ),
                ProveExpressionUnit::Fixed {
                    column_index,
                    rotation,
                } => {
                    if !fixed_buffers.contains_key(&column_index) {
                        let buf = device.alloc_device_buffer_from_slice_async(
                            &fixed[column_index][..],
                            stream,
                        )?;
                        fixed_buffers.insert(column_index, buf);
                    }
                    (fixed_buffers.get(&(column_index)).unwrap(), rotation.0)
                }
            };
            for _ in 0..exp {
                terms.push(buf.ptr());
                rots.push(rot);
            }
        }

        terms.push(0usize as _);
    }

    let terms_buf = device.alloc_device_buffer_from_slice_async(&terms[..], stream)?;
    let rots_buf = device.alloc_device_buffer_from_slice_async(&rots[..], stream)?;
    let res = device.alloc_device_buffer_async::<F>(size, &sw)?;
    let err = unsafe {
        field_op_batch_mul_sum(
            res.ptr(),
            terms_buf.ptr(),
            rots_buf.ptr(),
            terms.len() as i32,
            size as i32,
            stream,
        )
    };

    to_result((), err, "fail to run field_op_batch_mul_sum")?;

    sw.sync();

    Ok(res)
}

pub(crate) fn load_fixed_buffer_into_gpu_async<F: FieldExt>(
    device: &CudaDevice,
    fixed_buffers: &mut HashMap<usize, CudaDeviceBufRaw>,
    exprs: &[Expression<F>],
    fixed: &[&[F]],
    sw: &CudaStreamWrapper,
) -> DeviceResult<()> {
    for e in exprs {
        let pe = ProveExpression::<F>::from_expr(e);
        for (units, _) in flatten_prove_expression(pe) {
            for (unit, _) in units {
                match unit {
                    ProveExpressionUnit::Fixed { column_index, .. }
                        if !fixed_buffers.contains_key(&column_index) =>
                    {
                        let buf = device.alloc_device_buffer_from_slice_async(
                            &fixed[column_index][..],
                            sw.into(),
                        )?;
                        fixed_buffers.insert(column_index, buf);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn get_expr_degree<F: FieldExt>(expr: &Vec<Expression<F>>) -> usize {
    expr.iter().map(|x| x.degree()).max().unwrap()
}
