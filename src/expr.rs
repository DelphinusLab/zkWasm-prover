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
    exprs: &[Expression<F>],
    challenge: Option<F>,
    theta: F,
) -> Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)> {
    let composed_expr = compose_tuple_expressions(exprs, challenge, theta);
    flatten_prove_expression(composed_expr)
}

pub(crate) fn flatten_lookup_expression<F: FieldExt>(
    input: &Vec<Expression<F>>,
    table: &Vec<Expression<F>>,
    beta: F,
    gamma: F,
    theta: F,
) -> [Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>; 2] {
    [
        flatten_tuple_expressions(&input[..], Some(beta), theta),
        flatten_tuple_expressions(&table[..], Some(gamma), theta),
    ]
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
pub fn _evaluate_exprs<F: FieldExt>(
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

    let chunks = 12;
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

fn pick_prove_unit_slice<'a, F: FieldExt>(
    unit: &ProveExpressionUnit,
    fixed: &'a [&'a [F]],
    advice: &'a [&'a [F]],
    instance: &'a [&'a [F]],
) -> (&'a [F], i32) {
    match unit {
        ProveExpressionUnit::Fixed {
            column_index,
            rotation,
        } => (&fixed[*column_index], rotation.0),
        ProveExpressionUnit::Advice {
            column_index,
            rotation,
        } => (&advice[*column_index], rotation.0),
        ProveExpressionUnit::Instance {
            column_index,
            rotation,
        } => (&instance[*column_index], rotation.0),
    }
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

pub(crate) fn evaluate_exprs_in_gpu<F: FieldExt>(
    device: &CudaDevice,
    advice_buffer: &HashMap<usize, CudaDeviceBufRaw>,
    exprs: &[&[Expression<F>]],
    fixed: &[&[F]],
    advice: &[&[F]],
    instance: &[&[F]],
    theta: F,
    outputs: &mut [&mut [F]],
    size: usize,
) -> DeviceResult<Vec<CudaDeviceBufRaw>> {
    let mut unit_buffers = BTreeMap::new();
    let mut handlers = vec![];

    let (copy_sw, copy_stream) = CudaStreamWrapper::new_with_inner();
    let (calc_sw, calc_stream) = CudaStreamWrapper::new_with_inner();

    for (i, raw_expr) in exprs.into_iter().enumerate() {
        let expr = flatten_tuple_expressions(raw_expr, None, theta);

        let mut coeffs = vec![];
        for (_, y_map) in expr.iter() {
            assert_eq!(y_map.len(), 1);
            for (k, v) in y_map {
                assert_eq!(*k, 0);
                coeffs.push(*v);
            }
        }

        let coeffs_buf = device.alloc_device_buffer_from_slice_async(&coeffs[..], calc_stream)?;

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
                    } => (advice_buffer.get(&column_index).unwrap(), rotation.0),
                    ProveExpressionUnit::Instance {
                        column_index,
                        rotation,
                    } => (
                        advice_buffer.get(&(column_index + advice.len())).unwrap(),
                        rotation.0,
                    ),
                    _ => {
                        let (values, rot) = pick_prove_unit_slice(&unit, fixed, advice, instance);
                        let buf = unit_buffers.entry(unit.clone()).or_insert_with(move || {
                            device
                                .alloc_device_buffer_from_slice_async(values, copy_stream)
                                .unwrap()
                        });
                        (&*buf, rot)
                    }
                };
                for _ in 0..exp {
                    terms.push(buf.ptr());
                    rots.push(rot);
                }
            }

            terms.push(0usize as _);
        }

        let terms_buf = device.alloc_device_buffer_from_slice_async(&terms[..], calc_stream)?;
        let rots_buf = device.alloc_device_buffer_from_slice_async(&rots[..], calc_stream)?;
        let res = device.alloc_device_buffer_async::<F>(size, &calc_sw)?;

        copy_sw.sync();
        let err = unsafe {
            field_op_batch_mul_sum(
                res.ptr(),
                terms_buf.ptr(),
                rots_buf.ptr(),
                terms.len() as i32,
                size as i32,
                calc_stream,
            )
        };

        to_result((), err, "fail to run field_op_batch_mul_sum")?;

        device.copy_from_device_to_host_async(outputs[i], &res, calc_stream)?;
        handlers.push((res, coeffs_buf, terms_buf, rots_buf))
    }

    let mut res = vec![];

    for (buf, _, _, _) in handlers {
        res.push(buf);
    }

    Ok(res)
}
