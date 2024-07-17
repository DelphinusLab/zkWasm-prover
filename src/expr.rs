use crate::cuda::bn254_c::field_op_batch_mul_sum;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::Device;
use crate::device::DeviceResult;
use crate::to_result;
use crate::CudaDevice;
use crate::CudaDeviceBufRaw;

use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::evaluation_gpu::Bop;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Expression;
use rayon::prelude::*;
use std::collections::BTreeMap;

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

fn flatten_tuple_expressions<F: FieldExt>(
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
pub fn evaluate_exprs<F: FieldExt>(
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

    let chunks = 32;
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

pub(crate) fn evaluate_exprs_in_gpu<F: FieldExt>(
    device: &CudaDevice,
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
    for (i, expr) in exprs.into_iter().enumerate() {
        let expr = flatten_tuple_expressions(expr, None, theta);

        let stream_wrapper = CudaStreamWrapper::new();
        let stream = (&stream_wrapper).into();
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
        for (i, (units, _)) in expr.into_iter().enumerate() {
            terms.push(unsafe {
                coeffs_buf
                    .ptr()
                    .offset((i * core::mem::size_of::<F>()) as isize)
            });

            for (unit, exp) in units {
                let (values, rot) = pick_prove_unit_slice(&unit, fixed, advice, instance);
                let buf = unit_buffers.entry(unit).or_insert_with(|| {
                    device
                        .alloc_device_buffer_from_slice_async(values, stream)
                        .unwrap()
                });
                for _ in 0..exp {
                    terms.push(buf.ptr());
                    rots.push(rot);
                }
            }

            terms.push(0usize as _);
        }

        let terms_buf = device.alloc_device_buffer_from_slice_async(&terms[..], stream)?;
        let rots_buf = device.alloc_device_buffer_from_slice_async(&rots[..], stream)?;
        let res = device.alloc_device_buffer::<F>(size)?;

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

        device.copy_from_device_to_host_async(outputs[i], &res, stream)?;
        handlers.push((res, terms_buf, rots_buf, stream_wrapper))
    }

    let mut res = vec![];

    for (buf, _, _, _) in handlers {
        res.push(buf);
    }

    Ok(res)
}
