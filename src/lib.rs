#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

use std::sync::Arc;
use std::thread;

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::pairing::group::Group as _;
use halo2_proofs::plonk::lookup;
use halo2_proofs::plonk::Expression;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use hugetlb::HugePageAllocator;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::IntoParallelRefMutIterator as _;
use rayon::iter::ParallelIterator as _;

use crate::cuda::bn254::msm;
use crate::device::cuda::CudaDevice;
use crate::device::Device as _;
use crate::device::DeviceResult;

pub mod cuda;
pub mod device;
mod hugetlb;

#[macro_use]
extern crate lazy_static;

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Vec<Vec<C::Scalar, HugePageAllocator>> {
    let rows = 1 << pk.get_vk().domain.k();
    let columns = pk.get_vk().cs.num_advice_columns;
    let zero = C::Scalar::zero();
    (0..columns)
        .into_par_iter()
        .map(|_| {
            let mut buf = Vec::new_in(HugePageAllocator);
            buf.resize(rows, zero);
            buf
        })
        .collect()
}

#[derive(Debug)]
pub enum Error {
    DeviceError(device::Error),
}

impl From<device::Error> for Error {
    fn from(e: device::Error) -> Self {
        Error::DeviceError(e)
    }
}

pub fn create_proof_from_advices<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: Arc<ProvingKey<C>>,
    instances: &[&[C::Scalar]],
    mut advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
) -> Result<(), Error> {
    let size = 1 << pk.get_vk().domain.k();
    let rot_scale = 1 << (pk.get_vk().domain.extended_k() - pk.get_vk().domain.k());
    let meta = &pk.vk.cs;
    let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

    let timer = start_timer!(|| "create single instances");
    let instance =
        halo2_proofs::plonk::create_single_instances(params, &pk, &[instances], transcript)
            .unwrap();
    let instance = Arc::new(instance);
    end_timer!(timer);

    let device = CudaDevice::get_device(0).unwrap();

    let timer = start_timer!(|| "pin advice memory to gpu");
    unsafe { Arc::get_mut_unchecked(&mut advices) }
        .iter_mut()
        .map(|x| -> Result<(), Error> {
            device.pin_memory(&mut x[..])?;
            Ok(())
        })
        .collect::<Result<_, _>>()?;
    end_timer!(timer);

    // add random value
    if true {
        let named = &pk.vk.cs.named_advices;
        unsafe { Arc::get_mut_unchecked(&mut advices) }
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, advice)| {
                if named.iter().find(|n| n.1 as usize == i).is_none() {
                    for cell in &mut advice[unusable_rows_start..] {
                        *cell = C::Scalar::random(&mut OsRng);
                    }
                }
            });
    }

    let timer = start_timer!(|| format!("copy advice columns to gpu, count {}", advices.len()));
    let advices_device_buf = advices
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<DeviceResult<Vec<_>>>()?;
    end_timer!(timer);

    /*
    let timer =
        start_timer!(|| format!("copy fixed columns to gpu, count {}", pk.fixed_values.len()));
    let fixed_device_buf = pk
        .fixed_values
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<DeviceResult<Vec<_>>>()?;
    end_timer!(timer);
    */

    let timer = start_timer!(|| "copy g_lagrange buffer");
    let g_lagrange_buf = device
        .alloc_device_buffer_from_slice(&params.g_lagrange[..])
        .unwrap();
    end_timer!(timer);

    // thread for part of lookups
    let sub_pk = pk.clone();
    let sub_advices = advices.clone();
    let sub_instance = instance.clone();
    let sub_device = device.clone();
    let lookup_handler = thread::spawn(move || {
        let pk = sub_pk;
        let advices = sub_advices;
        let instance = sub_instance;
        let device = sub_device;
        let timer =
            start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
        let lookups = pk
            .vk
            .cs
            .lookups
            .par_iter()
            .map(|_| {
                let mut permuted_input = Vec::new_in(HugePageAllocator);
                permuted_input.resize(size, C::ScalarExt::zero());
                let mut permuted_table = Vec::new_in(HugePageAllocator);
                permuted_table.resize(size, C::ScalarExt::zero());
                let mut product = Vec::new_in(HugePageAllocator);
                product.resize(size, C::ScalarExt::zero());
                (permuted_input, permuted_table, product)
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let [mut single_unit_lookups, mut single_comp_lookups, mut tuple_lookups] =
            lookup_classify(&pk, lookups);

        let timer = start_timer!(|| format!("permute lookup unit {}", single_unit_lookups.len()));
        single_unit_lookups
            .par_iter_mut()
            .for_each(|(i, (input, table, z))| {
                let f = |expr: &Expression<_>, target: &mut [_]| {
                    if let Some(v) = expr.is_constant() {
                        target.fill(v);
                    } else if let Some(idx) = expr.is_pure_fixed() {
                        target
                            .clone_from_slice(&pk.fixed_values[idx].values[0..unusable_rows_start]);
                    } else if let Some(idx) = expr.is_pure_instance() {
                        target.clone_from_slice(
                            &instance[0].instance_values[idx].values[0..unusable_rows_start],
                        );
                    } else if let Some(idx) = expr.is_pure_advice() {
                        target.clone_from_slice(&advices[idx][0..unusable_rows_start]);
                    } else {
                        unreachable!()
                    }
                };

                f(
                    &pk.vk.cs.lookups[*i].input_expressions[0],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[*i].table_expressions[0],
                    &mut table[0..unusable_rows_start],
                );
                handle_lookup_pair(input, table, unusable_rows_start);
            });
        end_timer!(timer);

        let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instance[0]
            .instance_values
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..];

        let timer = start_timer!(|| format!("permute lookup comp {}", single_comp_lookups.len()));
        single_comp_lookups
            .par_iter_mut()
            .for_each(|(i, (input, table, _z))| {
                let f = |expr: &Expression<_>, target: &mut [_]| {
                    evaluate_expr(
                        expr,
                        size,
                        rot_scale as i32,
                        fixed_ref,
                        advice_ref,
                        instance_ref,
                        target,
                    )
                };

                f(
                    &pk.vk.cs.lookups[*i].input_expressions[0],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[*i].table_expressions[0],
                    &mut table[0..unusable_rows_start],
                );
                handle_lookup_pair(input, table, unusable_rows_start);
            });
        end_timer!(timer);

        return (single_unit_lookups, single_comp_lookups, tuple_lookups);
    });

    // Advice MSM
    let timer = start_timer!(|| format!("advices msm {}", advices_device_buf.len()));
    let msm_result = [C::Curve::identity()];
    let msm_result_buf = device.alloc_device_buffer_from_slice(&msm_result[..])?;
    for s_buf in advices_device_buf.iter() {
        let commitment = msm(&device, &msm_result_buf, &g_lagrange_buf, s_buf, size)?;
        transcript.write_point(commitment).unwrap();
    }
    end_timer!(timer);

    let theta: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("theta is {:?}", theta);

    let timer = start_timer!(|| "wait single lookups");
    let (mut single_unit_lookups, mut single_comp_lookups, mut tuple_lookups) =
        lookup_handler.join().unwrap();
    end_timer!(timer);

    // After theta
    let timer = start_timer!(|| format!("permute lookup tuple {}", tuple_lookups.len()));

    let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let advice_ref = &unsafe { Arc::get_mut_unchecked(&mut advices) }
        .iter()
        .map(|x| &x[..])
        .collect::<Vec<_>>()[..];
    let instance_ref = &instance[0]
        .instance_values
        .iter()
        .map(|x| &x[..])
        .collect::<Vec<_>>()[..];

    tuple_lookups
        .par_iter_mut()
        .for_each(|(i, (input, table, _z))| {
            let f = |expr: &[Expression<_>], target: &mut [_]| {
                evaluate_exprs(
                    expr,
                    size,
                    rot_scale as i32,
                    fixed_ref,
                    advice_ref,
                    instance_ref,
                    theta,
                    target,
                )
            };

            f(
                &pk.vk.cs.lookups[*i].input_expressions[..],
                &mut input[0..unusable_rows_start],
            );
            f(
                &pk.vk.cs.lookups[*i].table_expressions[..],
                &mut table[0..unusable_rows_start],
            );
            handle_lookup_pair(input, table, unusable_rows_start);
        });
    end_timer!(timer);

    let mut lookup_permuted_commitments = vec![C::identity(); pk.vk.cs.lookups.len() * 2];

    let timer = start_timer!(|| format!(
        "single lookup copy {} {}",
        single_unit_lookups.len(),
        single_comp_lookups.len()
    ));
    let single_unit_buffers = single_unit_lookups
        .iter_mut()
        .map(|x| {
            // FIXME: handle error
            let input_buf = device.alloc_device_buffer_from_slice(&x.1 .0[..])?;
            let table_buf = device.alloc_device_buffer_from_slice(&x.1 .1[..])?;
            Ok((x.0, input_buf, table_buf))
        })
        .collect::<DeviceResult<Vec<_>>>()?;

    let single_comp_buffers = single_comp_lookups
        .iter_mut()
        .map(|x| {
            // FIXME: handle error
            let input_buf = device.alloc_device_buffer_from_slice(&x.1 .0[..])?;
            let table_buf = device.alloc_device_buffer_from_slice(&x.1 .1[..])?;
            Ok((x.0, input_buf, table_buf))
        })
        .collect::<DeviceResult<Vec<_>>>()?;
    end_timer!(timer);

    let timer = start_timer!(|| format!(
        "single lookup msm {} {}",
        single_unit_lookups.len(),
        single_comp_lookups.len()
    ));

    for (i, permuted_input_buf, permuted_table_buf) in single_unit_buffers.iter() {
        lookup_permuted_commitments[i * 2] = msm(
            &device,
            &msm_result_buf,
            &g_lagrange_buf,
            &permuted_input_buf,
            size,
        )?;
        lookup_permuted_commitments[i * 2 + 1] = msm(
            &device,
            &msm_result_buf,
            &g_lagrange_buf,
            &permuted_table_buf,
            size,
        )?;
    }

    for (i, permuted_input_buf, permuted_table_buf) in single_comp_buffers.iter() {
        lookup_permuted_commitments[i * 2] = msm(
            &device,
            &msm_result_buf,
            &g_lagrange_buf,
            &permuted_input_buf,
            size,
        )?;
        lookup_permuted_commitments[i * 2 + 1] = msm(
            &device,
            &msm_result_buf,
            &g_lagrange_buf,
            &permuted_table_buf,
            size,
        )?;
    }

    end_timer!(timer);

    Ok(())
}

fn is_expression_pure_unit<F: FieldExt>(x: &Expression<F>) -> bool {
    x.is_constant().is_some()
        || x.is_pure_fixed().is_some()
        || x.is_pure_advice().is_some()
        || x.is_pure_instance().is_some()
}

fn lookup_classify<'a, 'b, C: CurveAffine, T>(
    pk: &'b ProvingKey<C>,
    lookups_buf: Vec<T>,
) -> [Vec<(usize, T)>; 3] {
    let mut single_unit_lookups = vec![];
    let mut single_comp_lookups = vec![];
    let mut tuple_lookups = vec![];

    pk.vk
        .cs
        .lookups
        .iter()
        .zip(lookups_buf.into_iter())
        .enumerate()
        .for_each(|(i, (lookup, buf))| {
            let is_single =
                lookup.input_expressions.len() == 1 && lookup.table_expressions.len() == 1;

            if is_single {
                let is_unit = is_expression_pure_unit(&lookup.input_expressions[0])
                    && is_expression_pure_unit(&lookup.table_expressions[0]);
                if is_unit {
                    single_unit_lookups.push((i, buf));
                } else {
                    single_comp_lookups.push((i, buf));
                }
            } else {
                tuple_lookups.push((i, buf))
            }
        });

    return [single_unit_lookups, single_comp_lookups, tuple_lookups];
}

fn handle_lookup_pair<F: FieldExt>(
    input: &mut Vec<F, HugePageAllocator>,
    table: &mut Vec<F, HugePageAllocator>,
    unusable_rows_start: usize,
) {
    let compare = |a: &_, b: &_| unsafe {
        let a: &[u64; 4] = std::mem::transmute(a);
        let b: &[u64; 4] = std::mem::transmute(b);
        a.cmp(b)
    };

    input[0..unusable_rows_start].sort_unstable_by(compare);
    table[0..unusable_rows_start].sort_unstable_by(compare);

    let mut permuted_table_state = Vec::new_in(HugePageAllocator);
    permuted_table_state.resize(input.len(), false);

    let mut permuted_table = Vec::new_in(HugePageAllocator);
    permuted_table.resize(input.len(), F::zero());

    input
        .iter()
        .zip(permuted_table_state.iter_mut())
        .zip(permuted_table.iter_mut())
        .enumerate()
        .for_each(|(row, ((input_value, table_state), table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != input[row - 1] {
                *table_state = true;
                *table_value = *input_value;
            }
        });

    let to_next_unique = |i: &mut usize| {
        while *i < unusable_rows_start && !permuted_table_state[*i] {
            *i += 1;
        }
    };

    let mut i_unique_input_idx = 0;
    let mut i_sorted_table_idx = 0;
    for i in 0..unusable_rows_start {
        to_next_unique(&mut i_unique_input_idx);
        while i_unique_input_idx < unusable_rows_start
            && permuted_table[i_unique_input_idx] == table[i_sorted_table_idx]
        {
            i_unique_input_idx += 1;
            i_sorted_table_idx += 1;
            to_next_unique(&mut i_unique_input_idx);
        }
        if !permuted_table_state[i] {
            permuted_table[i] = table[i_sorted_table_idx];
            i_sorted_table_idx += 1;
        }
    }

    *table = permuted_table;

    if true {
        for cell in &mut input[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
        for cell in &mut table[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate_expr<F: FieldExt>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[&[F]],
    advice: &[&[F]],
    instance: &[&[F]],
    res: &mut [F],
) {
    let isize = size as i32;

    let get_rotation_idx = |idx: usize, rot: i32, rot_scale: i32, isize: i32| -> usize {
        (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
    };

    for (idx, value) in res.iter_mut().enumerate() {
        *value = expression.evaluate(
            &|scalar| scalar,
            &|_| panic!("virtual selectors are removed during optimization"),
            &|_, column_index, rotation| {
                fixed[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
            },
            &|_, column_index, rotation| {
                advice[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
            },
            &|_, column_index, rotation| {
                instance[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
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
    }
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
    for (idx, value) in res.iter_mut().enumerate() {
        for expression in expressions {
            *value = *value * theta;
            *value += expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|_, column_index, rotation| {
                    fixed[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                },
                &|_, column_index, rotation| {
                    advice[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                },
                &|_, column_index, rotation| {
                    instance[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
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
        }
    }
}
