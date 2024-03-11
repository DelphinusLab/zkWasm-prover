#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

use std::collections::BTreeMap;
use std::iter;
use std::sync::Arc;
use std::thread;

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::pairing::group::ff::BatchInvert as _;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::Expression;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::IntoParallelRefMutIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::prelude::ParallelSliceMut as _;
use rayon::slice::ParallelSlice as _;

use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::intt_raw_async;
use crate::cuda::bn254::msm_single_buffer;
use crate::cuda::bn254::ntt_prepare;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::Device as _;
use crate::eval_h::evaluate_h_gates_and_vanishing_construct;
use crate::hugetlb::HugePageAllocator;
use crate::multiopen::gwc;
use crate::multiopen::lookup_open;
use crate::multiopen::permutation_product_open;
use crate::multiopen::shplonk;
use crate::multiopen::ProverQuery;

pub mod cuda;
pub mod device;

mod eval_h;
mod hugetlb;
mod multiopen;

const ADD_RANDOM: bool = true;

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
    pin_memory: bool,
) -> Vec<Vec<C::Scalar, HugePageAllocator>> {
    let rows = 1 << pk.get_vk().domain.k();
    let columns = pk.get_vk().cs.num_advice_columns;
    let zero = C::Scalar::zero();
    let advices = (0..columns)
        .into_par_iter()
        .map(|_| {
            let mut buf = Vec::new_in(HugePageAllocator);
            buf.resize(rows, zero);
            buf
        })
        .collect::<Vec<_>>();

    let device = CudaDevice::get_device(0).unwrap();
    if pin_memory {
        for x in advices.iter() {
            device.pin_memory(&x[..]).unwrap();
        }
        for x in pk.fixed_values.iter() {
            device.pin_memory(&x[..]).unwrap();
        }
        for x in pk.permutation.polys.iter() {
            device.pin_memory(&x[..]).unwrap();
        }
    }

    advices
}

pub fn unpin_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
    advices: &mut Vec<Vec<C::Scalar, HugePageAllocator>>,
) {
    let device = CudaDevice::get_device(0).unwrap();
    for x in advices.iter() {
        device.unpin_memory(&x[..]).unwrap();
    }
    for x in pk.fixed_values.iter() {
        device.unpin_memory(&x[..]).unwrap();
    }
    for x in pk.permutation.polys.iter() {
        device.unpin_memory(&x[..]).unwrap();
    }
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
) -> (Vec<F, HugePageAllocator>, Vec<F, HugePageAllocator>) {
    let compare = |a: &_, b: &_| unsafe {
        let a: &[u64; 4] = std::mem::transmute(a);
        let b: &[u64; 4] = std::mem::transmute(b);
        a.cmp(b)
    };

    let mut permuted_input = input.clone();
    let mut sorted_table = table.clone();

    permuted_input[0..unusable_rows_start].sort_unstable_by(compare);
    sorted_table[0..unusable_rows_start].sort_unstable_by(compare);

    let mut permuted_table_state = Vec::new_in(HugePageAllocator);
    permuted_table_state.resize(input.len(), false);

    let mut permuted_table = Vec::new_in(HugePageAllocator);
    permuted_table.resize(input.len(), F::zero());

    permuted_input
        .iter()
        .take(unusable_rows_start)
        .zip(permuted_table_state.iter_mut().take(unusable_rows_start))
        .zip(permuted_table.iter_mut().take(unusable_rows_start))
        .enumerate()
        .for_each(|(row, ((input_value, table_state), table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input[row - 1] {
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
            && permuted_table[i_unique_input_idx] == sorted_table[i_sorted_table_idx]
        {
            i_unique_input_idx += 1;
            i_sorted_table_idx += 1;
            to_next_unique(&mut i_unique_input_idx);
        }
        if !permuted_table_state[i] {
            permuted_table[i] = sorted_table[i_sorted_table_idx];
            i_sorted_table_idx += 1;
        }
    }

    if ADD_RANDOM {
        for cell in &mut permuted_input[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
        for cell in &mut permuted_table[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
    } else {
        for cell in &mut permuted_input[unusable_rows_start..] {
            *cell = F::zero();
        }
        for cell in &mut permuted_table[unusable_rows_start..] {
            *cell = F::zero();
        }
    }

    (permuted_input, permuted_table)
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
        for (i, expression) in expressions.iter().enumerate() {
            let v = expression.evaluate(
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

            if i > 0 {
                *value = *value * theta;
                *value += v;
            } else {
                *value = v;
            }
        }
    }
}

pub fn create_proof_from_advices<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
) -> Result<(), Error> {
    create_proof_from_advices_with_gwc(params, pk, instances, advices, transcript)
}

pub fn create_proof_from_advices_with_gwc<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
) -> Result<(), Error> {
    _create_proof_from_advices(params, pk, instances, advices, transcript, true)
}

pub fn create_proof_from_advices_with_shplonk<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
) -> Result<(), Error> {
    _create_proof_from_advices(params, pk, instances, advices, transcript, false)
}

fn _create_proof_from_advices<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    mut advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
    use_gwc: bool,
) -> Result<(), Error> {
    thread::scope(|s| {
        let k = pk.get_vk().domain.k() as usize;
        let size = 1 << pk.get_vk().domain.k();
        let meta = &pk.vk.cs;
        let unusable_rows_start = size - (meta.blinding_factors() + 1);
        let omega = pk.get_vk().domain.get_omega();

        let domain = &pk.vk.domain;

        pk.vk.hash_into(transcript).unwrap();

        let mut instances = Arc::new(
            instances
                .par_iter()
                .map(|x| {
                    let mut instance = Vec::new_in(HugePageAllocator);
                    instance.resize(size, C::Scalar::zero());
                    instance[0..x.len()].clone_from_slice(&x[..]);
                    instance
                })
                .collect::<Vec<_>>(),
        );

        let device = CudaDevice::get_device(0).unwrap();

        device.synchronize()?;
        device.print_memory_info()?;

        // add random value
        if ADD_RANDOM {
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

        let timer = start_timer!(|| "copy g_lagrange buffer");
        let g_lagrange_buf = device
            .alloc_device_buffer_from_slice(&params.g_lagrange[..])
            .unwrap();
        end_timer!(timer);

        // thread for part of lookups
        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instances = instances.clone();
        let lookup_handler = s.spawn(move || {
            let pk = sub_pk;
            let advices = sub_advices;
            let instances = sub_instances;
            let timer =
                start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
            let lookups = pk
                .vk
                .cs
                .lookups
                .par_iter()
                .map(|_| {
                    let mut permuted_input = Vec::new_in(HugePageAllocator);
                    permuted_input.resize(size, C::Scalar::zero());
                    let mut permuted_table = Vec::new_in(HugePageAllocator);
                    permuted_table.resize(size, C::Scalar::zero());
                    let mut z = Vec::new_in(HugePageAllocator);
                    z.resize(size, C::Scalar::zero());

                    (permuted_input, permuted_table, z)
                })
                .collect::<Vec<_>>();
            end_timer!(timer);

            let [single_unit_lookups, single_comp_lookups, tuple_lookups] =
                lookup_classify(&pk, lookups);

            //let timer = start_timer!(|| format!("permute lookup unit {}", single_unit_lookups.len()));
            let single_unit_lookups = single_unit_lookups
                .into_par_iter()
                .map(|(i, (mut input, mut table, z))| {
                    let f = |expr: &Expression<_>, target: &mut [_]| {
                        if let Some(v) = expr.is_constant() {
                            target.fill(v);
                        } else if let Some(idx) = expr.is_pure_fixed() {
                            target.clone_from_slice(&pk.fixed_values[idx].values[..]);
                        } else if let Some(idx) = expr.is_pure_instance() {
                            target.clone_from_slice(&instances[idx][..]);
                        } else if let Some(idx) = expr.is_pure_advice() {
                            target.clone_from_slice(&advices[idx][..]);
                        } else {
                            unreachable!()
                        }
                    };

                    f(&pk.vk.cs.lookups[i].input_expressions[0], &mut input[..]);
                    f(&pk.vk.cs.lookups[i].table_expressions[0], &mut table[..]);
                    let (permuted_input, permuted_table) =
                        handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
                    (i, (permuted_input, permuted_table, input, table, z))
                })
                .collect::<Vec<_>>();
            //end_timer!(timer);

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

            let timer =
                start_timer!(|| format!("permute lookup comp {}", single_comp_lookups.len()));
            let single_comp_lookups = single_comp_lookups
                .into_par_iter()
                .map(|(i, (mut input, mut table, z))| {
                    let f = |expr: &Expression<_>, target: &mut [_]| {
                        evaluate_expr(expr, size, 1, fixed_ref, advice_ref, instance_ref, target)
                    };

                    f(&pk.vk.cs.lookups[i].input_expressions[0], &mut input[..]);
                    f(&pk.vk.cs.lookups[i].table_expressions[0], &mut table[..]);
                    let (permuted_input, permuted_table) =
                        handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
                    (i, (permuted_input, permuted_table, input, table, z))
                })
                .collect::<Vec<_>>();
            end_timer!(timer);

            (single_unit_lookups, single_comp_lookups, tuple_lookups)
        });

        let mut s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let mut t_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

        // Advice MSM
        let timer = start_timer!(|| format!(
            "instances and advices msm {}",
            instances.len() + advices.len()
        ));
        let commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_lagrange_buf,
            [&s_buf, &t_buf],
            instances
                .iter()
                .chain(advices.iter())
                .map(|x| &x[..])
                .collect(),
            size,
        )?;
        for commitment in commitments.iter().take(instances.len()) {
            transcript.common_point(*commitment).unwrap();
        }
        for commitment in commitments.into_iter().skip(instances.len()) {
            transcript.write_point(commitment).unwrap();
        }
        end_timer!(timer);

        let theta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "wait single lookups");
        let (mut single_unit_lookups, mut single_comp_lookups, mut tuple_lookups) =
            lookup_handler.join().unwrap();
        end_timer!(timer);

        // After theta
        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instance = instances.clone();
        let tuple_lookup_handler = s.spawn(move || {
            let pk = sub_pk;
            let advices = sub_advices;
            let instances = sub_instance;
            let timer = start_timer!(|| format!("permute lookup tuple {}", tuple_lookups.len()));

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

            let mut buffers = vec![];
            for (i, (input, table, _)) in tuple_lookups.iter_mut() {
                buffers.push((&pk.vk.cs.lookups[*i].input_expressions[..], &mut input[..]));
                buffers.push((&pk.vk.cs.lookups[*i].table_expressions[..], &mut table[..]));
            }

            buffers.into_par_iter().for_each(|(expr, buffer)| {
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

            let tuple_lookups = tuple_lookups
                .into_par_iter()
                .map(|(i, (mut input, mut table, z))| {
                    let (permuted_input, permuted_table) =
                        handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
                    (i, (permuted_input, permuted_table, input, table, z))
                })
                .collect::<Vec<_>>();
            end_timer!(timer);

            tuple_lookups
        });

        let mut lookup_permuted_commitments = vec![C::identity(); pk.vk.cs.lookups.len() * 2];

        let timer = start_timer!(|| format!(
            "single lookup msm {} {}",
            single_unit_lookups.len(),
            single_comp_lookups.len()
        ));

        {
            let mut lookup_scalars = vec![];
            for (_, (permuted_input, permuted_table, _, _, _)) in single_unit_lookups.iter() {
                lookup_scalars.push(&permuted_input[..]);
                lookup_scalars.push(&permuted_table[..])
            }
            for (_, (permuted_input, permuted_table, _, _, _)) in single_comp_lookups.iter() {
                lookup_scalars.push(&permuted_input[..]);
                lookup_scalars.push(&permuted_table[..])
            }
            let commitments = crate::cuda::bn254::batch_msm::<C>(
                &g_lagrange_buf,
                [&s_buf, &t_buf],
                lookup_scalars,
                size,
            )?;
            let mut tidx = 0;
            for (i, _) in single_unit_lookups.iter() {
                lookup_permuted_commitments[i * 2] = commitments[tidx];
                lookup_permuted_commitments[i * 2 + 1] = commitments[tidx + 1];
                tidx += 2;
            }
            for (i, _) in single_comp_lookups.iter() {
                lookup_permuted_commitments[i * 2] = commitments[tidx];
                lookup_permuted_commitments[i * 2 + 1] = commitments[tidx + 1];
                tidx += 2;
            }
        }
        end_timer!(timer);

        let timer = start_timer!(|| "wait tuple lookup");
        let mut tuple_lookups = tuple_lookup_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| format!("tuple lookup msm {}", tuple_lookups.len()));
        {
            let mut lookup_scalars = vec![];
            for (_, (permuted_input, permuted_table, _, _, _)) in tuple_lookups.iter() {
                lookup_scalars.push(&permuted_input[..]);
                lookup_scalars.push(&permuted_table[..])
            }
            let commitments = crate::cuda::bn254::batch_msm::<C>(
                &g_lagrange_buf,
                [&s_buf, &t_buf],
                lookup_scalars,
                size,
            )?;
            let mut tidx = 0;
            for (i, _) in tuple_lookups.iter() {
                lookup_permuted_commitments[i * 2] = commitments[tidx];
                lookup_permuted_commitments[i * 2 + 1] = commitments[tidx + 1];
                tidx += 2;
            }
        }
        end_timer!(timer);

        for commitment in lookup_permuted_commitments.into_iter() {
            transcript.write_point(commitment).unwrap();
        }

        let beta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let gamma: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let mut lookups = vec![];
        lookups.append(&mut single_unit_lookups);
        lookups.append(&mut single_comp_lookups);
        lookups.append(&mut tuple_lookups);
        lookups.sort_by(|l, r| usize::cmp(&l.0, &r.0));

        let timer = start_timer!(|| "generate lookup z");
        {
            let mut last_stream = None;
            let mut last_z_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut last_input_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut last_table_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut last_permuted_input_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut last_permuted_table_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

            let mut z_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut input_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut table_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut permuted_input_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let mut permuted_table_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

            let beta_gamma_buf = device.alloc_device_buffer_from_slice(&[beta, gamma])?;
            for (_, (permuted_input, permuted_table, input, table, z)) in lookups.iter_mut() {
                unsafe {
                    use crate::cuda::bn254_c::eval_lookup_z;
                    use crate::device::cuda::to_result;
                    use crate::device::cuda::CudaBuffer;

                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;

                    for (d_buf, h_buf) in [
                        (&input_buf, input),
                        (&table_buf, table),
                        (&permuted_input_buf, permuted_input),
                        (&permuted_table_buf, permuted_table),
                    ] {
                        device.copy_from_host_to_device_async(d_buf, &h_buf[..], stream)?;
                    }

                    let err = eval_lookup_z(
                        z_buf.ptr(),
                        input_buf.ptr(),
                        table_buf.ptr(),
                        permuted_input_buf.ptr(),
                        permuted_table_buf.ptr(),
                        beta_gamma_buf.ptr(),
                        size as i32,
                        stream,
                    );

                    to_result((), err, "failed to run eval_lookup_z")?;

                    device
                        .copy_from_device_to_host_async(&mut z[..], &z_buf, stream)
                        .unwrap();

                    if let Some(last_stream) = last_stream {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }

                    last_stream = Some(stream);
                    std::mem::swap(&mut z_buf, &mut last_z_buf);
                    std::mem::swap(&mut input_buf, &mut last_input_buf);
                    std::mem::swap(&mut table_buf, &mut last_table_buf);
                    std::mem::swap(&mut permuted_input_buf, &mut last_permuted_input_buf);
                    std::mem::swap(&mut permuted_table_buf, &mut last_permuted_table_buf);
                }
            }

            unsafe {
                if let Some(last_stream) = last_stream {
                    cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                    cuda_runtime_sys::cudaStreamDestroy(last_stream);
                }
            }
        }
        let mut lookups = lookups
            .into_iter()
            .map(|(_, (permuted_input, permuted_table, input, table, z))| {
                (permuted_input, permuted_table, input, table, z)
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "prepare ntt");
        let (intt_omegas_buf, intt_pq_buf) =
            ntt_prepare(&device, pk.get_vk().domain.get_omega_inv(), k)?;
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])?;
        end_timer!(timer);

        let chunk_len = &pk.vk.cs.degree() - 2;

        let timer = start_timer!(|| format!(
            "product permutation {}",
            (&pk).vk.cs.permutation.columns.chunks(chunk_len).len()
        ));

        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instance = instances.clone();
        let permutation_products_handler = s.spawn(move || {
            let pk = sub_pk;
            let advices = sub_advices;
            let instances = sub_instance;

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let mut p_z = pk
                .vk
                .cs
                .permutation
                .columns
                .par_chunks(chunk_len)
                .zip((&pk).permutation.permutations.par_chunks(chunk_len))
                .enumerate()
                .map(|(i, (columns, permutations))| {
                    let mut delta_omega =
                        C::Scalar::DELTA.pow_vartime([i as u64 * chunk_len as u64]);

                    let mut modified_values = Vec::new_in(HugePageAllocator);
                    modified_values.resize(size, C::Scalar::one());

                    // Iterate over each column of the permutation
                    for (&column, permuted_column_values) in columns.iter().zip(permutations.iter())
                    {
                        let values = match column.column_type() {
                            Any::Advice => advice_ref,
                            Any::Fixed => fixed_ref,
                            Any::Instance => instance_ref,
                        };
                        let chunk_size = size >> 1;
                        modified_values
                            .par_chunks_mut(chunk_size)
                            .zip(permuted_column_values.par_chunks(chunk_size))
                            .zip(values[column.index()].par_chunks(chunk_size))
                            .for_each(|((res, p), v)| {
                                for i in 0..chunk_size {
                                    res[i] *= &(beta * p[i] + &gamma + v[i]);
                                }
                            });
                    }

                    // Invert to obtain the denominator for the permutation product polynomial
                    modified_values.iter_mut().batch_invert();

                    // Iterate over each column again, this time finishing the computation
                    // of the entire fraction by computing the numerators
                    for &column in columns.iter() {
                        let values = match column.column_type() {
                            Any::Advice => advice_ref,
                            Any::Fixed => fixed_ref,
                            Any::Instance => instance_ref,
                        };

                        let chunk_size = size >> 1;
                        modified_values
                            .par_chunks_mut(chunk_size)
                            .zip(values[column.index()].par_chunks(chunk_size))
                            .enumerate()
                            .for_each(|(idx, (res, v))| {
                                let mut delta_omega =
                                    delta_omega * omega.pow_vartime([(idx * chunk_size) as u64]);
                                for i in 0..chunk_size {
                                    res[i] *= &(delta_omega * &beta + &gamma + v[i]);
                                    delta_omega *= &omega;
                                }
                            });

                        delta_omega *= &C::Scalar::DELTA;
                    }

                    modified_values
                })
                .collect::<Vec<_>>();

            let mut tmp = C::Scalar::one();
            for z in p_z.iter_mut() {
                for i in 0..size {
                    std::mem::swap(&mut tmp, &mut z[i]);
                    tmp = tmp * z[i];
                }

                if ADD_RANDOM {
                    for v in z[unusable_rows_start + 1..].iter_mut() {
                        *v = C::Scalar::random(&mut OsRng);
                    }
                } else {
                    for v in z[unusable_rows_start + 1..].iter_mut() {
                        *v = C::Scalar::zero();
                    }
                }

                tmp = z[unusable_rows_start];
            }
            p_z
        });
        end_timer!(timer);

        let timer = start_timer!(|| "lookup z msm");
        let lookup_z_commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_lagrange_buf,
            [&s_buf, &t_buf],
            lookups.iter().map(|x| &x.4[..]).collect::<Vec<_>>(),
            size,
        )?;
        end_timer!(timer);

        let timer = start_timer!(|| "lookups intt");
        for (permuted_input, permuted_table, _, _, z) in lookups.iter_mut() {
            for values in [permuted_input, permuted_table, z] {
                device.copy_from_host_to_device(&s_buf, &values[..])?;
                intt_raw(
                    &device,
                    &mut s_buf,
                    &mut t_buf,
                    &intt_pq_buf,
                    &intt_omegas_buf,
                    &intt_divisor_buf,
                    k,
                )?;
                device.copy_from_device_to_host(&mut values[..], &s_buf)?;
            }
        }
        end_timer!(timer);

        let timer = start_timer!(|| "wait permutation_products");
        let mut permutation_products = permutation_products_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation z msm and intt");
        let permutation_commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_lagrange_buf,
            [&s_buf, &t_buf],
            permutation_products
                .iter()
                .map(|x| &x[..])
                .collect::<Vec<_>>(),
            size,
        )?;

        for values in permutation_products.iter_mut() {
            device.copy_from_host_to_device(&s_buf, &values[..])?;
            intt_raw(
                &device,
                &mut s_buf,
                &mut t_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            device.copy_from_device_to_host(&mut values[..], &s_buf)?;
        }
        end_timer!(timer);

        for commitment in permutation_commitments {
            transcript.write_point(commitment).unwrap();
        }

        for (_i, commitment) in lookup_z_commitments.into_iter().enumerate() {
            transcript.write_point(commitment).unwrap();
        }

        let g_buf = g_lagrange_buf;
        device.copy_from_host_to_device(&g_buf, &params.g[..])?;

        // TODO: move to sub-thread
        let timer = start_timer!(|| "random_poly");
        let random_poly = vanish_commit(&device, &s_buf, &g_buf, size, transcript).unwrap();
        end_timer!(timer);

        let y: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "h_poly");
        {
            let timer = start_timer!(|| "instances and advices intt");

            let mut last_stream = None;
            let mut last_tmp_buf = Some(device.alloc_device_buffer::<C::Scalar>(size)?);
            let mut last_ntt_buf = Some(device.alloc_device_buffer::<C::Scalar>(size)?);
            for col in unsafe {
                Arc::get_mut_unchecked(&mut instances)
                    .iter_mut()
                    .chain(Arc::get_mut_unchecked(&mut advices).iter_mut())
            } {
                unsafe {
                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;
                    device.copy_from_host_to_device_async(&s_buf, &col[..], stream)?;
                    intt_raw_async(
                        &device,
                        &mut s_buf,
                        &mut t_buf,
                        &intt_pq_buf,
                        &intt_omegas_buf,
                        &intt_divisor_buf,
                        k,
                        Some(stream),
                    )?;
                    device.copy_from_device_to_host_async(&mut col[..], &s_buf, stream)?;
                    if let Some(last_stream) = last_stream {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }
                    std::mem::swap(&mut s_buf, last_ntt_buf.as_mut().unwrap());
                    std::mem::swap(&mut t_buf, last_tmp_buf.as_mut().unwrap());
                    last_stream = Some(stream);
                }
            }
            if let Some(last_stream) = last_stream {
                unsafe {
                    cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                    cuda_runtime_sys::cudaStreamDestroy(last_stream);
                }
            }
            end_timer!(timer);
        }

        let fixed_ref = &pk.fixed_polys.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

        let (x, _xn, h_pieces) = evaluate_h_gates_and_vanishing_construct(
            &device,
            &pk,
            fixed_ref,
            advice_ref,
            instance_ref,
            &permutation_products
                .iter()
                .map(|x| &x[..])
                .collect::<Vec<_>>()[..],
            &lookups
                .iter()
                .map(|(v0, v1, v2, v3, v4)| (&v0[..], &v1[..], &v2[..], &v3[..], &v4[..]))
                .collect::<Vec<_>>()[..],
            y,
            beta,
            gamma,
            theta,
            intt_pq_buf,
            intt_omegas_buf,
            intt_divisor_buf,
            &g_buf,
            transcript,
        )?;
        end_timer!(timer);

        let mut inputs = vec![(&h_pieces[..], x)];

        meta.instance_queries.iter().for_each(|&(column, at)| {
            inputs.push((&instances[column.index()][..], domain.rotate_omega(x, at)))
        });

        meta.advice_queries.iter().for_each(|&(column, at)| {
            inputs.push((&advices[column.index()], domain.rotate_omega(x, at)))
        });

        meta.fixed_queries.iter().for_each(|&(column, at)| {
            inputs.push((&pk.fixed_polys[column.index()], domain.rotate_omega(x, at)))
        });

        inputs.push((&random_poly, x));

        for poly in pk.permutation.polys.iter() {
            inputs.push((&poly, x));
        }

        let permutation_products_len = permutation_products.len();
        for (i, poly) in permutation_products.iter().enumerate() {
            inputs.push((&poly, x));
            inputs.push((&poly, domain.rotate_omega(x, Rotation::next())));
            if i != permutation_products_len - 1 {
                inputs.push((
                    &poly,
                    domain.rotate_omega(x, Rotation(-((meta.blinding_factors() + 1) as i32))),
                ));
            }
        }

        let x_inv = domain.rotate_omega(x, Rotation::prev());
        let x_next = domain.rotate_omega(x, Rotation::next());

        for (permuted_input, permuted_table, _, _, z) in lookups.iter() {
            inputs.push((&z, x));
            inputs.push((&z, x_next));
            inputs.push((&permuted_input, x));
            inputs.push((&permuted_input, x_inv));
            inputs.push((&permuted_table, x));
        }

        let timer = start_timer!(|| format!("compute eval {}", inputs.len()));
        let mut collection = BTreeMap::new();
        let mut deg_buffer = BTreeMap::new();
        for (idx, (p, x)) in inputs.iter().enumerate() {
            collection
                .entry(p.as_ptr() as usize)
                .and_modify(|arr: &mut (_, Vec<_>)| arr.1.push((idx, x)))
                .or_insert((p, vec![(idx, x)]));
        }
        let mut evals = vec![C::Scalar::zero(); inputs.len()];

        let mut eval_map = BTreeMap::new();

        let poly_buf = &s_buf;
        let eval_buf = &t_buf;
        let tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        for (_, (p, arr)) in collection {
            device.copy_from_host_to_device(poly_buf, p)?;
            for (idx, x) in arr {
                if !deg_buffer.contains_key(&x) {
                    let mut buf = vec![*x];
                    for _ in 1..k {
                        buf.push(*buf.last().unwrap() * buf.last().unwrap());
                    }
                    deg_buffer.insert(x, device.alloc_device_buffer_from_slice(&buf)?);
                }

                unsafe {
                    use crate::device::cuda::CudaBuffer;
                    let err = crate::cuda::bn254_c::poly_eval(
                        poly_buf.ptr(),
                        eval_buf.ptr(),
                        tmp_buf.ptr(),
                        deg_buffer.get(&x).unwrap().ptr(),
                        size as i32,
                    );
                    crate::device::cuda::to_result((), err, "fail to run poly_eval")?;
                }
                device.copy_from_device_to_host(&mut evals[idx..idx + 1], eval_buf)?;

                eval_map.insert((p.as_ptr() as usize, *x), evals[idx]);
            }
        }

        for (_i, eval) in evals.into_iter().skip(1).enumerate() {
            transcript.write_scalar(eval).unwrap();
        }

        end_timer!(timer);

        let timer = start_timer!(|| "multi open");
        let instance_arr = [instances];
        let advices_arr = [advices];
        let permutation_products_arr = [permutation_products];
        let lookups_arr = [lookups];

        let queries = instance_arr
            .iter()
            .zip(advices_arr.iter())
            .zip(permutation_products_arr.iter())
            .zip(lookups_arr.iter())
            .flat_map(|(((instance, advice), permutation), lookups)| {
                iter::empty()
                    .chain(
                        (&pk)
                            .vk
                            .cs
                            .instance_queries
                            .iter()
                            .map(|&(column, at)| ProverQuery {
                                point: domain.rotate_omega(x, at),
                                rotation: at,
                                poly: &instance[column.index()][..],
                            }),
                    )
                    .chain(
                        (&pk)
                            .vk
                            .cs
                            .advice_queries
                            .iter()
                            .map(|&(column, at)| ProverQuery {
                                point: domain.rotate_omega(x, at),
                                rotation: at,
                                poly: &advice[column.index()],
                            }),
                    )
                    .chain(permutation_product_open(&pk, &permutation[..], x))
                    .chain(
                        lookups
                            .iter()
                            .flat_map(|lookup| {
                                lookup_open(&pk, (&lookup.0[..], &lookup.1[..], &lookup.4[..]), x)
                            })
                            .into_iter(),
                    )
            })
            .chain(
                (&pk)
                    .vk
                    .cs
                    .fixed_queries
                    .iter()
                    .map(|&(column, at)| ProverQuery {
                        point: domain.rotate_omega(x, at),
                        rotation: at,
                        poly: &pk.fixed_polys[column.index()],
                    }),
            )
            .chain((&pk).permutation.polys.iter().map(move |poly| ProverQuery {
                point: x,
                rotation: Rotation::cur(),
                poly: &poly.values[..],
            }))
            // We query the h(X) polynomial at x
            .chain(
                iter::empty()
                    .chain(Some(ProverQuery {
                        point: x,
                        rotation: Rotation::cur(),
                        poly: &h_pieces,
                    }))
                    .chain(Some(ProverQuery {
                        point: x,
                        rotation: Rotation::cur(),
                        poly: &random_poly,
                    })),
            );
        if use_gwc {
            gwc::multiopen(
                &device,
                &g_buf,
                queries,
                size,
                [&s_buf, &t_buf],
                eval_map,
                transcript,
            )?;
        } else {
            shplonk::multiopen(
                &pk,
                &device,
                &g_buf,
                queries,
                size,
                [&s_buf, &t_buf],
                eval_map,
                transcript,
            )?;
        }
        end_timer!(timer);

        Ok(())
    })
}

fn vanish_commit<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    device: &CudaDevice,
    s_buf: &CudaDeviceBufRaw,
    g_buf: &CudaDeviceBufRaw,
    size: usize,
    transcript: &mut T,
) -> Result<Vec<C::Scalar, HugePageAllocator>, Error> {
    use rand::thread_rng;
    use rand::RngCore;

    let random_nr = 32;
    let mut random_poly = Vec::new_in(HugePageAllocator);
    random_poly.resize(size, C::Scalar::zero());

    let random = vec![0; 32usize]
        .iter()
        .map(|_| C::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();

    random_poly.par_iter_mut().for_each(|coeff| {
        if ADD_RANDOM {
            let mut rng = thread_rng();
            *coeff = (C::Scalar::random(&mut rng) + random[rng.next_u64() as usize % random_nr])
                * (C::Scalar::random(&mut rng) + random[rng.next_u64() as usize % random_nr])
        }
    });

    // Commit
    device.copy_from_host_to_device(&s_buf, &random_poly[..])?;
    let commitment = msm_single_buffer(&g_buf, &s_buf, size)?;
    transcript.write_point(commitment).unwrap();

    Ok(random_poly)
}
