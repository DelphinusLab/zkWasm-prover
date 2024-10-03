#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

#[macro_use]
extern crate lazy_static;

use std::collections::BTreeMap;
use std::iter;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;

use analyze::lookup_classify;
use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use buffer::prepare_lookup_buffer;
use cuda::bn254::intt_raw_async;
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

use crate::buffer::*;
use crate::cuda::bn254::batch_intt_raw;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254_c::eval_lookup_z;
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::cuda::CUDA_BUFFER_ALLOCATOR;
use crate::device::Device as _;
use crate::eval_h::evaluate_h_gates_and_vanishing_construct;
use crate::expr::is_expression_pure_unit;
use crate::hugetlb::print_pinned_cache_info;
use crate::hugetlb::HugePageAllocator;
use crate::multiopen::gwc;
use crate::multiopen::lookup_open;
use crate::multiopen::permutation_product_open;
use crate::multiopen::shplonk;
use crate::multiopen::shuffle_open;
use crate::multiopen::ProverQuery;

pub mod buffer;
pub mod cuda;
pub mod device;

mod analyze;
mod eval_h;
mod expr;
mod hugetlb;
mod multiopen;

const ADD_RANDOM: bool = true;

pub(crate) fn fill_random<F: FieldExt>(data: &mut [F]) {
    for cell in data {
        if ADD_RANDOM {
            *cell = F::random(&mut OsRng);
        } else {
            *cell = F::zero();
        }
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

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
    _pin_memory: bool,
) -> Vec<Vec<C::Scalar, HugePageAllocator>> {
    crate::buffer::prepare_advice_buffer(pk)
}

fn handle_lookup_pair<F: FieldExt>(
    input: &mut Vec<F, HugePageAllocator>,
    table: &mut Vec<F, HugePageAllocator>,
    mut permuted_input: Vec<F, HugePageAllocator>,
    mut permuted_table: Vec<F, HugePageAllocator>,
    unusable_rows_start: usize,
) -> (Vec<F, HugePageAllocator>, Vec<F, HugePageAllocator>) {
    let compare = |a: &_, b: &_| unsafe {
        let a: &[u64; 4] = std::mem::transmute(a);
        let b: &[u64; 4] = std::mem::transmute(b);
        a.cmp(b)
    };

    permuted_input[..].clone_from_slice(&input[..]);
    let mut sorted_table = table.clone();

    permuted_input[0..unusable_rows_start].sort_unstable_by(compare);
    sorted_table[0..unusable_rows_start].sort_unstable_by(compare);

    let mut permuted_table_state = Vec::new_in(HugePageAllocator);
    permuted_table_state.resize(input.len(), false);

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

    fill_random(&mut permuted_input[unusable_rows_start..]);
    fill_random(&mut permuted_table[unusable_rows_start..]);

    (permuted_input, permuted_table)
}

/// Simple evaluation of an expression
pub(crate) fn evaluate_expr<F: FieldExt>(
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
    if pk.ev.gpu_gates_expr.len() != 1 {
        println!("Multi-GPU detected, please set CUDA_VISIBLE_DEVICES to use one GPU");
        assert!(false);
    }

    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
    println!("k is {}", k);
    print_pinned_cache_info();

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        let count = if k < 23 { 140 } else { 67 };
        allocator.reset((1 << k) * core::mem::size_of::<C::Scalar>(), count);
    }

    thread::scope(|s| {
        let meta = &pk.vk.cs;
        let unusable_rows_start = size - (meta.blinding_factors() + 1);
        let omega = pk.get_vk().domain.get_omega();

        let domain = &pk.vk.domain;

        pk.vk.hash_into(transcript).unwrap();

        assert!(instances.len() == pk.get_vk().cs.num_instance_columns);

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

        let named = &pk.vk.cs.named_advices;
        unsafe { Arc::get_mut_unchecked(&mut advices) }
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, advice)| {
                if named.iter().find(|n| n.1 as usize == i).is_none() {
                    fill_random(&mut advice[unusable_rows_start..]);
                }
            });

        let timer = start_timer!(|| "prepare ntt");
        let (intt_omegas_buf, intt_pq_buf) =
            ntt_prepare(&device, pk.get_vk().domain.get_omega_inv(), k)?;
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])?;
        end_timer!(timer);

        let timer = start_timer!(|| "copy g_lagrange buffer");
        let g_buf = device
            .alloc_device_buffer_from_slice(&params.g[..])
            .unwrap();
        let g_lagrange_buf = device
            .alloc_device_buffer_from_slice(&params.g_lagrange[..])
            .unwrap();
        end_timer!(timer);

        // thread for part of lookups
        let sub_pk = pk;
        let sub_advices = advices.clone();
        let sub_instances = instances.clone();
        let lookup_handler = s.spawn(move || {
            let timer = start_timer!(|| "prepare buffers");
            let lookups = prepare_lookup_buffer(pk).unwrap();
            let permutations = prepare_permutation_buffers(pk).unwrap();
            let shuffles = prepare_shuffle_buffers(pk).unwrap();
            let random_poly = generate_random_poly(size);
            end_timer!(timer);

            let pk = sub_pk;
            let advices = sub_advices;
            let instances = sub_instances;

            let [single_unit_lookups, single_comp_lookups, tuple_lookups] =
                lookup_classify(&pk, lookups);

            //let timer = start_timer!(|| format!("permute lookup unit {}", single_unit_lookups.len()));
            let single_unit_lookups = single_unit_lookups
                .into_par_iter()
                .map(
                    |(i, (mut input, mut table, permuted_input, permuted_table, z))| {
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
                        let (permuted_input, permuted_table) = handle_lookup_pair(
                            &mut input,
                            &mut table,
                            permuted_input,
                            permuted_table,
                            unusable_rows_start,
                        );
                        (i, (permuted_input, permuted_table, input, table, z))
                    },
                )
                .collect::<Vec<_>>();
            //end_timer!(timer);

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

            let timer =
                start_timer!(|| format!("permute lookup comp {}", single_comp_lookups.len()));
            let single_comp_lookups = single_comp_lookups
                .into_par_iter()
                .map(
                    |(i, (mut input, mut table, permuted_input, permuted_table, z))| {
                        let f = |expr: &Expression<_>, target: &mut [_]| {
                            evaluate_expr(
                                expr,
                                size,
                                1,
                                fixed_ref,
                                advice_ref,
                                instance_ref,
                                target,
                            )
                        };

                        f(&pk.vk.cs.lookups[i].input_expressions[0], &mut input[..]);
                        f(&pk.vk.cs.lookups[i].table_expressions[0], &mut table[..]);
                        let (permuted_input, permuted_table) = handle_lookup_pair(
                            &mut input,
                            &mut table,
                            permuted_input,
                            permuted_table,
                            unusable_rows_start,
                        );
                        (i, (permuted_input, permuted_table, input, table, z))
                    },
                )
                .collect::<Vec<_>>();
            end_timer!(timer);

            (
                single_unit_lookups,
                single_comp_lookups,
                tuple_lookups,
                permutations,
                shuffles,
                random_poly,
            )
        });

        let s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let t_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

        // Advice MSM
        let timer = start_timer!(|| format!(
            "instances and advices msm {}",
            instances.len() + advices.len()
        ));
        let commitments = crate::cuda::msm::batch_msm::<C>(
            &device,
            &g_lagrange_buf,
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
        let (
            mut single_unit_lookups,
            mut single_comp_lookups,
            mut tuple_lookups,
            permutations,
            shuffles,
            random_poly,
        ) = lookup_handler.join().unwrap();
        end_timer!(timer);

        // After theta
        let sub_pk = pk;
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
            for (i, (input, table, _, _, _)) in tuple_lookups.iter_mut() {
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
                .map(
                    |(i, (mut input, mut table, permuted_input, permuted_table, z))| {
                        let (permuted_input, permuted_table) = handle_lookup_pair(
                            &mut input,
                            &mut table,
                            permuted_input,
                            permuted_table,
                            unusable_rows_start,
                        );
                        (i, (permuted_input, permuted_table, input, table, z))
                    },
                )
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
            let commitments =
                crate::cuda::msm::batch_msm::<C>(&device, &g_lagrange_buf, lookup_scalars, size)?;
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
            let commitments =
                crate::cuda::msm::batch_msm::<C>(&device, &g_lagrange_buf, lookup_scalars, size)?;
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

        let chunk_len = &pk.vk.cs.degree() - 2;

        let waker = Arc::new((Mutex::new(false), Condvar::new()));
        let waiter = Arc::clone(&waker);
        let permutation_products_handler = {
            let timer = start_timer!(|| format!(
                "product permutation {}",
                (&pk).vk.cs.permutation.columns.chunks(chunk_len).len()
            ));

            let sub_pk = pk;
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
                    .zip(permutations)
                    .enumerate()
                    .map(|(i, ((columns, permutations), mut modified_values))| {
                        let mut delta_omega =
                            C::Scalar::DELTA.pow_vartime([i as u64 * chunk_len as u64]);

                        let chunk_size = size >> 2;
                        // Iterate over each column of the permutation
                        for (j, (&column, permuted_column_values)) in
                            columns.iter().zip(permutations.iter()).enumerate()
                        {
                            let values = match column.column_type() {
                                Any::Advice => advice_ref,
                                Any::Fixed => fixed_ref,
                                Any::Instance => instance_ref,
                            };
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
                            let values = match column.column_type() {
                                Any::Advice => advice_ref,
                                Any::Fixed => fixed_ref,
                                Any::Instance => instance_ref,
                            };

                            modified_values
                                .par_chunks_mut(chunk_size)
                                .zip(values[column.index()].par_chunks(chunk_size))
                                .enumerate()
                                .for_each(|(idx, (res, v))| {
                                    let mut delta_omega = delta_omega
                                        * omega.pow_vartime([(idx * chunk_size) as u64])
                                        * &beta;
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

                let (lock, cvar) = &*waker;
                let mut started = lock.lock().unwrap();
                *started = true;
                cvar.notify_one();

                let mut tails: Vec<_> = p_z
                    .par_iter_mut()
                    .map(|z| {
                        let mut tmp = C::Scalar::one();
                        for i in 0..size {
                            std::mem::swap(&mut tmp, &mut z[i]);
                            tmp = tmp * z[i];
                        }

                        fill_random(&mut z[unusable_rows_start + 1..]);
                        z[unusable_rows_start]
                    })
                    .collect();

                for i in 1..tails.len() {
                    tails[i] = tails[i] * tails[i - 1];
                }

                p_z.par_iter_mut().skip(1).enumerate().for_each(|(i, z)| {
                    for row in 0..=unusable_rows_start {
                        z[row] = z[row] * tails[i];
                    }
                });

                p_z
            });
            end_timer!(timer);
            permutation_products_handler
        };

        let shuffle_products_handler = {
            let shuffle_groups = pk.vk.cs.shuffles.group(pk.vk.cs.degree());
            let timer = start_timer!(|| format!(
                "product shuffles total={}, group={}",
                (&pk).vk.cs.shuffles.0.len(),
                shuffle_groups.len()
            ));

            let sub_pk = pk;
            let sub_advices = advices.clone();
            let sub_instance = instances.clone();
            let shuffle_products_handler = s.spawn(move || {
                let (lock, cvar) = &*waiter;
                let mut started = lock.lock().unwrap();
                while !*started {
                    started = cvar.wait(started).unwrap();
                }

                let pk = sub_pk;
                let advices = sub_advices;
                let instances = sub_instance;

                let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
                let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
                let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

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
                                    tuple_expres.push((
                                        &elements.input_expressions[..],
                                        buffer.0.as_mut().unwrap(),
                                    ));
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

                        let chunk_size = size >> 2;
                        group.iter().zip(beta_pows.iter()).enumerate().for_each(
                            |(j, (e, beta_pow_i))| {
                                modified_values
                                    .par_chunks_mut(chunk_size)
                                    .zip(e.1.par_chunks(chunk_size))
                                    .for_each(|(values, shuffles)| {
                                        for i in 0..chunk_size {
                                            if j == 0 {
                                                values[i] = shuffles[i] + beta_pow_i;
                                            } else {
                                                values[i] *= &(shuffles[i] + beta_pow_i);
                                            }
                                        }
                                    })
                            },
                        );
                        modified_values.par_chunks_mut(chunk_size).for_each(|x| {
                            x.iter_mut().batch_invert();
                        });
                        group
                            .iter()
                            .zip(beta_pows.iter())
                            .for_each(|(e, beta_pow_i)| {
                                modified_values
                                    .par_chunks_mut(chunk_size)
                                    .zip(e.0.par_chunks(chunk_size))
                                    .for_each(|(values, inputs)| {
                                        for i in 0..chunk_size {
                                            values[i] *= &(inputs[i] + beta_pow_i);
                                        }
                                    })
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
            });
            end_timer!(timer);
            shuffle_products_handler
        };

        let timer = start_timer!(|| "generate lookup z");
        {
            const MAX_CONCURRENCY: usize = 3;
            let mut streams = [None; MAX_CONCURRENCY];
            let mut buffers = [0; MAX_CONCURRENCY].map(|_| {
                Rc::new([0; 5].map(|_| (device.alloc_device_buffer::<C::Scalar>(size).unwrap())))
            });

            let beta_gamma_buf = device.alloc_device_buffer_from_slice(&[beta, gamma])?;
            for (i, (permuted_input, permuted_table, input, table, z)) in lookups.iter_mut() {
                unsafe {
                    let idx = *i % MAX_CONCURRENCY;
                    let [z_buf, input_buf, table_buf, permuted_input_buf, permuted_table_buf] =
                        Rc::get_mut(&mut buffers[idx]).unwrap();

                    if let Some(last_stream) = streams[idx] {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }

                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;

                    for (d_buf, h_buf) in [
                        (&*input_buf, &mut input[..]),
                        (&*table_buf, &mut table[..]),
                        (&*permuted_input_buf, &mut permuted_input[..]),
                        (&*permuted_table_buf, &mut permuted_table[..]),
                    ] {
                        device.copy_from_host_to_device_async(d_buf, h_buf, stream)?;
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

                    for s_buf in [
                        &mut *permuted_input_buf,
                        &mut *permuted_table_buf,
                        &mut *z_buf,
                    ] {
                        intt_raw_async(
                            &device,
                            s_buf,
                            &mut *input_buf,
                            &intt_pq_buf,
                            &intt_omegas_buf,
                            &intt_divisor_buf,
                            k,
                            Some(stream),
                        )?;
                    }

                    for (col, s_buf) in [
                        (&mut permuted_input[..], permuted_input_buf),
                        (&mut permuted_table[..], permuted_table_buf),
                        (&mut z[..], z_buf),
                    ] {
                        device.copy_from_device_to_host_async(col, &s_buf, stream)?;
                    }

                    streams[idx] = Some(stream);
                }
            }

            unsafe {
                for last_stream in streams {
                    if let Some(last_stream) = last_stream {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }
                }
            }
        }

        let mut lookups = lookups.into_iter().map(|(_, b)| b).collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| format!("lookup z msm {}", lookups.len()));
        let mut lookup_z_and_random_commitments = crate::cuda::msm::batch_msm::<C>(
            &device,
            &g_buf,
            lookups
                .iter()
                .map(|x| &x.4[..])
                .chain([&random_poly[..]])
                .collect::<Vec<_>>(),
            size,
        )?;
        let random_commitment = lookup_z_and_random_commitments.pop().unwrap();
        let lookup_z_commitments = lookup_z_and_random_commitments;
        end_timer!(timer);

        let timer = start_timer!(|| "wait permutation_products");
        let mut permutation_products = permutation_products_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation z msm and intt");
        let permutation_commitments = crate::cuda::msm::batch_msm::<C>(
            &device,
            &g_lagrange_buf,
            permutation_products
                .iter()
                .map(|x| &x[..])
                .collect::<Vec<_>>(),
            size,
        )?;

        batch_intt_raw(
            &device,
            permutation_products
                .iter_mut()
                .map(|x| &mut x[..])
                .collect::<Vec<_>>(),
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        end_timer!(timer);

        let timer = start_timer!(|| "wait shuffle_products");
        let mut shuffle_products = shuffle_products_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffle z msm and intt");
        let shuffle_commitments = crate::cuda::msm::batch_msm::<C>(
            &device,
            &g_lagrange_buf,
            shuffle_products.iter().map(|x| &x[..]).collect::<Vec<_>>(),
            size,
        )?;
        drop(g_lagrange_buf);

        batch_intt_raw(
            &device,
            shuffle_products
                .iter_mut()
                .map(|x| &mut x[..])
                .collect::<Vec<_>>(),
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        end_timer!(timer);

        for commitment in permutation_commitments {
            transcript.write_point(commitment).unwrap();
        }

        for (_i, commitment) in lookup_z_commitments.into_iter().enumerate() {
            transcript.write_point(commitment).unwrap();
        }

        for commitment in shuffle_commitments {
            transcript.write_point(commitment).unwrap();
        }

        transcript.write_point(random_commitment).unwrap();

        let y: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        drop(s_buf);
        drop(t_buf);

        let timer = start_timer!(|| "h_poly");
        {
            let timer = start_timer!(|| "instances and advices intt");

            let buffers = unsafe {
                Arc::get_mut_unchecked(&mut instances)
                    .iter_mut()
                    .map(|x| &mut x[..])
                    .chain(
                        Arc::get_mut_unchecked(&mut advices)
                            .iter_mut()
                            .map(|x| &mut x[..]),
                    )
                    .collect::<Vec<_>>()
            };
            batch_intt_raw(
                &device,
                buffers,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;

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
            &mut lookups
                .iter_mut()
                .map(|(v0, v1, v2, v3, v4)| {
                    (
                        &mut v0[..],
                        &mut v1[..],
                        &mut v2[..],
                        &mut v3[..],
                        &mut v4[..],
                    )
                })
                .collect::<Vec<_>>()[..],
            &shuffle_products.iter().map(|x| &x[..]).collect::<Vec<_>>()[..],
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
        for z in shuffle_products.iter() {
            inputs.push((&z, x));
            inputs.push((&z, x_next));
        }

        let mut collection = BTreeMap::new();
        let mut x_sets = vec![];
        for (idx, (p, x)) in inputs.iter().enumerate() {
            collection
                .entry(p.as_ptr() as usize)
                .and_modify(|arr: &mut (_, Vec<_>)| arr.1.push((idx, x)))
                .or_insert((p, vec![(idx, x)]));
            x_sets.push(x);
        }
        x_sets.sort_unstable();
        x_sets.dedup();
        let mut x_extend_sets = vec![];
        for x in x_sets.iter() {
            x_extend_sets.push(**x);
            for _ in 1..k {
                x_extend_sets.push(x_extend_sets.last().unwrap().square());
            }
        }

        let x_buf = device.alloc_device_buffer_from_slice(&x_extend_sets)?;
        let mut x_map = BTreeMap::new();
        for (i, x) in x_sets.into_iter().enumerate() {
            x_map.insert(
                x,
                ManuallyDrop::new(CudaDeviceBufRaw {
                    ptr: unsafe {
                        x_buf
                            .ptr()
                            .offset((i * k * core::mem::size_of::<C::Scalar>()) as isize)
                    },
                    device: device.clone(),
                    size: core::mem::size_of::<C::Scalar>(),
                }),
            );
        }

        let mut evals = vec![C::Scalar::zero(); inputs.len()];

        let timer = start_timer!(|| format!("compute eval {}", collection.len()));
        let mut eval_map = BTreeMap::new();

        let mut streams = vec![];
        let mut bufs = vec![];
        let max = 2;
        for _ in 0..max {
            bufs.push((
                device.alloc_device_buffer::<C::Scalar>(size)?,
                device.alloc_device_buffer::<C::Scalar>(size)?,
                device.alloc_device_buffer::<C::Scalar>(size)?,
            ));
            streams.push(CudaStreamWrapper::new_with_inner());
        }

        let mut collection = collection.into_iter().collect::<Vec<_>>();
        collection.sort_by(|a, b| a.1 .1.len().cmp(&b.1 .1.len()));

        let mut poly_buf_cache = BTreeMap::new();
        let cache_count = 80;

        for i in 0..collection.len() {
            let (p, arr) = &collection[i].1;
            let p = *p;
            let stream = streams[i % max].1;
            let (poly_buf, eval_buf, tmp_buf) = &bufs[i % max];
            let poly_buf = if poly_buf_cache.len() < cache_count {
                let key = p.as_ptr() as usize;
                let buf = device.alloc_device_buffer::<C::Scalar>(size)?;
                poly_buf_cache.insert(key, buf);
                poly_buf_cache.get(&key).unwrap()
            } else {
                poly_buf
            };

            device.copy_from_host_to_device_async(poly_buf, p, stream)?;
            for (idx, x) in arr {
                unsafe {
                    let err = crate::cuda::bn254_c::poly_eval(
                        poly_buf.ptr(),
                        eval_buf.ptr(),
                        tmp_buf.ptr(),
                        x_map.get(x).unwrap().ptr(),
                        size as i32,
                        stream,
                    );
                    crate::device::cuda::to_result((), err, "fail to run poly_eval")?;
                    device.copy_from_device_to_host_async(
                        &mut evals[*idx..*idx + 1],
                        eval_buf,
                        stream,
                    )?;
                    eval_map.insert(((*p).as_ptr() as usize, **x), *idx);
                }
            }
        }

        drop(streams);
        drop(bufs);

        let eval_map = eval_map
            .into_iter()
            .map(|(k, v)| (k, evals[v]))
            .collect::<BTreeMap<(usize, C::ScalarExt), C::ScalarExt>>();

        for (_i, eval) in evals.into_iter().skip(1).enumerate() {
            transcript.write_scalar(eval).unwrap();
        }

        end_timer!(timer);

        let timer = start_timer!(|| "multi open");
        let instance_arr = [instances];
        let advices_arr = [advices];
        let permutation_products_arr = [permutation_products];
        let lookups_arr = [lookups];
        let shuffles_arr = [shuffle_products];

        let queries = instance_arr
            .iter()
            .zip(advices_arr.iter())
            .zip(permutation_products_arr.iter())
            .zip(lookups_arr.iter())
            .zip(shuffles_arr.iter())
            .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
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
                    .chain(
                        shuffles
                            .iter()
                            .flat_map(|shuffle| shuffle_open(&pk, &shuffle[..], x))
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

        let s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let t_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
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
                poly_buf_cache,
                transcript,
            )?;
        }
        end_timer!(timer);

        Ok(())
    })
}
