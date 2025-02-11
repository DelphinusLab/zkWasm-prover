#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

#[macro_use]
extern crate lazy_static;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;

use analyze::analyze_involved_advices;
use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use async_copy_queue::AsyncCopyQueue;
use buffer::prepare_lookup_buffer;
use cuda::msm::InttArgs;
use cuda::ntt::ntt_raw;
use cuda::ntt::ntt_sync;
use cuda_runtime_sys::cudaError;
use device::cuda::CudaStreamWrapper;
use eval_poly::batch_poly_eval;
use expr::evaluate_exprs;
use expr::evaluate_exprs_in_gpu_pure;
use expr::get_expr_degree;
use expr::load_fixed_buffer_into_gpu_async;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use log::debug;
use log::error;
use log::info;
use permutation::generate_permutation_product;
use rayon::prelude::*;
use shuffle::generate_shuffle_product;

use crate::buffer::*;
use crate::cuda::bn254::logup_sum_input_inv;
use crate::cuda::bn254_c::eval_logup_z;
use crate::cuda::bn254_c::eval_logup_z_pure;
use crate::cuda::ntt::generate_ntt_buffers;
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
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
pub mod hugetlb;

mod analyze;
mod async_copy_queue;
mod eval_h;
mod eval_poly;
mod expr;
mod multiopen;
mod permutation;
mod profile;
mod shuffle;

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

pub fn lookup_compute_multiplicity<F: FieldExt>(
    inputs_set: &Vec<Vec<Vec<F, HugePageAllocator>>>,
    table: &Vec<F, HugePageAllocator>,
    multiplicity: &mut Vec<F, HugePageAllocator>,
    unusable_rows_start: usize,
) {
    let timer = start_timer!(|| "lookup table sort");
    let mut sorted_table_with_indices = table
        .iter()
        .take(unusable_rows_start)
        .enumerate()
        .map(|(i, t)| (unsafe { std::mem::transmute::<_, &[u64; 4]>(t) }, i))
        .collect::<Vec<_>>();
    sorted_table_with_indices.par_sort_by_key(|(&t, _)| t);
    end_timer!(timer);

    let timer = start_timer!(|| "lookup construct m(X) values");
    let num_threads = rayon::current_num_threads();
    let chunk_size = (sorted_table_with_indices.len() + num_threads - 1) / num_threads;
    let res = inputs_set
        .iter()
        .flatten()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|inputs| {
            inputs[..unusable_rows_start]
                .par_chunks(chunk_size)
                .map(|chunks| {
                    let mut map_count: BTreeMap<usize, usize> = BTreeMap::new();
                    let mut map_cache: BTreeMap<_, usize> = BTreeMap::new();
                    for fi in chunks {
                        let index = if let Some(idx) = map_cache.get(fi) {
                            *idx
                        } else {
                            let index = sorted_table_with_indices
                                .binary_search_by_key(
                                    &unsafe { std::mem::transmute::<_, &[u64; 4]>(fi) },
                                    |&(t, _)| t,
                                )
                                .expect("binary_search_by_key");
                            let index = sorted_table_with_indices[index].1;
                            map_cache.insert(fi, index);
                            index
                        };
                        map_count
                            .entry(index)
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    }
                    map_count
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    res.iter().for_each(|trees| {
        trees.iter().for_each(|tree| {
            for (index, count) in tree {
                multiplicity[*index] += F::from(*count as u64);
            }
        })
    });
    end_timer!(timer);

    if false {
        let random = F::random(&mut OsRng);
        let res = (0..unusable_rows_start)
            .into_par_iter()
            .map(|r| {
                let inputs = inputs_set.iter().fold(F::zero(), |acc, set| {
                    // ∑ 1/(f_i(X)+beta)
                    let sum = set.iter().fold(F::zero(), |acc, input| {
                        acc + (input[r] + random).invert().unwrap()
                    });
                    acc + sum
                });
                // ∑ 1/(φ_i(X)) - m(X) / τ(X)))
                inputs - ((table[r] + random).invert().unwrap() * &multiplicity[r])
            })
            .collect::<Vec<_>>();
        let last_z = res.iter().fold(F::zero(), |acc, v| acc + v);
        assert_eq!(last_z, F::zero());
    }

    fill_random(&mut multiplicity[unusable_rows_start..]);
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

fn print_prove_info<C: CurveAffine>(pk: &ProvingKey<C>) -> Result<(), Error> {
    info!(
        "permutation groups is {}",
        (&pk)
            .vk
            .cs
            .permutation
            .columns
            .chunks(&pk.vk.cs.degree() - 2)
            .len()
    );

    let _lookup_sets = pk
        .vk
        .cs
        .lookups
        .iter()
        .map(|lookup| lookup.input_expressions_sets.len())
        .collect::<Vec<_>>();
    info!(
        "lookups len={}, sets={:?}",
        pk.vk.cs.lookups.len(),
        _lookup_sets
    );

    info!(
        "product shuffles total={}, group={}",
        (&pk).vk.cs.shuffles.0.len(),
        pk.vk.cs.shuffles.group(pk.vk.cs.degree()).len()
    );

    info!("k is {}", pk.get_vk().domain.k());

    print_pinned_cache_info();

    let device = CudaDevice::get_device(0).unwrap();
    device.synchronize()?;
    device.print_memory_info()?;

    Ok(())
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
        error!("Multi-GPU detected, please set CUDA_VISIBLE_DEVICES to use one GPU");
        assert!(false);
    }

    let gpu_reserve_chuncks = std::env::var("ZKWASM_PROVER_GPU_RESERVE_CHUNCKS")
        .ok()
        .and_then(|s| usize::from_str_radix(&s, 10).ok())
        .unwrap_or(144);

    thread::scope(|s| {
        {
            let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
            allocator.reset(
                (1 << 22) * core::mem::size_of::<C::Scalar>(),
                gpu_reserve_chuncks,
            );
        }

        let timer = start_timer!(|| "proof prepare");

        let k = pk.get_vk().domain.k() as usize;
        let size = 1 << pk.get_vk().domain.k();
        let advices_len = advices.len();
        let instances_len = instances.len();
        let meta = &pk.vk.cs;
        let unusable_rows_start = size - (meta.blinding_factors() + 1);
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

        let named = &pk.vk.cs.named_advices;
        unsafe { Arc::get_mut_unchecked(&mut advices) }
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, advice)| {
                if named.iter().find(|n| n.1 as usize == i).is_none() {
                    fill_random(&mut advice[unusable_rows_start..]);
                }
            });

        let device = CudaDevice::get_device(0).unwrap();

        let (intt_omegas_buf, intt_pq_buf) =
            generate_ntt_buffers(&device, pk.get_vk().domain.get_omega_inv(), k)?;
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])?;

        let (
            uninvolved_units,
            uninvolved_units_after_single_lookup,
            uninvolved_units_after_tuple_lookup,
            uninvolved_units_after_permutation,
            uninvolved_units_after_shuffle,
        ) = analyze_involved_advices(pk);
        end_timer!(timer);

        print_prove_info(pk)?;

        let mut fixed_buffer_handler = s.spawn(move || prepare_fixed_buffer(pk));

        let timer = start_timer!(|| "copy g_lagrange buffer");
        let g_buf = device
            .alloc_device_buffer_from_slice(&params.g[..])
            .unwrap();
        let g_lagrange_buf = device
            .alloc_device_buffer_from_slice(&params.g_lagrange[..])
            .unwrap();
        end_timer!(timer);

        let buffer_handler = s.spawn(move || {
            let timer = start_timer!(|| "prepare buffers");
            let lookups = prepare_lookup_buffer(pk).unwrap();
            let permutations = prepare_permutation_buffers(pk).unwrap();
            let shuffles = prepare_shuffle_buffers(pk).unwrap();
            let random_poly = generate_random_poly::<C::Scalar>(size);
            end_timer!(timer);

            (lookups, permutations, shuffles, random_poly)
        });

        // GPU Task: MSM & INTT of advices & instances, load fixed values into GPU
        let mut fixed_buffers = std::collections::HashMap::new();
        let mut fixed_values = vec![];
        let mut fixed_polys = vec![];
        let mut advice_and_instance_device_buffers = {
            let timer = start_timer!(|| format!(
                "instances and advices msm {}",
                instances.len() + advices.len()
            ));

            let copy_sw = CudaStreamWrapper::new();

            let (commitments, advice_and_instance_device_buffers) =
                crate::cuda::msm::batch_msm_and_intt_ext::<C>(
                    &device,
                    &g_lagrange_buf,
                    unsafe {
                        Arc::get_mut_unchecked(&mut advices)
                            .iter_mut()
                            .chain(Arc::get_mut_unchecked(&mut instances).iter_mut())
                            .map(|x| &mut x[..])
                            .collect()
                    },
                    InttArgs {
                        pq_buf: &intt_pq_buf,
                        omegas_buf: &intt_omegas_buf,
                        divisor_buf: &intt_divisor_buf,
                        len_log: k,
                        selector: &|x| uninvolved_units.contains(&x),
                    },
                    &|x| !uninvolved_units.contains(&x),
                    &mut || {
                        let mut s = s.spawn(move || [vec![], vec![]]);
                        core::mem::swap(&mut s, &mut fixed_buffer_handler);

                        let timer = start_timer!(|| "wait fixed buffers");
                        let mut buffers = s.join().unwrap();
                        end_timer!(timer);
                        core::mem::swap(&mut buffers[0], &mut fixed_values);
                        core::mem::swap(&mut buffers[1], &mut fixed_polys);

                        let fixed_ref =
                            &fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
                        for (_, lookup) in pk.vk.cs.lookups.iter().enumerate() {
                            load_fixed_buffer_into_gpu_async(
                                &device,
                                &mut fixed_buffers,
                                &lookup.table_expressions[..],
                                fixed_ref,
                                &copy_sw,
                            )
                            .unwrap();

                            for (_, input_sets) in lookup.input_expressions_sets.iter().enumerate()
                            {
                                for (_, input) in input_sets.0.iter().enumerate() {
                                    load_fixed_buffer_into_gpu_async(
                                        &device,
                                        &mut fixed_buffers,
                                        &input[..],
                                        fixed_ref,
                                        &copy_sw,
                                    )
                                    .unwrap();
                                }
                            }
                        }
                    },
                    size,
                    true,
                )?;
            for commitment in commitments.iter().skip(advices.len()) {
                transcript.common_point(*commitment).unwrap();
            }
            for commitment in commitments.into_iter().take(advices.len()) {
                transcript.write_point(commitment).unwrap();
            }
            copy_sw.sync();
            end_timer!(timer);
            advice_and_instance_device_buffers
        };

        info!(
            "advice_and_instance_device_buffers size is {}",
            advice_and_instance_device_buffers.len()
        );

        let theta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "wait prepare buffers");
        let (mut lookups, permutations, shuffles, random_poly) = buffer_handler.join().unwrap();
        let random_buffer = device.alloc_device_buffer_from_slice(&random_poly[..])?;
        end_timer!(timer);

        info!("fixed_buffers size is {}", fixed_buffers.len());

        let timer = start_timer!(|| "calc lookup Mi(x)");
        let mut lookup_device_buffers = HashMap::new();
        let fixed_ref = &fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        {
            let index_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let sorted_index_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let start_offset_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
            let candidate_sort_temp_storage_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

            let tmp_buf0 = device.alloc_device_buffer::<C::Scalar>(size)?;
            let tmp_buf1 = device.alloc_device_buffer::<C::Scalar>(size)?;

            let (calc_sw, calc_stream) = CudaStreamWrapper::new_with_inner();

            for (i, lookup) in pk.vk.cs.lookups.iter().enumerate() {
                let m_device_buffer = device.alloc_device_buffer::<C::Scalar>(size)?;

                // let timer = start_timer!(|| "eval table");
                let table_device_buffer = evaluate_exprs_in_gpu_pure(
                    &device,
                    &advice_and_instance_device_buffers,
                    &mut fixed_buffers,
                    &lookup.table_expressions[..],
                    fixed_ref,
                    theta,
                    advices_len,
                    size,
                )?;
                // end_timer!(timer);

                // let timer = start_timer!(|| "prepare_table_lookup");
                unsafe {
                    let res = crate::cuda::bn254_c::prepare_table_lookup(
                        table_device_buffer.ptr(),
                        tmp_buf0.ptr(),
                        tmp_buf1.ptr(),
                        index_buf.ptr(),
                        sorted_index_buf.ptr(),
                        start_offset_buf.ptr(),
                        candidate_sort_temp_storage_buf.ptr(),
                        (std::mem::size_of::<C::Scalar>() * size) as u32,
                        k as u32,
                        unusable_rows_start as u32,
                        calc_stream,
                    );

                    assert!(res == cudaError::cudaSuccess);
                }
                calc_sw.sync();
                // end_timer!(timer);

                for (_, input_sets) in lookup.input_expressions_sets.iter().enumerate() {
                    for (_, input) in input_sets.0.iter().enumerate() {
                        // calc about 20us for k22 in RTX4090

                        // let timer = start_timer!(|| "eval input");
                        let input_device_buffer = evaluate_exprs_in_gpu_pure(
                            &device,
                            &advice_and_instance_device_buffers,
                            &mut fixed_buffers,
                            &input[..],
                            fixed_ref,
                            theta,
                            advices_len,
                            size,
                        )?;
                        // end_timer!(timer);

                        // calc about 1ms for k22 in RTX4090

                        // let timer = start_timer!(|| "calc_m");
                        unsafe {
                            let res = crate::cuda::bn254_c::calc_m(
                                m_device_buffer.ptr(),
                                table_device_buffer.ptr(),
                                input_device_buffer.ptr(),
                                sorted_index_buf.ptr(),
                                start_offset_buf.ptr(),
                                tmp_buf0.ptr(),
                                tmp_buf1.ptr(),
                                candidate_sort_temp_storage_buf.ptr(),
                                (std::mem::size_of::<C::Scalar>() * size) as u32,
                                k as u32,
                                unusable_rows_start as u32,
                                calc_stream,
                            );

                            assert!(res == cudaError::cudaSuccess);
                        }
                        calc_sw.sync();
                        // end_timer!(timer);
                    }
                }

                lookup_device_buffers.insert((i, 1), table_device_buffer);
                lookup_device_buffers.insert((i, 2), m_device_buffer);
            }
        };
        end_timer!(timer);

        drop(fixed_values);

        let permutation_polys_handler = s.spawn(|| prepare_permutation_poly_buffer(pk));

        debug!(
            "lookup_device_buffers size is {}",
            lookup_device_buffers.len()
        );

        // MSM on Mi(x)
        let timer = start_timer!(|| "lookup Mi(x) msm");
        let lookup_multiplicity_commitments = crate::cuda::msm::batch_msm_ext::<C, _>(
            &device,
            &g_lagrange_buf,
            (0..pk.vk.cs.lookups.len())
                .map(|i| lookup_device_buffers.get(&(i, 2)).unwrap())
                .collect(),
            &mut || {},
            size,
            true,
        )?;
        end_timer!(timer);

        for commitment in lookup_multiplicity_commitments.into_iter() {
            transcript.write_point(commitment).unwrap();
        }

        let beta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let gamma: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        // CPU Task: Generate Permutation product and Shuffle product
        let waker = Arc::new((Mutex::new(false), Condvar::new()));
        let waiter = Arc::clone(&waker);
        let permutation_products_handler = {
            let sub_advices = advices.clone();
            let sub_instance = instances.clone();
            let permutation_products_handler = s.spawn(move || {
                let z = generate_permutation_product(
                    pk,
                    sub_instance,
                    sub_advices,
                    permutations,
                    beta,
                    gamma,
                    size,
                    unusable_rows_start,
                );

                let (lock, cvar) = &*waker;
                let mut started = lock.lock().unwrap();
                *started = true;
                cvar.notify_one();

                z
            });
            permutation_products_handler
        };

        let shuffle_products_handler = {
            let sub_advices = advices.clone();
            let sub_instance = instances.clone();
            let shuffle_products_handler = s.spawn(move || {
                let (lock, cvar) = &*waiter;
                let mut started = lock.lock().unwrap();
                while !*started {
                    started = cvar.wait(started).unwrap();
                }

                generate_shuffle_product(
                    pk,
                    sub_instance,
                    sub_advices,
                    shuffles,
                    beta,
                    theta,
                    size,
                    unusable_rows_start,
                )
            });
            shuffle_products_handler
        };

        let mut copy_queue = AsyncCopyQueue::new(28);

        // GPU Task: Generate Lookup product
        let mut z_buffers = vec![];
        let timer = start_timer!(|| "generate lookup z");
        {
            let beta_buf = device.alloc_device_buffer_from_slice(&[beta])?;

            let last_z_buf = device.alloc_device_buffer::<C::Scalar>(pk.vk.cs.lookups.len())?;
            let tmp_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;
            let mut z_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;
            let mut t_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;
            let mut sum_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;

            let (calc_sw, calc_stream) = CudaStreamWrapper::new_with_inner();

            for ((lookup_i, lookup), (input, table, multiplicity, z_set)) in
                pk.vk.cs.lookups.iter().enumerate().zip(lookups.iter_mut())
            {
                let mut multiplicity_buf = lookup_device_buffers.remove(&(lookup_i, 2)).unwrap();
                let table_buf = lookup_device_buffers.remove(&(lookup_i, 1)).unwrap();

                //let timer = start_timer!(|| "copy table");
                if get_expr_degree(&lookup.table_expressions) == 1 {
                    copy_queue.push_with_copy_buffer(&device, &table_buf, &mut table[..], size)?;
                }
                //end_timer!(timer);

                for (set_i, input_sets) in lookup.input_expressions_sets.iter().enumerate() {
                    for (set_inner_i, input_expr) in input_sets.0.iter().enumerate() {
                        //let timer = start_timer!(|| "eval input");
                        let input_buf = evaluate_exprs_in_gpu_pure(
                            &device,
                            &advice_and_instance_device_buffers,
                            &mut fixed_buffers,
                            &input_expr[..],
                            &[],
                            theta,
                            advices_len,
                            size,
                        )?;
                        //end_timer!(timer);

                        if get_expr_degree(input_expr) == 1 {
                            //let timer = start_timer!(|| "copy input");
                            copy_queue.push_with_copy_buffer(
                                &device,
                                &input_buf,
                                &mut input[set_i][set_inner_i][..],
                                size,
                            )?;
                            //end_timer!(timer);
                        }

                        //let timer = start_timer!(|| "logup_sum_input_inv");
                        logup_sum_input_inv(
                            &device,
                            &sum_buf,
                            &input_buf,
                            &tmp_buf,
                            &beta_buf,
                            set_inner_i,
                            size,
                            Some(calc_stream),
                        )?;

                        calc_sw.sync();
                        //end_timer!(timer);
                    }

                    let intt_tasks = if set_i == 0 {
                        //let timer = start_timer!(|| "eval_logup_z");
                        unsafe {
                            let err = eval_logup_z(
                                z_buf.ptr(),
                                sum_buf.ptr(),
                                table_buf.ptr(),
                                multiplicity_buf.ptr(),
                                beta_buf.ptr(),
                                last_z_buf
                                    .ptr()
                                    .add(lookup_i * core::mem::size_of::<C::Scalar>()),
                                unusable_rows_start as i32,
                                size as i32,
                                calc_stream,
                            );

                            to_result((), err, "failed to run eval_logup_z")?;
                        }
                        calc_sw.sync();
                        //end_timer!(timer);

                        vec![
                            (&mut multiplicity_buf, &mut multiplicity[..], true),
                            (&mut z_buf, &mut z_set[set_i][..], false),
                        ]
                    } else {
                        //let timer = start_timer!(|| "eval_logup_z_pure");
                        unsafe {
                            core::mem::swap(&mut z_buf, &mut sum_buf);
                            let err = eval_logup_z_pure(
                                z_buf.ptr(),
                                sum_buf.ptr(),
                                table_buf.ptr(),
                                last_z_buf
                                    .ptr()
                                    .add(lookup_i * core::mem::size_of::<C::Scalar>()),
                                unusable_rows_start as i32,
                                size as i32,
                                calc_stream,
                            );

                            to_result((), err, "failed to run eval_logup_z")?;
                        }
                        //end_timer!(timer);

                        vec![(&mut z_buf, &mut z_set[set_i][..], false)]
                    };

                    for (s_buf, col, copy) in intt_tasks {
                        ntt_raw(
                            &device,
                            s_buf,
                            &mut t_buf,
                            &intt_pq_buf,
                            &intt_omegas_buf,
                            k,
                            Some(&intt_divisor_buf),
                            Some(&calc_sw),
                        )?;
                        calc_sw.sync();
                        if copy {
                            copy_queue
                                .push_with_new_buffer::<C::Scalar>(&device, s_buf, col, size)?;
                        }
                    }

                    z_buffers.push((z_buf, lookup_i, set_i));
                    z_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;
                }
            }

            if false {
                let (ntt_omegas_buf, ntt_pq_buf) =
                    generate_ntt_buffers(&device, pk.get_vk().domain.get_omega(), k)?;
                // While in Lagrange basis, check that grand sum is correctly constructed
                /*
                     φ_i(X) = f_i(X) + α
                     τ(X) = t(X) + α
                     LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                     RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))

                     extend inputs:
                     φ_i(X) = f_i(X) + α
                     LHS = Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                     RHS = Π(φ_i(X)) * (∑ 1/(φ_i(X)))

                */
                for (inputs_sets, table, m_coeff, zs_coeff) in lookups.iter() {
                    let mut s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
                    let mut tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

                    let mut m_lagrange = table.clone();
                    device.copy_from_host_to_device(&mut s_buf, m_coeff)?;
                    ntt_sync(
                        &device,
                        &mut s_buf,
                        &mut tmp_buf,
                        &ntt_pq_buf,
                        &ntt_omegas_buf,
                        &mut m_lagrange[..],
                        k,
                    )?;

                    let mut zs_lagrange = (0..zs_coeff.len())
                        .map(|_| table.clone())
                        .collect::<Vec<_>>();
                    for (z, z_lagrange) in zs_coeff.iter().zip(zs_lagrange.iter_mut()) {
                        device.copy_from_host_to_device(&mut s_buf, z)?;
                        ntt_sync(
                            &device,
                            &mut s_buf,
                            &mut tmp_buf,
                            &ntt_pq_buf,
                            &ntt_omegas_buf,
                            &mut z_lagrange[..],
                            k,
                        )?;
                    }

                    let u = unusable_rows_start;
                    let z_first = zs_lagrange.first().unwrap();
                    let z_last = zs_lagrange.last().unwrap();
                    // l_0(X) * (z_first(X)) = 0
                    assert_eq!(z_first[0], C::Scalar::zero());
                    let mut input_set_sums = inputs_sets
                        .par_iter()
                        .map(|input_set| {
                            (0..u)
                                .into_par_iter()
                                .map(|i| {
                                    input_set
                                        .iter()
                                        .map(|input| (input[i] + beta).invert().unwrap())
                                        .fold(C::Scalar::zero(), |acc, e| acc + e)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();

                    for ((input, table), m) in input_set_sums[0]
                        .iter_mut()
                        .zip(table.iter())
                        .zip(m_lagrange.iter())
                    {
                        *input = *input - *m * &(*table + beta).invert().unwrap();
                    }

                    for (zi, (input_set_sum, z_lag)) in
                        input_set_sums.iter().zip(zs_lagrange.iter()).enumerate()
                    {
                        for (i, input_sum) in input_set_sum.iter().enumerate() {
                            if z_lag[i + 1] != *input_sum + z_lag[i] {
                                debug!("bug at {} {}", zi, i);
                                assert!(false);
                            }
                        }
                    }

                    zs_lagrange
                        .iter()
                        .skip(1)
                        .zip(zs_lagrange.iter())
                        .for_each(|(z, z_pre)| assert_eq!(z[0], z_pre[u]));

                    // l_last(X) * (z_last(X)) = 0
                    assert_eq!(z_last[u], C::Scalar::zero());
                }
            }
        }
        end_timer!(timer);

        drop(fixed_buffers);

        let (lookup_z_commitments, random_commitment) = {
            let (sw, stream) = CudaStreamWrapper::new_with_inner();

            let mut s_bufs: Vec<_> = (0..advices_len)
                .into_iter()
                .filter(|i| {
                    uninvolved_units_after_single_lookup.contains(i)
                        || uninvolved_units_after_tuple_lookup.contains(i)
                })
                .map(|i| advice_and_instance_device_buffers.remove(&i).unwrap())
                .collect();

            let timer =
                start_timer!(|| format!("INTT on lookup involved advices {}", s_bufs.len()));
            let mut t_buf = device.alloc_device_buffer_non_zeroed::<C::Scalar>(size)?;

            for s_buf in s_bufs.iter_mut() {
                ntt_raw(
                    &device,
                    s_buf,
                    &mut t_buf,
                    &intt_pq_buf,
                    &intt_omegas_buf,
                    k,
                    Some(&intt_divisor_buf),
                    Some(&sw),
                )
                .unwrap();
            }
            sw.sync();
            copy_queue.sync_all();
            end_timer!(timer);

            let timer = start_timer!(|| format!("lookup z msm {}", z_buffers.len() + 1));
            let mut buffers = unsafe {
                Arc::get_mut_unchecked(&mut advices)
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| {
                        uninvolved_units_after_single_lookup.contains(i)
                            || uninvolved_units_after_tuple_lookup.contains(i)
                    })
                    .map(|(_, x)| &mut x[..])
                    .collect::<Vec<_>>()
            };

            for (dev_buf, host_buf) in s_bufs.iter().zip(buffers.iter_mut()) {
                device
                    .copy_from_device_to_host_async(host_buf, dev_buf, stream)
                    .unwrap();
            }

            let mut lookup_z_and_random_commitments = crate::cuda::msm::batch_msm_ext::<C, _>(
                &device,
                &g_buf,
                z_buffers
                    .iter()
                    .map(|x| &x.0)
                    .chain(vec![random_buffer].iter())
                    .collect::<Vec<_>>(),
                &mut || {},
                size,
                false,
            )?;

            for (z, lookup_i, set_i) in z_buffers {
                copy_queue.push(&device, z, &mut lookups[lookup_i].3[set_i])?;
            }
            drop(sw);
            drop(buffers);
            drop(s_bufs);
            end_timer!(timer);

            let random_commitment = lookup_z_and_random_commitments.pop().unwrap();
            (lookup_z_and_random_commitments, random_commitment)
        };

        let timer = start_timer!(|| "wait permutation_products");
        let mut permutation_products = permutation_products_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation z msm and intt & advice intt");
        let sw = CudaStreamWrapper::new();
        let mut s_bufs: Vec<_> = (0..advices_len)
            .into_iter()
            .filter(|i| uninvolved_units_after_permutation.contains(i))
            .map(|i| advice_and_instance_device_buffers.remove(&i).unwrap())
            .collect();

        {
            let timer =
                start_timer!(|| format!("INTT on permutation involved advices {}", s_bufs.len()));
            let mut t_buf = device
                .alloc_device_buffer_non_zeroed::<C::Scalar>(size)
                .unwrap();
            for s_buf in s_bufs.iter_mut() {
                ntt_raw(
                    &device,
                    s_buf,
                    &mut t_buf,
                    &intt_pq_buf,
                    &intt_omegas_buf,
                    k,
                    Some(&intt_divisor_buf),
                    Some(&sw),
                )?;
            }
            sw.sync();
            end_timer!(timer);
        }

        let mut buffers = unsafe {
            Arc::get_mut_unchecked(&mut advices)
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| uninvolved_units_after_permutation.contains(i))
                .map(|(_, x)| &mut x[..])
                .collect::<Vec<_>>()
        };

        let (permutation_commitments, _) = crate::cuda::msm::batch_msm_and_intt_ext::<C>(
            &device,
            &g_lagrange_buf,
            permutation_products
                .iter_mut()
                .map(|x| &mut x[..])
                .collect::<Vec<_>>(),
            InttArgs {
                pq_buf: &intt_pq_buf,
                omegas_buf: &intt_omegas_buf,
                divisor_buf: &intt_divisor_buf,
                len_log: k,
                selector: &|_| true,
            },
            &|_| false,
            &mut || {
                for (dev_buf, host_buf) in s_bufs.iter_mut().zip(buffers.iter_mut()) {
                    copy_queue
                        .push_with_new_buffer(&device, dev_buf, *host_buf, size)
                        .unwrap();
                }
            },
            size,
            false,
        )?;

        drop(sw);
        drop(s_bufs);
        drop(buffers);
        end_timer!(timer);

        let timer = start_timer!(|| "wait shuffle_products");
        let mut shuffle_products = shuffle_products_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffle z msm and intt");
        {
            let mut buffers = unsafe {
                Arc::get_mut_unchecked(&mut advices)
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| uninvolved_units_after_shuffle.contains(i))
                    .map(|(_, x)| &mut x[..])
                    .chain(
                        Arc::get_mut_unchecked(&mut instances)
                            .iter_mut()
                            .map(|x| &mut x[..]),
                    )
                    .collect::<Vec<_>>()
            };

            let mut s_bufs: Vec<_> = (0..advices_len + instances_len)
                .into_iter()
                .filter_map(|i| advice_and_instance_device_buffers.remove(&i))
                .collect();

            let (sw, stream) = CudaStreamWrapper::new_with_inner();

            let (shuffle_commitments, _) = crate::cuda::msm::batch_msm_and_intt_ext::<C>(
                &device,
                &g_lagrange_buf,
                shuffle_products
                    .iter_mut()
                    .map(|x| &mut x[..])
                    .collect::<Vec<_>>(),
                InttArgs {
                    pq_buf: &intt_pq_buf,
                    omegas_buf: &intt_omegas_buf,
                    divisor_buf: &intt_divisor_buf,
                    len_log: k,
                    selector: &|_| true,
                },
                &|_| false,
                &mut || {
                    let mut t_buf = device.alloc_device_buffer::<C::Scalar>(size).unwrap();
                    for s_buf in s_bufs.iter_mut() {
                        ntt_raw(
                            &device,
                            s_buf,
                            &mut t_buf,
                            &intt_pq_buf,
                            &intt_omegas_buf,
                            k,
                            Some(&intt_divisor_buf),
                            None,
                        )
                        .unwrap();
                    }
                    for (dev_buf, host_buf) in s_bufs.iter().zip(buffers.iter_mut()) {
                        device
                            .copy_from_device_to_host_async(host_buf, dev_buf, stream)
                            .unwrap();
                    }
                },
                size,
                false,
            )?;
            drop(g_lagrange_buf);
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
            sw.sync();
        }

        let timer = start_timer!(|| "wait copy queue");
        copy_queue.sync_all();
        end_timer!(timer);

        let y: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "h_poly");
        let fixed_ref = &fixed_polys.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

        let permutation_polys = permutation_polys_handler.join().unwrap();

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
            &permutation_polys.iter().map(|x| &x[..]).collect::<Vec<_>>()[..],
            &mut lookups
                .iter_mut()
                .map(|(v0, v1, v2, v3)| (v0, v1, v2, v3))
                .collect::<Vec<_>>(),
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

        let mut inputs = vec![(&h_pieces[..], 1, x)];

        meta.instance_queries.iter().for_each(|&(column, at)| {
            inputs.push((
                &instances[column.index()][..],
                1,
                domain.rotate_omega(x, at),
            ))
        });

        meta.advice_queries.iter().for_each(|&(column, at)| {
            inputs.push((&advices[column.index()], 1, domain.rotate_omega(x, at)))
        });

        meta.fixed_queries.iter().for_each(|&(column, at)| {
            inputs.push((
                &pk.fixed_polys[column.index()],
                0,
                domain.rotate_omega(x, at),
            ))
        });

        inputs.push((&random_poly, 1, x));

        for poly in permutation_polys.iter() {
            inputs.push((&poly, 0, x));
        }

        let permutation_products_len = permutation_products.len();
        for (i, poly) in permutation_products.iter().enumerate() {
            inputs.push((&poly, 1, x));
            inputs.push((&poly, 1, domain.rotate_omega(x, Rotation::next())));
            if i != permutation_products_len - 1 {
                inputs.push((
                    &poly,
                    1,
                    domain.rotate_omega(x, Rotation(-((meta.blinding_factors() + 1) as i32))),
                ));
            }
        }

        let x_next = domain.rotate_omega(x, Rotation::next());
        let x_last = domain.rotate_omega(x, Rotation(-((meta.blinding_factors() + 1) as i32)));

        for (_, _, multiplicity, zs) in lookups.iter() {
            inputs.push((&multiplicity, 1, x));
            for (i, z) in zs.iter().enumerate() {
                inputs.push((z, 1, x));
                inputs.push((z, 1, x_next));
                if i != zs.len() - 1 {
                    inputs.push((z, 1, x_last));
                }
            }
        }
        for z in shuffle_products.iter() {
            inputs.push((&z, 1, x));
            inputs.push((&z, 1, x_next));
        }

        let timer = start_timer!(|| "eval poly");
        let cache_count = 160 >> (k.max(22) - 22);
        let (poly_buf_cache, eval_map, evals) = batch_poly_eval(&device, inputs, k, cache_count)?;

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

        let queries =
            instance_arr
                .iter()
                .zip(advices_arr.iter())
                .zip(permutation_products_arr.iter())
                .zip(lookups_arr.iter())
                .zip(shuffles_arr.iter())
                .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
                    iter::empty()
                        .chain((&pk).vk.cs.instance_queries.iter().map(|&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(x, at),
                                rotation: at,
                                poly: &instance[column.index()][..],
                            }
                        }))
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
                                    lookup_open(&pk, (&lookup.2[..], &lookup.3[..]), x)
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
                .chain(permutation_polys.iter().map(move |poly| ProverQuery {
                    point: x,
                    rotation: Rotation::cur(),
                    poly: &poly[..],
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
            gwc::multiopen(&device, &g_buf, queries, size, eval_map, transcript)?;
        } else {
            shplonk::multiopen(
                &pk,
                &device,
                &g_buf,
                queries,
                size,
                eval_map,
                poly_buf_cache,
                transcript,
            )?;
        }
        end_timer!(timer);

        Ok(())
    })
}

pub fn warm_up_buffers<C: CurveAffine>(pk: &ProvingKey<C>) {
    let timer = start_timer!(|| "warm_up_buffers");
    let gpu_reserve_chuncks = std::env::var("ZKWASM_PROVER_GPU_RESERVE_CHUNCKS")
        .ok()
        .and_then(|s| usize::from_str_radix(&s, 10).ok())
        .unwrap_or(144);

    {
        let mut allocator = CUDA_BUFFER_ALLOCATOR.lock().unwrap();
        allocator.reset(
            (1 << 22) * core::mem::size_of::<C::Scalar>(),
            gpu_reserve_chuncks,
        );
    }

    let size = 1 << pk.get_vk().domain.k();
    prepare_lookup_buffer(pk).unwrap();
    prepare_permutation_buffers(pk).unwrap();
    prepare_shuffle_buffers(pk).unwrap();
    generate_random_poly::<C::Scalar>(size);
    end_timer!(timer);
}
