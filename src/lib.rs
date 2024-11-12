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

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use cuda::bn254::batch_msm_v2;
use cuda::bn254::intt_raw_async;
use cuda::bn254::ntt;
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

use crate::cuda::bn254::batch_intt_raw;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::logup_sum_input_inv;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254_c::{eval_logup_z, eval_logup_z_pure};
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::Device as _;
use crate::eval_h::evaluate_h_gates_and_vanishing_construct;
use crate::hugetlb::HugePageAllocator;
use crate::hugetlb::UnpinnedHugePageAllocator;
use crate::multiopen::gwc;
use crate::multiopen::lookup_open;
use crate::multiopen::permutation_product_open;
use crate::multiopen::shplonk;
use crate::multiopen::shuffle_open;
use crate::multiopen::ProverQuery;

pub mod cuda;
pub mod device;

mod eval_h;
mod hugetlb;
mod multiopen;

const ADD_RANDOM: bool = true;

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
    _pin_memory: bool,
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
    if false {
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

// fn lookup_classify<'a, 'b, C: CurveAffine, T>(
//     pk: &'b ProvingKey<C>,
//     lookups_buf: Vec<T>,
// ) -> [Vec<(usize, T)>; 3] {
//     let mut single_unit_lookups = vec![];
//     let mut single_comp_lookups = vec![];
//     let mut tuple_lookups = vec![];
//
//     pk.vk
//         .cs
//         .lookups
//         .iter()
//         .zip(lookups_buf.into_iter())
//         .enumerate()
//         .for_each(|(i, (lookup, buf))| {
//             let is_single =
//                 lookup.input_expressions.len() == 1 && lookup.table_expressions.len() == 1;
//
//             if is_single {
//                 let is_unit = is_expression_pure_unit(&lookup.input_expressions[0])
//                     && is_expression_pure_unit(&lookup.table_expressions[0]);
//                 if is_unit {
//                     single_unit_lookups.push((i, buf));
//                 } else {
//                     single_comp_lookups.push((i, buf));
//                 }
//             } else {
//                 tuple_lookups.push((i, buf))
//             }
//         });
//
//     return [single_unit_lookups, single_comp_lookups, tuple_lookups];
// }

fn lookup_classify<'a, 'b, C: CurveAffine, T>(
    pk: &'b ProvingKey<C>,
    lookups_buf: Vec<T>,
) -> [Vec<(usize, T)>; 2] {
    let mut single_lookups = vec![];
    let mut tuple_lookups = vec![];

    pk.vk
        .cs
        .lookups
        .iter()
        .zip(lookups_buf.into_iter())
        .enumerate()
        .for_each(|(i, (lookup, buf))| {
            //any input expressions which belongs to same table has the same len with table
            let is_single = lookup.table_expressions.len() == 1;

            if is_single {
                single_lookups.push((i, buf))
                // if is_expression_pure_unit(&lookup.table_expressions[0]){
                //     single_unit_lookups.push((i, 0,0,0,&buf.1))
                // }else{
                //     single_comp_lookups.push((i, 0,0,0,&buf.1))
                // }
                // lookup.input_expressions_sets.iter().enumerate().for_each(|(j,set)|{
                //     set.0.iter().enumerate().for_each(|(k,input_expressions)|{
                //         assert_eq!(input_expressions.len(),1);
                //         if is_expression_pure_unit(&input_expressions[0]) {
                //             single_unit_lookups.push((i,1,j,k, &buf.0[j][k]));
                //         } else {
                //             single_comp_lookups.push((i,,1,j,k &buf.0[j][k]));
                //         }
                //     })
                // });

                // let is_unit = is_expression_pure_unit(&lookup.input_expressions[0])
                //     && is_expression_pure_unit(&lookup.table_expressions[0]);
                // if is_unit {
                //     single_unit_lookups.push((i, buf));
                // } else {
                //     single_comp_lookups.push((i, buf));
                // }
            } else {
                tuple_lookups.push((i, buf))
            }
        });

    return [single_lookups, tuple_lookups];
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

    let mut permuted_table_state = Vec::new_in(UnpinnedHugePageAllocator);
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

pub fn lookup_compute_multiplicity<F: FieldExt>(
    inputs_set: &Vec<Vec<Vec<F, HugePageAllocator>>>,
    table: &Vec<F, HugePageAllocator>,
    multiplicity: &mut Vec<F, HugePageAllocator>,
    unusable_rows_start: usize,
) {
    let timer = start_timer!(|| "lookup table sort");
    let mut sorted_table_with_indices = table
        .par_iter()
        .take(unusable_rows_start)
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect::<Vec<_>>();
    sorted_table_with_indices.par_sort_by_key(|(&t, _)| t);
    end_timer!(timer);

    let timer = start_timer!(|| "lookup construct m(X) values");
    use std::sync::atomic::{AtomicU64, Ordering};
    let m_values: Vec<AtomicU64> = (0..multiplicity.len()).map(|_| AtomicU64::new(0)).collect();
    for inputs in inputs_set.iter().flatten() {
        inputs.par_iter().take(unusable_rows_start).for_each(|fi| {
            let index = sorted_table_with_indices
                .binary_search_by_key(&fi, |&(t, _)| t)
                .expect("binary_search_by_key");
            let index = sorted_table_with_indices[index].1;
            m_values[index].fetch_add(1, Ordering::Relaxed);
        });
    }

    multiplicity
        .par_iter_mut()
        .zip(m_values.par_iter())
        .take(unusable_rows_start)
        .for_each(|(m, v)| *m = F::from(v.load(Ordering::Relaxed)));
    end_timer!(timer);

    if ADD_RANDOM {
        for cell in &mut multiplicity[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
    } else {
        for cell in &mut multiplicity[unusable_rows_start..] {
            *cell = F::zero();
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

//
// pub fn prepare_lookup_buffer<C: CurveAffine>(
//     pk: &ProvingKey<C>,
// ) -> Result<
//     Vec<(
//         Vec<C::Scalar, HugePageAllocator>,
//         Vec<C::Scalar, HugePageAllocator>,
//         Vec<C::Scalar, HugePageAllocator>,
//         Vec<C::Scalar, HugePageAllocator>,
//         Vec<C::Scalar, HugePageAllocator>,
//     )>,
//     Error,
// > {
//     let size = 1 << pk.get_vk().domain.k();
//     let timer = start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
//     let lookups = pk
//         .vk
//         .cs
//         .lookups
//         .par_iter()
//         .map(|_| {
//             let mut input = Vec::new_in(HugePageAllocator);
//             input.resize(size, C::Scalar::zero());
//             let mut table = Vec::new_in(HugePageAllocator);
//             table.resize(size, C::Scalar::zero());
//             let mut permuted_input = Vec::new_in(HugePageAllocator);
//             permuted_input.resize(size, C::Scalar::zero());
//             let mut permuted_table = Vec::new_in(HugePageAllocator);
//             permuted_table.resize(size, C::Scalar::zero());
//             let mut z = Vec::new_in(HugePageAllocator);
//             z.resize(size, C::Scalar::zero());
//
//             if false {
//                 let device = CudaDevice::get_device(0).unwrap();
//                 device.pin_memory(&permuted_input[..]).unwrap();
//                 device.pin_memory(&permuted_table[..]).unwrap();
//                 device.pin_memory(&z[..]).unwrap();
//             }
//
//             (input, table, permuted_input, permuted_table, z)
//         })
//         .collect::<Vec<_>>();
//     end_timer!(timer);
//     Ok(lookups)
// }

pub fn prepare_lookup_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Result<
    Vec<(
        Vec<Vec<Vec<C::Scalar, HugePageAllocator>>>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<Vec<C::Scalar, HugePageAllocator>>,
    )>,
    Error,
> {
    let size = 1 << pk.get_vk().domain.k();
    let timer = start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
    let lookups = pk
        .vk
        .cs
        .lookups
        .par_iter()
        .map(|argument| {
            let mut table = Vec::new_in(HugePageAllocator);
            table.resize(size, C::Scalar::zero());
            let mut multiplicity = Vec::new_in(HugePageAllocator);
            multiplicity.resize(size, C::Scalar::zero());

            let inputs_sets = argument
                .input_expressions_sets
                .iter()
                .map(|set| {
                    set.0
                        .iter()
                        .map(|_| {
                            let mut input = Vec::new_in(HugePageAllocator);
                            input.resize(size, C::Scalar::zero());
                            input
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let z_set: Vec<_> = (0..argument.input_expressions_sets.len())
                .map(|_| {
                    let mut z = Vec::new_in(HugePageAllocator);
                    z.resize(size, C::Scalar::zero());
                    z
                })
                .collect();

            // if false {
            //     let device = CudaDevice::get_device(0).unwrap();
            //     device.pin_memory(&permuted_input[..]).unwrap();
            //     device.pin_memory(&permuted_table[..]).unwrap();
            //     device.pin_memory(&z[..]).unwrap();
            // }

            (inputs_sets, table, multiplicity, z_set)
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(lookups)
}

pub fn prepare_permutation_buffers<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Result<Vec<Vec<C::Scalar, HugePageAllocator>>, Error> {
    let size = 1 << pk.get_vk().domain.k();
    let chunk_len = &pk.vk.cs.degree() - 2;
    let timer = start_timer!(|| format!(
        "prepare permutation buffer, count {}",
        pk.vk.cs.permutation.columns.par_chunks(chunk_len).len()
    ));
    let buffers = pk
        .vk
        .cs
        .permutation
        .columns
        .par_chunks(chunk_len)
        .map(|_| {
            let mut z = Vec::new_in(HugePageAllocator);
            z.resize(size, C::Scalar::one());

            if false {
                let device = CudaDevice::get_device(0).unwrap();
                device.pin_memory(&z[..]).unwrap();
            }

            z
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(buffers)
}

pub fn prepare_shuffle_buffers<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Result<Vec<Vec<C::Scalar, HugePageAllocator>>, Error> {
    let size = 1 << pk.get_vk().domain.k();
    let timer = start_timer!(|| format!(
        "prepare shuffle buffer, count {}",
        pk.vk.cs.shuffles.group(pk.vk.cs.degree()).len()
    ));
    let buffers = pk
        .vk
        .cs
        .shuffles
        .group(pk.vk.cs.degree())
        .iter()
        .map(|_| {
            let mut z = Vec::new_in(HugePageAllocator);
            z.resize(size, C::Scalar::one());

            if false {
                let device = CudaDevice::get_device(0).unwrap();
                device.pin_memory(&z[..]).unwrap();
            }

            z
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(buffers)
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

    println!("k is {}", pk.get_vk().domain.k());

    thread::scope(|s| {
        let k = pk.get_vk().domain.k() as usize;
        let size = 1 << pk.get_vk().domain.k();
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
        let g_buf = device
            .alloc_device_buffer_from_slice(&params.g[..])
            .unwrap();
        end_timer!(timer);

        let _lookup_sets = pk
            .vk
            .cs
            .lookups
            .iter()
            .map(|lookup| lookup.input_expressions_sets.len())
            .collect::<Vec<_>>();
        println!(
            "lookups len={}, sets={:?}",
            pk.vk.cs.lookups.len(),
            _lookup_sets
        );

        // thread for part of lookups
        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instances = instances.clone();
        let lookup_handler = s.spawn(move || {
            let timer = start_timer!(|| "prepare buffers");
            let lookups = prepare_lookup_buffer(pk).unwrap();
            let permutations = prepare_permutation_buffers(pk).unwrap();
            let shuffles = prepare_shuffle_buffers(pk).unwrap();
            end_timer!(timer);

            let pk = sub_pk;
            let advices = sub_advices;
            let instances = sub_instances;

            let [mut single_lookups, tuple_lookups] = lookup_classify(&pk, lookups);

            let mut single_unit_buff = vec![];
            let mut single_comp_buff = vec![];
            for (i, v) in single_lookups.iter_mut() {
                let lookup = &pk.vk.cs.lookups[*i];
                if is_expression_pure_unit(&lookup.table_expressions[0]) {
                    single_unit_buff.push((&lookup.table_expressions[0], &mut v.1[..]))
                } else {
                    single_comp_buff.push((&lookup.table_expressions[0], &mut v.1[..]))
                }

                for (set, inputs) in lookup.input_expressions_sets.iter().zip(v.0.iter_mut()) {
                    for (input_expressions, val) in set.0.iter().zip(inputs.iter_mut()) {
                        assert_eq!(input_expressions.len(), 1);
                        if is_expression_pure_unit(&input_expressions[0]) {
                            single_unit_buff.push((&input_expressions[0], &mut val[..]));
                        } else {
                            single_comp_buff.push((&input_expressions[0], &mut val[..]));
                        }
                    }
                }
            }
            // let timer = start_timer!(|| format!("single lookup  {}", single_lookups.as_mut_slice().len()));
            let f = |(expr, target): (&Expression<_>, &mut [_])| {
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
            single_unit_buff.into_par_iter().for_each(f);

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

            single_comp_buff.into_par_iter().for_each(
                |(expr, target): (&Expression<_>, &mut [_])| {
                    evaluate_expr(expr, size, 1, fixed_ref, advice_ref, instance_ref, target)
                },
            );

            single_lookups.par_iter_mut().for_each(|(_, arg)| {
                lookup_compute_multiplicity(&arg.0, &arg.1, &mut arg.2, unusable_rows_start)
            });
            // end_timer!(timer);

            (single_lookups, tuple_lookups, permutations, shuffles)
        });

        let s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let t_buf = device.alloc_device_buffer::<C::Scalar>(size)?;

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
        let (mut single_lookups, mut tuple_lookups, permutations, shuffles) =
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
            let timer = start_timer!(|| format!("lookup tuple {}", tuple_lookups.len()));

            let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
            let instance_ref = &instances.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];

            let mut buffers = vec![];
            for (i, (input_sets, table, _, _)) in tuple_lookups.iter_mut() {
                buffers.push((&pk.vk.cs.lookups[*i].table_expressions[..], &mut table[..]));
                for (set, inputs) in pk.vk.cs.lookups[*i]
                    .input_expressions_sets
                    .iter()
                    .zip(input_sets.iter_mut())
                {
                    for (input_expr, input) in set.0.iter().zip(inputs.iter_mut()) {
                        buffers.push((&input_expr[..], &mut input[..]));
                    }
                }
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

            tuple_lookups.par_iter_mut().for_each(|(_, arg)| {
                lookup_compute_multiplicity(&arg.0, &arg.1, &mut arg.2, unusable_rows_start)
            });
            end_timer!(timer);

            tuple_lookups
        });

        let mut lookup_multiplicity_commitments = vec![C::identity(); pk.vk.cs.lookups.len()];

        let timer = start_timer!(|| format!("single lookup msm {}", single_lookups.len(),));

        {
            let mut lookup_scalars = vec![];
            for (_, (_, _, multiplicity, _)) in single_lookups.iter() {
                lookup_scalars.push(&multiplicity[..]);
            }
            let commitments = crate::cuda::bn254::batch_msm::<C>(
                &g_lagrange_buf,
                [&s_buf, &t_buf],
                lookup_scalars,
                size,
            )?;
            single_lookups
                .iter()
                .zip(commitments.into_iter())
                .for_each(|((i, _), commitment)| lookup_multiplicity_commitments[*i] = commitment);
        }
        end_timer!(timer);

        let timer = start_timer!(|| "wait tuple lookup");
        let mut tuple_lookups = tuple_lookup_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| format!("tuple lookup msm {}", tuple_lookups.len()));
        {
            let mut lookup_scalars = vec![];
            for (_, (_, _, multiplicity, _)) in tuple_lookups.iter() {
                lookup_scalars.push(&multiplicity[..]);
            }
            let commitments = crate::cuda::bn254::batch_msm::<C>(
                &g_lagrange_buf,
                [&s_buf, &t_buf],
                lookup_scalars,
                size,
            )?;
            tuple_lookups
                .iter()
                .zip(commitments.into_iter())
                .for_each(|((i, _), commitment)| lookup_multiplicity_commitments[*i] = commitment);
        }
        end_timer!(timer);

        for commitment in lookup_multiplicity_commitments.into_iter() {
            transcript.write_point(commitment).unwrap();
        }

        let beta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let gamma: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let mut lookups = vec![];
        lookups.append(&mut single_lookups);
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

                        if ADD_RANDOM {
                            for v in z[unusable_rows_start + 1..].iter_mut() {
                                *v = C::Scalar::random(&mut OsRng);
                            }
                        } else {
                            for v in z[unusable_rows_start + 1..].iter_mut() {
                                *v = C::Scalar::zero();
                            }
                        }

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

            let sub_pk = pk.clone();
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
                                    let mut buffer = Vec::new_in(UnpinnedHugePageAllocator);
                                    buffer.resize(size, C::Scalar::zero());
                                    Some(buffer)
                                };

                                let shuffle_buffer = if elements.shuffle_expressions.len() == 1
                                    && is_expression_pure_unit(&elements.input_expressions[0])
                                {
                                    None
                                } else {
                                    let mut buffer = Vec::new_in(UnpinnedHugePageAllocator);
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

                    if ADD_RANDOM {
                        for v in z[unusable_rows_start + 1..].iter_mut() {
                            *v = C::Scalar::random(&mut OsRng);
                        }
                    } else {
                        for v in z[unusable_rows_start + 1..].iter_mut() {
                            *v = C::Scalar::zero();
                        }
                    }
                });
                p_z
            });
            end_timer!(timer);
            shuffle_products_handler
        };

        let timer = start_timer!(|| "prepare ntt");
        let (intt_omegas_buf, intt_pq_buf) =
            ntt_prepare(&device, pk.get_vk().domain.get_omega_inv(), k)?;
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])?;
        end_timer!(timer);
        let (ntt_omegas_buf, ntt_pq_buf) = ntt_prepare(&device, pk.get_vk().domain.get_omega(), k)?;

        let timer = start_timer!(|| "generate lookup z");
        {
            const MAX_CONCURRENCY: usize = 3;
            let mut streams = [None; MAX_CONCURRENCY];
            let mut buffers = [0; MAX_CONCURRENCY].map(|_| {
                Rc::new([0; 4].map(|_| (device.alloc_device_buffer::<C::Scalar>(size).unwrap())))
            });

            let beta_buf = device.alloc_device_buffer_from_slice(&[beta])?;
            let mut last_z_bufs = (0..lookups.len())
                .map(|_| device.alloc_device_buffer::<C::Scalar>(1).unwrap())
                .collect::<Vec<_>>();

            for ((i, (inputs_sets, table, multiplicity, z_set)), last_z_buf) in
                lookups.iter_mut().zip(last_z_bufs.iter_mut())
            {
                unsafe {
                    let idx = *i % MAX_CONCURRENCY;
                    let [z_buf, input_buf, table_buf, multiplicity_buf] =
                        Rc::get_mut(&mut buffers[idx]).unwrap();

                    if let Some(last_stream) = streams[idx] {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }

                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;

                    for (i, input) in inputs_sets[0].iter_mut().enumerate() {
                        device.copy_from_host_to_device_async(table_buf, input, stream)?;
                        // sum  1/(input_i+beta)
                        logup_sum_input_inv(
                            &device,
                            input_buf,
                            table_buf,
                            multiplicity_buf,
                            &beta_buf,
                            i,
                            size,
                            Some(stream),
                        )?;
                    }
                    for (d_buf, h_buf) in [
                        (&*table_buf, &mut table[..]),
                        (&*multiplicity_buf, &mut multiplicity[..]),
                    ] {
                        device.copy_from_host_to_device_async(d_buf, h_buf, stream)?;
                    }

                    let err = eval_logup_z(
                        z_buf.ptr(),
                        input_buf.ptr(),
                        table_buf.ptr(),
                        multiplicity_buf.ptr(),
                        beta_buf.ptr(),
                        last_z_buf.ptr(),
                        unusable_rows_start as i32,
                        size as i32,
                        stream,
                    );

                    to_result((), err, "failed to run eval_lookup_z")?;

                    for s_buf in [&mut *multiplicity_buf, &mut *z_buf] {
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
                        (&mut multiplicity[..], &multiplicity_buf),
                        (&mut z_set[0][..], &z_buf),
                    ] {
                        device.copy_from_device_to_host_async(col, s_buf, stream)?;
                    }

                    //extra inputs and zs
                    // for (inputs_set, z) in inputs_sets.iter().zip(z_set.iter_mut()).skip(1) {
                    //     for (i, input) in inputs_set.iter().enumerate() {
                    //         device.copy_from_host_to_device_async(input_buf, input, stream)?;
                    //         logup_sum_input_inv(
                    //             &device,
                    //             z_buf,
                    //             input_buf,
                    //             table_buf,
                    //             &beta_buf,
                    //             i,
                    //             size,
                    //             Some(stream),
                    //         )?;
                    //     }
                    //
                    //     let err = eval_logup_z_pure(
                    //         z_buf.ptr(),
                    //         input_buf.ptr(),
                    //         table_buf.ptr(),
                    //         last_z_buf.ptr(),
                    //         unusable_rows_start as i32,
                    //         size as i32,
                    //         stream,
                    //     );
                    //
                    //     to_result((), err, "failed to run eval_lookup_z")?;
                    //
                    //     intt_raw_async(
                    //         &device,
                    //         &mut *z_buf,
                    //         &mut *input_buf,
                    //         &intt_pq_buf,
                    //         &intt_omegas_buf,
                    //         &intt_divisor_buf,
                    //         k,
                    //         Some(stream),
                    //     )?;
                    //
                    //     device.copy_from_device_to_host_async(&mut z[..], &z_buf, stream)?;
                    // }

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

            let mut streams = [None; MAX_CONCURRENCY];
            // assemble the extra inputs set
            let mut buff = vec![];
            let mut map = BTreeMap::new();
            for (i, (inputs_sets, _, _, z_set)) in lookups.iter_mut() {
                let mut inner = vec![];
                for (inputs_set, z) in inputs_sets.iter().zip(z_set.iter_mut()).skip(1) {
                    inner.push((i, &inputs_set[..], &mut z[..]))
                }
                if !inner.is_empty() {
                    buff.push(inner);
                }
            }
            let mut buff = buff.into_iter().map(|v| v.into_iter()).collect::<Vec<_>>();
            let mut buff_interleave = Vec::new();
            loop {
                let mut pushed = false;
                for (i, v) in buff.iter_mut().enumerate() {
                    if let Some(value) = v.next() {
                        buff_interleave.push((i, value));
                        pushed = true;
                    }
                }
                if !pushed {
                    break;
                }
            }

            println!("lookup extra buff={}", buff_interleave.len());
            for (i, (lookup_index, inputs_set, z)) in buff_interleave.iter_mut().enumerate() {
                unsafe {
                    let idx = i % MAX_CONCURRENCY;
                    let [z_buf, input_buf, table_buf, _] = Rc::get_mut(&mut buffers[idx]).unwrap();

                    if let Some(last_stream) = streams[idx] {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }

                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;

                    for (i, input) in inputs_set.iter().enumerate() {
                        device.copy_from_host_to_device_async(input_buf, input, stream)?;
                        logup_sum_input_inv(
                            &device,
                            z_buf,
                            input_buf,
                            table_buf,
                            &beta_buf,
                            i,
                            size,
                            Some(stream),
                        )?;
                    }

                    let err = eval_logup_z_pure(
                        z_buf.ptr(),
                        input_buf.ptr(),
                        table_buf.ptr(),
                        last_z_buf[*lookup_index].ptr(),
                        unusable_rows_start as i32,
                        size as i32,
                        stream,
                    );

                    to_result((), err, "failed to run eval_lookup_z")?;

                    intt_raw_async(
                        &device,
                        &mut *z_buf,
                        &mut *input_buf,
                        &intt_pq_buf,
                        &intt_omegas_buf,
                        &intt_divisor_buf,
                        k,
                        Some(stream),
                    )?;

                    device.copy_from_device_to_host_async(&mut z[..], &z_buf, stream)?;

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

            if false {
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
                for (_, (inputs_sets, table, m_coeff, zs_coeff)) in lookups.iter() {
                    let [z_buf, input_buf, table_buf, multiplicity_buf] =
                        Rc::get_mut(&mut buffers[0]).unwrap();
                    let mut m_lagrange = table.clone();
                    device.copy_from_host_to_device(z_buf, m_coeff)?;
                    ntt(
                        &device,
                        z_buf,
                        input_buf,
                        &ntt_pq_buf,
                        &ntt_omegas_buf,
                        &mut m_lagrange[..],
                        k,
                    )?;

                    let mut zs_lagrange = (0..zs_coeff.len())
                        .map(|_| table.clone())
                        .collect::<Vec<_>>();
                    for (z, z_lagrange) in zs_coeff.iter().zip(zs_lagrange.iter_mut()) {
                        device.copy_from_host_to_device(z_buf, z)?;
                        ntt(
                            &device,
                            z_buf,
                            input_buf,
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
                    // l_last(X) * (z_last(X)) = 0
                    assert_eq!(z_last[u], C::Scalar::zero());
                    let mut input_set_sums = inputs_sets
                        .iter()
                        .map(|input_set| {
                            (0..u)
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

                    for (j, (input_set_sum, z_lag)) in
                        input_set_sums.iter().zip(zs_lagrange.iter()).enumerate()
                    {
                        for (i, input_sum) in input_set_sum.iter().enumerate() {
                            assert_eq!(z_lag[i + 1], *input_sum + z_lag[i]);
                        }
                    }

                    zs_lagrange
                        .iter()
                        .skip(1)
                        .zip(zs_lagrange.iter())
                        .for_each(|(z, z_pre)| assert_eq!(z[0], z_pre[u]))
                }
            }
        }

        let mut lookups = lookups.into_iter().map(|(_, b)| b).collect::<Vec<_>>();
        end_timer!(timer);

        let z_counts = lookups.iter().map(|x| x.3.len()).sum::<usize>();
        let timer = start_timer!(|| format!("lookup z msm {}", z_counts));
        let lookup_z_commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_buf,
            [&s_buf, &t_buf],
            lookups
                .iter()
                .flat_map(|x| x.3.iter().map(|z| &z[..]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            size,
        )?;
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
        let shuffle_commitments = crate::cuda::bn254::batch_msm::<C>(
            &g_lagrange_buf,
            [&s_buf, &t_buf],
            shuffle_products.iter().map(|x| &x[..]).collect::<Vec<_>>(),
            size,
        )?;

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
                .map(|(_, v1, v2, v3)| (v1, v2, v3))
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

        let x_next = domain.rotate_omega(x, Rotation::next());
        let x_last = domain.rotate_omega(x, Rotation(-((meta.blinding_factors() + 1) as i32)));

        for (_, _, multiplicity, zs) in lookups.iter() {
            inputs.push((&multiplicity, x));
            for (i, z) in zs.iter().enumerate() {
                inputs.push((z, x));
                inputs.push((z, x_next));
                if i != zs.len() - 1 {
                    inputs.push((z, x_last));
                }
            }
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

        let mut poly_buf_cache = BTreeMap::new();
        let extended_buffers_count = if k < 23 { 30 } else { 15 };
        let mut extended_buffers = vec![];
        let mut cache_buffers = vec![];
        let extended_k = pk.vk.domain.extended_k() as usize;
        for _ in 0..extended_buffers_count {
            let buf = device.alloc_device_buffer::<C::Scalar>(1 << extended_k)?;
            for i in 0..1 << (extended_k - k) {
                cache_buffers.push(ManuallyDrop::new(CudaDeviceBufRaw {
                    ptr: unsafe {
                        buf.ptr()
                            .offset(((i << k) * core::mem::size_of::<C::Scalar>()) as isize)
                    },
                    device: device.clone(),
                    size: core::mem::size_of::<C::Scalar>(),
                }));
            }
            extended_buffers.push(buf);
        }

        let mut evals = vec![C::Scalar::zero(); inputs.len()];

        let timer = start_timer!(|| format!("compute eval {}", collection.len()));
        let mut eval_map = BTreeMap::new();

        let mut streams = vec![];
        let mut bufs = vec![];
        let max = 6;
        for _ in 0..max {
            bufs.push((
                device.alloc_device_buffer::<C::Scalar>(size)?,
                device.alloc_device_buffer::<C::Scalar>(size)?,
                device.alloc_device_buffer::<C::Scalar>(size)?,
            ));
            unsafe {
                let mut stream = std::mem::zeroed();
                let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                assert_eq!(err, cuda_runtime_sys::cudaError::cudaSuccess);
                streams.push(stream);
            }
        }

        let mut collection = collection.into_iter().collect::<Vec<_>>();
        collection.sort_by(|a, b| a.1 .1.len().cmp(&b.1 .1.len()));

        let mut l = 0;
        let mut r = collection.len();
        let mut inc = false;
        let mut used_cache_idx = 0;
        while l < r {
            let i = if inc { l } else { r - 1 };
            if inc {
                l += 1;
            } else {
                r -= 1;
            }
            inc = !inc;
            let (p, arr) = &collection[i].1;
            let p = *p;
            unsafe {
                let stream = streams[i % max];
                let (poly_buf, eval_buf, tmp_buf) = &bufs[i % max];
                let poly_buf = if used_cache_idx < cache_buffers.len() {
                    let buf = &cache_buffers[used_cache_idx];
                    poly_buf_cache.insert((*p).as_ptr() as usize, buf);
                    used_cache_idx += 1;
                    buf
                } else {
                    poly_buf
                };
                device.copy_from_host_to_device_async(poly_buf, p, stream)?;
                for (idx, x) in arr {
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

        for stream in streams {
            unsafe {
                cuda_runtime_sys::cudaStreamSynchronize(stream);
                cuda_runtime_sys::cudaStreamDestroy(stream);
            }
        }

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
                poly_buf_cache,
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
    let commitment = batch_msm_v2(&g_buf, vec![&s_buf], size)?;
    transcript.write_point(commitment[0]).unwrap();

    Ok(random_poly)
}
