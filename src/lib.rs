#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use ark_std::end_timer;
use ark_std::iterable::Iterable;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use cuda::bn254::buffer_copy_with_shift;
use cuda::bn254::extended_prepare;
use cuda::bn254::field_mul;
use cuda::bn254::field_sum;
use cuda::bn254::FieldOp;
use cuda_runtime_sys::cudaMemset;
use device::cuda::CudaBuffer;
use device::cuda::CudaDeviceBufRaw;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::pairing::group::ff::BatchInvert as _;
use halo2_proofs::plonk::evaluation_gpu::Bop;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Any;
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
use rayon::slice::ParallelSlice as _;
use std::collections::BTreeMap;

use crate::cuda::bn254::field_op_v2;
use crate::cuda::bn254::field_sub;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::msm;
use crate::cuda::bn254::msm_with_groups;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254::ntt_raw;
use crate::cuda::bn254::permutation_eval_h_l;
use crate::cuda::bn254::permutation_eval_h_p1;
use crate::cuda::bn254::permutation_eval_h_p2;
use crate::cuda::bn254::pick_from_buf;
use crate::cuda::bn254_c::lookup_eval_h;
use crate::device::cuda::to_result;
use crate::device::cuda::CudaDevice;
use crate::device::Device as _;
use crate::device::DeviceResult;

pub mod cuda;
pub mod device;
mod hugetlb;

const ADD_RANDOM: bool = false;

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
    input: &Vec<F, HugePageAllocator>,
    table: &Vec<F, HugePageAllocator>,
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
        .zip(permuted_table_state.iter_mut())
        .zip(permuted_table.iter_mut())
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

    if true {
        let mut last = None;
        for (a, b) in permuted_input
            .iter()
            .zip(permuted_table.iter())
            .take(unusable_rows_start)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    if ADD_RANDOM {
        for cell in &mut permuted_input[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
        for cell in &mut permuted_table[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
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
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
    let extended_k = pk.get_vk().domain.extended_k() as usize;
    let rot_scale = 1 << (extended_k - k);
    let meta = &pk.vk.cs;
    let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);
    let omega = pk.get_vk().domain.get_omega();

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

    let timer = start_timer!(|| format!("copy advice columns to gpu, count {}", advices.len()));
    let advices_device_buf = advices
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<DeviceResult<Vec<_>>>()?;
    end_timer!(timer);

    let timer = start_timer!(|| "copy g_lagrange buffer");
    let g_lagrange_buf = device
        .alloc_device_buffer_from_slice(&params.g_lagrange[..])
        .unwrap();
    end_timer!(timer);

    // thread for part of lookups
    let sub_pk = pk.clone();
    let sub_advices = advices.clone();
    let sub_instance = instance.clone();
    let lookup_handler = thread::spawn(move || {
        let pk = sub_pk;
        let advices = sub_advices;
        let instance = sub_instance;
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
                let mut z = Vec::new_in(HugePageAllocator);
                z.resize(size, C::ScalarExt::zero());
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
                    &pk.vk.cs.lookups[i].input_expressions[0],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[i].table_expressions[0],
                    &mut table[0..unusable_rows_start],
                );
                let (permuted_input, permuted_table) =
                    handle_lookup_pair(&input, &table, unusable_rows_start);
                (i, (permuted_input, permuted_table, input, table, z))
            })
            .collect::<Vec<_>>();
        //end_timer!(timer);

        let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instance[0]
            .instance_values
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..];

        let timer = start_timer!(|| format!("permute lookup comp {}", single_comp_lookups.len()));
        let single_comp_lookups = single_comp_lookups
            .into_par_iter()
            .map(|(i, (mut input, mut table, z))| {
                let f = |expr: &Expression<_>, target: &mut [_]| {
                    evaluate_expr(expr, size, 1, fixed_ref, advice_ref, instance_ref, target)
                };

                f(
                    &pk.vk.cs.lookups[i].input_expressions[0],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[i].table_expressions[0],
                    &mut table[0..unusable_rows_start],
                );
                let (permuted_input, permuted_table) =
                    handle_lookup_pair(&input, &table, unusable_rows_start);
                (i, (permuted_input, permuted_table, input, table, z))
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        (single_unit_lookups, single_comp_lookups, tuple_lookups)
    });

    // Advice MSM
    let timer = start_timer!(|| format!("advices msm {}", advices_device_buf.len()));
    for s_buf in advices_device_buf {
        let commitment = msm(&device, &g_lagrange_buf, &s_buf, size)?;
        transcript.write_point(commitment).unwrap();
    }
    end_timer!(timer);

    let theta: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("theta is {:?}", theta);

    let timer = start_timer!(|| "wait single lookups");
    let (mut single_unit_lookups, mut single_comp_lookups, tuple_lookups) =
        lookup_handler.join().unwrap();
    end_timer!(timer);

    // After theta
    let sub_pk = pk.clone();
    let sub_advices = advices.clone();
    let sub_instance = instance.clone();
    let tuple_lookup_handler = thread::spawn(move || {
        let pk = sub_pk;
        let advices = sub_advices;
        let instance = sub_instance;
        //let timer = start_timer!(|| format!("permute lookup tuple {}", tuple_lookups.len()));

        let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instance[0]
            .instance_values
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..];

        let tuple_lookups = tuple_lookups
            .into_par_iter()
            .map(|(i, (mut input, mut table, z))| {
                let f = |expr: &[Expression<_>], target: &mut [_]| {
                    evaluate_exprs(
                        expr,
                        size,
                        1,
                        fixed_ref,
                        advice_ref,
                        instance_ref,
                        theta,
                        target,
                    )
                };

                f(
                    &pk.vk.cs.lookups[i].input_expressions[..],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[i].table_expressions[..],
                    &mut table[0..unusable_rows_start],
                );
                let (permuted_input, permuted_table) =
                    handle_lookup_pair(&input, &table, unusable_rows_start);
                (i, (permuted_input, permuted_table, input, table, z))
            })
            .collect::<Vec<_>>();
        //end_timer!(timer);

        tuple_lookups
    });

    let mut lookup_permuted_commitments = vec![C::identity(); pk.vk.cs.lookups.len() * 2];

    let timer = start_timer!(|| format!(
        "single lookup msm {} {}",
        single_unit_lookups.len(),
        single_comp_lookups.len()
    ));
    for (i, (permuted_input, permuted_table, _, _, _)) in single_unit_lookups.iter() {
        let permuted_input_buf = device.alloc_device_buffer_from_slice(&permuted_input[..])?;
        let permuted_table_buf = device.alloc_device_buffer_from_slice(&permuted_table[..])?;
        lookup_permuted_commitments[i * 2] =
            msm(&device, &g_lagrange_buf, &permuted_input_buf, size)?;
        lookup_permuted_commitments[i * 2 + 1] =
            msm(&device, &g_lagrange_buf, &permuted_table_buf, size)?;
    }
    for (i, (permuted_input, permuted_table, _, _, _)) in single_comp_lookups.iter() {
        let permuted_input_buf = device.alloc_device_buffer_from_slice(&permuted_input[..])?;
        let permuted_table_buf = device.alloc_device_buffer_from_slice(&permuted_table[..])?;
        lookup_permuted_commitments[i * 2] =
            msm(&device, &g_lagrange_buf, &permuted_input_buf, size)?;
        lookup_permuted_commitments[i * 2 + 1] =
            msm(&device, &g_lagrange_buf, &permuted_table_buf, size)?;
    }
    end_timer!(timer);

    let timer = start_timer!(|| "wait tuple lookup");
    let mut tuple_lookups = tuple_lookup_handler.join().unwrap();
    end_timer!(timer);

    let timer = start_timer!(|| format!("tuple lookup msm {}", tuple_lookups.len(),));
    for (i, (permuted_input, permuted_table, _, _, _)) in tuple_lookups.iter() {
        let permuted_input_buf = device.alloc_device_buffer_from_slice(&permuted_input[..])?;
        let permuted_table_buf = device.alloc_device_buffer_from_slice(&permuted_table[..])?;
        lookup_permuted_commitments[i * 2] =
            msm(&device, &g_lagrange_buf, &permuted_input_buf, size)?;
        lookup_permuted_commitments[i * 2 + 1] =
            msm(&device, &g_lagrange_buf, &permuted_table_buf, size)?;
    }
    end_timer!(timer);

    for commitment in lookup_permuted_commitments.into_iter() {
        transcript.write_point(commitment).unwrap();
    }

    let beta: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("beta is {:?}", beta);
    let gamma: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("gamma is {:?}", gamma);

    let mut lookups = vec![];
    lookups.append(&mut single_unit_lookups);
    lookups.append(&mut single_comp_lookups);
    lookups.append(&mut tuple_lookups);
    lookups.sort_by(|l, r| usize::cmp(&l.0, &r.0));

    let timer = start_timer!(|| "generate lookup z");
    lookups
        .par_iter_mut()
        .for_each(|(_, (permuted_input, permuted_table, input, table, z))| {
            for ((z, permuted_input_value), permuted_table_value) in z
                .iter_mut()
                .zip(permuted_input.iter())
                .zip(permuted_table.iter())
            {
                *z = (beta + permuted_input_value) * &(gamma + permuted_table_value);
            }

            z.batch_invert();

            for ((z, input_value), table_value) in z.iter_mut().zip(input.iter()).zip(table.iter())
            {
                *z *= (beta + input_value) * &(gamma + table_value);
            }

            let mut tmp = C::ScalarExt::one();
            for i in 0..=unusable_rows_start {
                std::mem::swap(&mut tmp, &mut z[i]);
                tmp = tmp * z[i];
            }

            if ADD_RANDOM {
                for cell in &mut z[unusable_rows_start + 1..] {
                    *cell = C::Scalar::random(&mut OsRng);
                }
            } else {
                for cell in &mut z[unusable_rows_start + 1..] {
                    *cell = C::Scalar::zero();
                }
            }
        });

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
        .alloc_device_buffer_from_slice::<C::ScalarExt>(&[pk.get_vk().domain.ifft_divisor])?;
    end_timer!(timer);

    let chunk_len = &pk.vk.cs.degree() - 2;

    let timer = start_timer!(|| format!(
        "product permutation {}",
        (&pk).vk.cs.permutation.columns.chunks(chunk_len).len()
    ));

    let sub_pk = pk.clone();
    let sub_advices = advices.clone();
    let sub_instance = instance.clone();
    let permutation_products_handler = thread::spawn(move || {
        let pk = sub_pk;
        let advices = sub_advices;
        let instance = sub_instance;

        let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
        let instance_ref = &instance[0]
            .instance_values
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..];
        let mut p_z = pk
            .vk
            .cs
            .permutation
            .columns
            .par_chunks(chunk_len)
            .zip((&pk).permutation.permutations.par_chunks(chunk_len))
            .enumerate()
            .map(|(i, (columns, permutations))| {
                let mut delta_omega = C::Scalar::DELTA.pow_vartime([i as u64 * chunk_len as u64]);

                let mut modified_values = Vec::new_in(HugePageAllocator);
                modified_values.resize(size, C::ScalarExt::one());

                // Iterate over each column of the permutation
                for (&column, permuted_column_values) in columns.iter().zip(permutations.iter()) {
                    let values = match column.column_type() {
                        Any::Advice => advice_ref,
                        Any::Fixed => fixed_ref,
                        Any::Instance => instance_ref,
                    };
                    for i in 0..size as usize {
                        modified_values[i] *= &(beta * permuted_column_values[i]
                            + &gamma
                            + values[column.index()][i]);
                    }
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
                    for i in 0..size as usize {
                        modified_values[i] *=
                            &(delta_omega * &beta + &gamma + values[column.index()][i]);
                        delta_omega *= &omega;
                    }
                    delta_omega *= &C::Scalar::DELTA;
                }

                modified_values
            })
            .collect::<Vec<_>>();

        let mut tmp = C::ScalarExt::one();
        for z in p_z.iter_mut() {
            for i in 0..size {
                std::mem::swap(&mut tmp, &mut z[i]);
                tmp = tmp * z[i];
            }

            tmp = z[unusable_rows_start];

            for v in z[unusable_rows_start + 1..].iter_mut() {
                if ADD_RANDOM {
                    *v = C::Scalar::random(&mut OsRng);
                }
            }
        }
        p_z
    });
    end_timer!(timer);

    let mut lookup_z_commitments = vec![];

    let timer = start_timer!(|| "lookup intt and z msm");
    let mut tmp_buf = device.alloc_device_buffer::<C::ScalarExt>(size)?;
    let mut ntt_buf = device.alloc_device_buffer::<C::ScalarExt>(size)?;
    for (permuted_input, permuted_table, input, table, z) in lookups.iter_mut() {
        device.copy_from_host_to_device(&ntt_buf, &z[..])?;
        let commitment = msm_with_groups(&device, &g_lagrange_buf, &ntt_buf, size, 1)?;
        lookup_z_commitments.push(commitment);
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut z[..], &ntt_buf)?;

        device.copy_from_host_to_device(&ntt_buf, &permuted_input[..])?;
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut permuted_input[..], &ntt_buf)?;

        device.copy_from_host_to_device(&ntt_buf, &permuted_table[..])?;
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut permuted_table[..], &ntt_buf)?;
    }
    end_timer!(timer);

    let timer = start_timer!(|| "wait permutation_products");
    let mut permutation_products = permutation_products_handler.join().unwrap();
    end_timer!(timer);

    let timer = start_timer!(|| "permutation z msm and intt");
    for (i, z) in permutation_products.iter_mut().enumerate() {
        device.copy_from_host_to_device(&ntt_buf, &z[..])?;
        let commitment = msm_with_groups(&device, &g_lagrange_buf, &ntt_buf, size, 1)?;
        transcript.write_point(commitment).unwrap();
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &intt_pq_buf,
            &intt_omegas_buf,
            &intt_divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut z[..], &ntt_buf)?;
    }

    for (i, commitment) in lookup_z_commitments.into_iter().enumerate() {
        transcript.write_point(commitment).unwrap();
    }

    end_timer!(timer);
    let vanishing =
        halo2_proofs::plonk::vanishing::Argument::commit(params, &pk.vk.domain, OsRng, transcript)
            .unwrap();

    let y: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("y is {:?}", y);

    let timer = start_timer!(|| "h_poly");
    {
        let timer = start_timer!(|| "advices intt");

        for advices in unsafe { Arc::get_mut_unchecked(&mut advices) }.iter_mut() {
            device.copy_from_host_to_device(&ntt_buf, &advices[..])?;
            intt_raw(
                &device,
                &mut ntt_buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            device.copy_from_device_to_host(&mut advices[..], &ntt_buf)?;
        }
        end_timer!(timer);
    }

    let fixed_ref = &pk.fixed_polys.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let instance_ref = &instance[0]
        .instance_polys
        .iter()
        .map(|x| &x[..])
        .collect::<Vec<_>>()[..];

    let mut h = Vec::new_in(HugePageAllocator);
    h.resize(1 << extended_k, C::ScalarExt::zero());
    end_timer!(timer);

    evaluate_h_gates(
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
        &mut h[..],
    )?;
    end_timer!(timer);

    Ok(())
}

struct EvalHContext<F: FieldExt> {
    y: Vec<F>,
    extended_allocator: Vec<CudaDeviceBufRaw>,
    extended_k: usize,
    k: usize,
    size: usize,
    extended_size: usize,
    extended_ntt_omegas_buf: CudaDeviceBufRaw,
    extended_ntt_pq_buf: CudaDeviceBufRaw,
    coset_powers_buf: CudaDeviceBufRaw,
}

impl<F: FieldExt> EvalHContext<F> {
    fn alloc(&mut self, device: &CudaDevice) -> DeviceResult<CudaDeviceBufRaw> {
        let buf = self.extended_allocator.pop();
        if buf.is_none() {
            device.alloc_device_buffer::<F>(self.extended_size)
        } else {
            Ok(buf.unwrap())
        }
    }
}

pub(crate) fn analyze_expr_tree<F: FieldExt>(
    expr: &ProveExpression<F>
) -> Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>> {
    let tree = expr.clone().flatten();
    let tree = tree
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    let limit = 20;
    let mut v = HashSet::new();

    let mut expr_group = vec![];
    let mut expr_groups = vec![];
    for (_, (units, coeff)) in tree.iter().enumerate() {
        let mut v_new = v.clone();
        let mut v_new_clean = HashSet::new();
        let mut muls_new = 0;
        for (unit, exp) in units {
            v_new.insert(unit.get_group());
            v_new_clean.insert(unit.get_group());
            muls_new += exp;
        }
        muls_new -= 1;

        if v_new.len() > limit {
            v = v_new_clean;

            expr_groups.push(expr_group);
            expr_group = vec![(units.clone(), coeff.clone())];
        } else {
            v = v_new;
            expr_group.push((units.clone(), coeff.clone()));
        }
    }

    expr_groups.push(expr_group);
    expr_groups
}

pub fn export_evaluate_h_gates<C: CurveAffine>(
    pk: &ProvingKey<C>,
    fixed: &[&[C::ScalarExt]],
    advice: &[&[C::ScalarExt]],
    instance: &[&[C::ScalarExt]],
    permutation_products: &[&[C::ScalarExt]],
    lookup_products: &[(
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
    )],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    res: &mut [C::ScalarExt],
) {
    let device = CudaDevice::get_device(0).unwrap();
    let (intt_omegas_buf, intt_pq_buf) = ntt_prepare(
        &device,
        pk.get_vk().domain.get_omega_inv(),
        pk.vk.domain.k() as usize,
    )
    .unwrap();
    let intt_divisor_buf = device
        .alloc_device_buffer_from_slice::<C::ScalarExt>(&[pk.get_vk().domain.ifft_divisor])
        .unwrap();

    let h_buf = evaluate_h_gates(
        &device,
        pk,
        fixed,
        advice,
        instance,
        permutation_products,
        lookup_products,
        y,
        beta,
        gamma,
        theta,
        intt_pq_buf,
        intt_omegas_buf,
        intt_divisor_buf,
        res,
    )
    .unwrap();
}

fn evaluate_h_gates<C: CurveAffine>(
    device: &CudaDevice,
    pk: &ProvingKey<C>,
    fixed: &[&[C::ScalarExt]],
    advice: &[&[C::ScalarExt]],
    instance: &[&[C::ScalarExt]],
    permutation_products: &[&[C::ScalarExt]],
    lookup_products: &[(
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
        &[C::ScalarExt],
    )],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    intt_pq_buf: CudaDeviceBufRaw,
    intt_omegas_buf: CudaDeviceBufRaw,
    intt_divisor_buf: CudaDeviceBufRaw,
    res: &mut [C::ScalarExt],
) -> DeviceResult<()> {
    let timer = start_timer!(|| "evaluate_h setup");
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
    let omega = pk.vk.domain.get_omega();
    let extended_k = pk.get_vk().domain.extended_k() as usize;
    let extended_size = 1 << extended_k;
    let extended_omega = pk.vk.domain.get_extended_omega();

    let (extended_ntt_omegas_buf, extended_ntt_pq_buf) =
        ntt_prepare(device, extended_omega, extended_k)?;
    let coset_powers_buf = device.alloc_device_buffer_from_slice(&[
        pk.get_vk().domain.g_coset,
        pk.get_vk().domain.g_coset_inv,
    ])?;

    let mut ctx = EvalHContext {
        y: vec![C::ScalarExt::one(), y],
        extended_allocator: vec![],
        k,
        extended_k,
        size,
        extended_size,
        extended_ntt_omegas_buf,
        extended_ntt_pq_buf,
        coset_powers_buf,
    };

    let timer = start_timer!(|| "evaluate_h gates");
    assert!(pk.ev.gpu_gates_expr.len() == 1);
    let exprs = analyze_expr_tree(&pk.ev.gpu_gates_expr[0]);
    let h_buf = evaluate_prove_expr(
        device,
        &exprs,
        fixed,
        advice,
        instance,
        &mut ctx,
    )?;
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h prepare buffers for constants");
    let y_buf = device.alloc_device_buffer_from_slice(&[y][..])?;
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..])?;
    let gamma_buf = device.alloc_device_buffer_from_slice(&[gamma][..])?;

    let l0 = &pk.l0;
    let l_last = &pk.l_last;
    let l_active_row = &pk.l_active_row;
    let l0_buf = do_extended_ntt_v2(device, &mut ctx, &l0.values[..])?;
    let l_last_buf = do_extended_ntt_v2(device, &mut ctx, &l_last.values[..])?;
    let l_active_buf = device.alloc_device_buffer_from_slice(&l_active_row.values[..])?;
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h permutation");
    if permutation_products.len() > 0 {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let last_rotation = (ctx.size - (blinding_factors + 1)) << (extended_k - k);
        let chunk_len = pk.vk.cs.degree() - 2;

        let extended_p_buf = permutation_products
            .iter()
            .map(|x| do_extended_ntt_v2(device, &mut ctx, x))
            .collect::<Result<Vec<_>, _>>()?;

        {
            permutation_eval_h_p1(
                device,
                &h_buf,
                extended_p_buf.first().unwrap(),
                extended_p_buf.last().unwrap(),
                &l0_buf,
                &l_last_buf,
                &y_buf,
                ctx.extended_size,
            )?;

            permutation_eval_h_p2(
                device,
                &h_buf,
                &extended_p_buf[..],
                &l0_buf,
                &l_last_buf,
                &y_buf,
                last_rotation,
                ctx.extended_size,
            )?;

            let mut curr_delta = beta * &C::Scalar::ZETA;
            for ((extended_p_buf, columns), polys) in extended_p_buf
                .into_iter()
                .zip(pk.vk.cs.permutation.columns.chunks(chunk_len))
                .zip(pk.permutation.polys.chunks(chunk_len))
            {
                let l = ctx.alloc(device)?;
                buffer_copy_with_shift::<C::ScalarExt>(
                    &device,
                    &l,
                    &extended_p_buf,
                    1 << (extended_k - k),
                    ctx.extended_size,
                )?;

                let r = extended_p_buf;

                for (value, permutation) in columns
                    .iter()
                    .map(|&column| match column.column_type() {
                        Any::Advice => &advice[column.index()],
                        Any::Fixed => &fixed[column.index()],
                        Any::Instance => &instance[column.index()],
                    })
                    .zip(polys.iter())
                {
                    let mut l_res = ctx.alloc(device)?;
                    let mut r_res = ctx.alloc(device)?;
                    let p_coset_buf = ctx.alloc(device)?;
                    device.copy_from_host_to_device(&p_coset_buf, &permutation.values[..])?;

                    device.copy_from_host_to_device(&l_res, value)?;
                    device
                        .copy_from_device_to_device::<C::Scalar>(&r_res, 0, &l_res, 0, ctx.size)?;
                    permutation_eval_h_l(
                        &device,
                        &l_res,
                        &beta_buf,
                        &gamma_buf,
                        &p_coset_buf,
                        ctx.size,
                    )?;
                    do_extended_ntt(&device, &mut ctx, &mut l_res)?;
                    field_mul::<C::ScalarExt>(&device, &l, &l_res, ctx.extended_size)?;

                    do_extended_prepare(device, &mut ctx, &mut r_res)?;
                    let coeff =
                        pick_from_buf::<C::ScalarExt>(device, &r_res, 0, 1, ctx.extended_size)?;
                    let short = vec![value[0] + gamma, coeff + curr_delta];
                    device.copy_from_host_to_device(&r_res, &short[..])?;
                    do_extended_ntt_pure(device, &mut ctx, &mut r_res)?;

                    field_mul::<C::ScalarExt>(&device, &r, &r_res, ctx.extended_size)?;
                    curr_delta *= &C::Scalar::DELTA;

                    ctx.extended_allocator.push(l_res);
                    ctx.extended_allocator.push(r_res);
                    ctx.extended_allocator.push(p_coset_buf);
                }

                field_sub::<C::ScalarExt>(&device, &l, &r, ctx.extended_size)?;
                field_mul::<C::ScalarExt>(&device, &l, &l_active_buf, ctx.extended_size)?;
                field_op_v2::<C::ScalarExt>(
                    &device,
                    &h_buf,
                    Some(&h_buf),
                    None,
                    None,
                    Some(y),
                    ctx.extended_size,
                    FieldOp::Mul,
                )?;
                field_sum::<C::ScalarExt>(&device, &h_buf, &l, ctx.extended_size)?;

                ctx.extended_allocator.push(l);
                ctx.extended_allocator.push(r);
            }
        }
    }
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h lookup");
    for (i, (lookup, (permuted_input, permuted_table, input, table, z))) in pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookup_products.iter())
        .enumerate()
    {
        let input_deg = get_expr_degree(&lookup.input_expressions);
        let table_deg = get_expr_degree(&lookup.table_expressions);

        let [e1, e2] = flatten_lookup_expression(
            &lookup.input_expressions,
            &lookup.table_expressions,
            beta,
            gamma,
            theta,
        );

        let input_buf = if input_deg > 1 {
            evaluate_prove_expr(device, &vec![e1], fixed, advice, instance, &mut ctx)?
        } else {
            let mut buf = ctx.alloc(device)?;
            device.copy_from_host_to_device(&buf, &input)?;

            let mut tmp_buf = ctx.alloc(device)?;
            intt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            ctx.extended_allocator.push(tmp_buf);

            let coeff = pick_from_buf::<C::ScalarExt>(device, &buf, 0, 0, ctx.size)?;
            let short = vec![coeff + beta];
            device.copy_from_host_to_device(&buf, &short[..])?;
            do_extended_ntt(device, &mut ctx, &mut buf)?;

            buf
        };
        let table_buf = if table_deg > 1 {
            evaluate_prove_expr(device, &vec![e2], fixed, advice, instance, &mut ctx)?
        } else {
            let mut buf = ctx.alloc(device)?;
            device.copy_from_host_to_device(&buf, &table)?;

            let mut tmp_buf = ctx.alloc(device)?;
            intt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            ctx.extended_allocator.push(tmp_buf);

            let coeff = pick_from_buf::<C::ScalarExt>(device, &buf, 0, 0, ctx.size)?;
            let short = vec![coeff + gamma];
            device.copy_from_host_to_device(&buf, &short[..])?;
            do_extended_ntt(device, &mut ctx, &mut buf)?;

            buf
        };

        let permuted_input_buf = do_extended_ntt_v2(device, &mut ctx, permuted_input)?;
        let permuted_table_buf = do_extended_ntt_v2(device, &mut ctx, permuted_table)?;
        let z_buf = do_extended_ntt_v2(device, &mut ctx, z)?;

        unsafe {
            let err = lookup_eval_h(
                h_buf.ptr(),
                input_buf.ptr(),
                table_buf.ptr(),
                permuted_input_buf.ptr(),
                permuted_table_buf.ptr(),
                z_buf.ptr(),
                l0_buf.ptr(),
                l_last_buf.ptr(),
                l_active_buf.ptr(),
                y_buf.ptr(),
                beta_buf.ptr(),
                gamma_buf.ptr(),
                1 << (extended_k - k),
                ctx.extended_size as i32,
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;
            device.synchronize()?;
        }

        ctx.extended_allocator.push(input_buf);
        ctx.extended_allocator.push(table_buf);
        ctx.extended_allocator.push(permuted_input_buf);
        ctx.extended_allocator.push(permuted_table_buf);
        ctx.extended_allocator.push(z_buf);
    }

    device.copy_from_device_to_host(res, &h_buf)?;
    println!("after lookup res[0..4] is {:?}", &res[0..4]);
    end_timer!(timer);

    Ok(())
}

fn get_expr_degree<F: FieldExt>(expr: &Vec<halo2_proofs::plonk::circuit::Expression<F>>) -> usize {
    let mut deg = 0;
    for expr in expr {
        deg = deg.max(expr.degree());
    }
    deg
}

fn flatten_lookup_expression<F: FieldExt>(
    input: &Vec<halo2_proofs::plonk::circuit::Expression<F>>,
    table: &Vec<halo2_proofs::plonk::circuit::Expression<F>>,
    beta: F,
    gamma: F,
    theta: F,
) -> [Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>; 2] {
    let mut expr_input = ProveExpression::<F>::from_expr(&input[0]);
    for input in input.iter().skip(1) {
        expr_input = ProveExpression::Scale(
            Box::new(expr_input),
            BTreeMap::from_iter([(0, theta)].into_iter()),
        );
        expr_input = ProveExpression::Op(
            Box::new(expr_input),
            Box::new(ProveExpression::<F>::from_expr(input)),
            Bop::Sum,
        );
    }

    expr_input = ProveExpression::Op(
        Box::new(expr_input),
        Box::new(ProveExpression::Y(BTreeMap::from_iter(
            [(0, beta)].into_iter(),
        ))),
        Bop::Sum,
    );

    let expr_input = expr_input
        .flatten()
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    let mut expr_table = ProveExpression::<F>::from_expr(&table[0]);
    for table in table.iter().skip(1) {
        expr_table = ProveExpression::Scale(
            Box::new(expr_table),
            BTreeMap::from_iter([(0, theta)].into_iter()),
        );
        expr_table = ProveExpression::Op(
            Box::new(expr_table),
            Box::new(ProveExpression::<F>::from_expr(table)),
            Bop::Sum,
        );
    }

    expr_table = ProveExpression::Op(
        Box::new(expr_table),
        Box::new(ProveExpression::Y(BTreeMap::from_iter(
            [(0, gamma)].into_iter(),
        ))),
        Bop::Sum,
    );

    let expr_table = expr_table
        .flatten()
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    [expr_input, expr_table]
}

fn do_extended_ntt_v2<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<CudaDeviceBufRaw> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        buf.unwrap()
    };
    device.copy_from_host_to_device::<F>(&buf, data)?;
    do_extended_ntt(device, ctx, &mut buf)?;

    Ok(buf)
}

fn do_extended_prepare<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
) -> DeviceResult<()> {
    extended_prepare(
        device,
        data,
        &ctx.coset_powers_buf,
        3,
        ctx.size,
        ctx.extended_size,
    )
}

fn do_extended_ntt<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
) -> DeviceResult<()> {
    do_extended_prepare(device, ctx, data)?;
    do_extended_ntt_pure(device, ctx, data)?;
    Ok(())
}

fn do_extended_ntt_pure<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
) -> DeviceResult<()> {
    let tmp = ctx.extended_allocator.pop();
    let mut tmp = if tmp.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        tmp.unwrap()
    };
    ntt_raw(
        device,
        data,
        &mut tmp,
        &ctx.extended_ntt_pq_buf,
        &ctx.extended_ntt_omegas_buf,
        ctx.extended_k,
    )?;
    device.synchronize()?;
    ctx.extended_allocator.push(tmp);
    Ok(())
}

fn eval_ys<F: FieldExt>(ys: &BTreeMap<u32, F>, ctx: &mut EvalHContext<F>) -> F {
    let max_y_order = *ys.keys().max().unwrap();
    for _ in (ctx.y.len() as u32)..=max_y_order {
        ctx.y.push(ctx.y[1] * ctx.y.last().unwrap());
    }
    ys.iter().fold(F::zero(), |acc, (y_order, f)| {
        acc + ctx.y[*y_order as usize] * f
    })
}

fn evaluate_prove_expr<F: FieldExt>(
    device: &CudaDevice,
    exprs: &Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>>,
    fixed: &[&[F]],
    advice: &[&[F]],
    instance: &[&[F]],
    ctx: &mut EvalHContext<F>,
) -> DeviceResult<CudaDeviceBufRaw> {
    let res = ctx.alloc(device)?;
    unsafe {
        cudaMemset(res.ptr(), 0, ctx.extended_size * core::mem::size_of::<F>());
    }
    let mut last_bufs = BTreeMap::new();

    for expr in exprs.iter() {
        let mut bufs = BTreeMap::new();
        let mut coeffs = vec![];
        for (_, ys) in expr {
            coeffs.push(eval_ys(&ys, ctx));
        }
        let coeffs_buf = device.alloc_device_buffer_from_slice(&coeffs[..])?;

        unsafe {
            let mut group = vec![];
            let mut rots = vec![];

            for (units, _) in expr.iter() {
                for (u, _) in units {
                    let id = u.get_group();
                    if !bufs.contains_key(&id) && last_bufs.contains_key(&id) {
                        let buf = last_bufs.remove(&id).unwrap();
                        bufs.insert(id, buf);
                    }
                }
            }

            for (_, buf) in last_bufs {
                ctx.extended_allocator.push(buf)
            }

            for (i, (units, _)) in expr.iter().enumerate() {
                group.push(
                    coeffs_buf
                        .ptr()
                        .offset((i * core::mem::size_of::<F>()) as isize),
                );

                for (u, exp) in units {
                    let id = u.get_group();
                    let (src, rot) = match u {
                        ProveExpressionUnit::Fixed {
                            column_index,
                            rotation,
                        } => (&fixed[*column_index], rotation),
                        ProveExpressionUnit::Advice {
                            column_index,
                            rotation,
                        } => (&advice[*column_index], rotation),
                        ProveExpressionUnit::Instance {
                            column_index,
                            rotation,
                        } => (&instance[*column_index], rotation),
                    };
                    if !bufs.contains_key(&id) {
                        let buf = do_extended_ntt_v2(device, ctx, src)?;
                        bufs.insert(id, buf);
                    }
                    for _ in 0..*exp {
                        group.push(bufs.get(&id).unwrap().ptr());
                        rots.push(rot.0 << (ctx.extended_k - ctx.k));
                    }
                }

                group.push(0usize as _);
            }
            let group_buf = device.alloc_device_buffer_from_slice(&group[..])?;
            let rots_buf = device.alloc_device_buffer_from_slice(&rots[..])?;

            let err = cuda::bn254_c::field_op_batch_mul_sum(
                res.ptr(),
                group_buf.ptr(),
                rots_buf.ptr(),
                group.len() as i32,
                ctx.extended_size as i32,
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;

            last_bufs = bufs;
        }
    }

    Ok(res)
}
