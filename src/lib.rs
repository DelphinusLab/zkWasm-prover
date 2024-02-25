#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

use std::iter;
use std::sync::Arc;
use std::thread;

use ark_std::end_timer;
use ark_std::iterable::Iterable;
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
use halo2_proofs::poly::multiopen::ProverQueryGeneral;
use halo2_proofs::poly::multiopen::Query;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::IntoParallelRefMutIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::slice::ParallelSlice as _;

use crate::cuda::bn254::field_op_v3;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::intt_raw_async;
use crate::cuda::bn254::msm;
use crate::cuda::bn254::msm_with_groups;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254::FieldOp;
use crate::device::cuda::CudaDevice;
use crate::device::Device as _;
use crate::device::DeviceResult;
use crate::eval_h::evaluate_h_gates;
use crate::hugetlb::HugePageAllocator;

pub mod cuda;
pub mod device;
mod eval_h;
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
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    mut advices: Arc<Vec<Vec<C::Scalar, HugePageAllocator>>>,
    transcript: &mut T,
) -> Result<(), Error> {
    thread::scope(|s| {
        let k = pk.get_vk().domain.k() as usize;
        let size = 1 << pk.get_vk().domain.k();
        let extended_k = pk.get_vk().domain.extended_k() as usize;
        let meta = &pk.vk.cs;
        let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);
        let omega = pk.get_vk().domain.get_omega();

        let domain = &pk.vk.domain;

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

        let timer = start_timer!(|| "copy g_lagrange buffer");
        let g_lagrange_buf = device
            .alloc_device_buffer_from_slice(&params.g_lagrange[..])
            .unwrap();
        end_timer!(timer);

        // thread for part of lookups
        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instance = instance.clone();
        let lookup_handler = s.spawn(move || {
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
                            target.clone_from_slice(
                                &pk.fixed_values[idx].values[0..unusable_rows_start],
                            );
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

            let timer =
                start_timer!(|| format!("permute lookup comp {}", single_comp_lookups.len()));
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
        let timer = start_timer!(|| format!("advices msm {}", advices.len()));
        let s_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        for advice in advices.iter() {
            device.copy_from_host_to_device(&s_buf, &advice[..])?;
            let commitment = msm(&device, &g_lagrange_buf, &s_buf, size)?;
            transcript.write_point(commitment).unwrap();
        }
        end_timer!(timer);

        let theta: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "wait single lookups");
        let (mut single_unit_lookups, mut single_comp_lookups, tuple_lookups) =
            lookup_handler.join().unwrap();
        end_timer!(timer);

        // After theta
        let sub_pk = pk.clone();
        let sub_advices = advices.clone();
        let sub_instance = instance.clone();
        let tuple_lookup_handler = s.spawn(move || {
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
            device.copy_from_host_to_device(&s_buf, &permuted_input[..])?;
            lookup_permuted_commitments[i * 2] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
            device.copy_from_host_to_device(&s_buf, &permuted_table[..])?;
            lookup_permuted_commitments[i * 2 + 1] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
        }
        for (i, (permuted_input, permuted_table, _, _, _)) in single_comp_lookups.iter() {
            device.copy_from_host_to_device(&s_buf, &permuted_input[..])?;
            lookup_permuted_commitments[i * 2] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
            device.copy_from_host_to_device(&s_buf, &permuted_table[..])?;
            lookup_permuted_commitments[i * 2 + 1] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
        }
        end_timer!(timer);

        let timer = start_timer!(|| "wait tuple lookup");
        let mut tuple_lookups = tuple_lookup_handler.join().unwrap();
        end_timer!(timer);

        let timer = start_timer!(|| format!("tuple lookup msm {}", tuple_lookups.len(),));
        for (i, (permuted_input, permuted_table, _, _, _)) in tuple_lookups.iter() {
            device.copy_from_host_to_device(&s_buf, &permuted_input[..])?;
            lookup_permuted_commitments[i * 2] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
            device.copy_from_host_to_device(&s_buf, &permuted_table[..])?;
            lookup_permuted_commitments[i * 2 + 1] = msm(&device, &g_lagrange_buf, &s_buf, size)?;
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
        lookups.par_iter_mut().for_each(
            |(_, (permuted_input, permuted_table, input, table, z))| {
                for ((z, permuted_input_value), permuted_table_value) in z
                    .iter_mut()
                    .zip(permuted_input.iter())
                    .zip(permuted_table.iter())
                {
                    *z = (beta + permuted_input_value) * &(gamma + permuted_table_value);
                }

                z.batch_invert();

                for ((z, input_value), table_value) in
                    z.iter_mut().zip(input.iter()).zip(table.iter())
                {
                    *z *= (beta + input_value) * &(gamma + table_value);
                }

                let mut tmp = C::Scalar::one();
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
            },
        );

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
        let sub_instance = instance.clone();
        let permutation_products_handler = s.spawn(move || {
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

            let mut tmp = C::Scalar::one();
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
        let mut tmp_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        let mut ntt_buf = device.alloc_device_buffer::<C::Scalar>(size)?;
        for (permuted_input, permuted_table, _, _, z) in lookups.iter_mut() {
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
        for (_i, z) in permutation_products.iter_mut().enumerate() {
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

        for (_i, commitment) in lookup_z_commitments.into_iter().enumerate() {
            transcript.write_point(commitment).unwrap();
        }

        end_timer!(timer);
        let vanishing = halo2_proofs::plonk::vanishing::Argument::commit(
            params,
            &pk.vk.domain,
            OsRng,
            transcript,
        )
        .unwrap();

        let y: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();

        let timer = start_timer!(|| "h_poly");
        {
            let timer = start_timer!(|| "advices intt");

            let mut last_stream = None;
            let mut last_tmp_buf = Some(device.alloc_device_buffer::<C::Scalar>(size)?);
            let mut last_ntt_buf = Some(device.alloc_device_buffer::<C::Scalar>(size)?);
            for advices in unsafe { Arc::get_mut_unchecked(&mut advices) }.iter_mut() {
                unsafe {
                    let mut stream = std::mem::zeroed();
                    let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                    crate::device::cuda::to_result((), err, "fail to run cudaStreamCreate")?;
                    device.copy_from_host_to_device_async(&ntt_buf, &advices[..], stream)?;
                    intt_raw_async(
                        &device,
                        &mut ntt_buf,
                        &mut tmp_buf,
                        &intt_pq_buf,
                        &intt_omegas_buf,
                        &intt_divisor_buf,
                        k,
                        Some(stream),
                    )?;
                    device.copy_from_device_to_host_async(&mut advices[..], &ntt_buf, stream)?;
                    if let Some(last_stream) = last_stream {
                        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                        cuda_runtime_sys::cudaStreamDestroy(last_stream);
                    }
                    std::mem::swap(&mut ntt_buf, last_ntt_buf.as_mut().unwrap());
                    std::mem::swap(&mut tmp_buf, last_tmp_buf.as_mut().unwrap());
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
        let instance_ref = &instance[0]
            .instance_polys
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..];

        let mut h = Vec::new_in(HugePageAllocator);
        h.resize(1 << extended_k, C::Scalar::zero());

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

        let timer = start_timer!(|| "vanishing construct");
        // Construct the vanishing argument's h(X) commitments
        let vanishing = vanishing
            .construct_general(params, domain, &mut h[..], transcript)
            .unwrap();

        let x: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let xn = x.pow_vartime(&[params.n as u64]);
        end_timer!(timer);

        let mut inputs = vec![];

        for instance in instance.iter() {
            meta.instance_queries.iter().for_each(|&(column, at)| {
                inputs.push((
                    &instance.instance_polys[column.index()].values[..],
                    domain.rotate_omega(x, at),
                ))
            })
        }

        meta.advice_queries.iter().for_each(|&(column, at)| {
            inputs.push((&advices[column.index()], domain.rotate_omega(x, at)))
        });

        meta.fixed_queries.iter().for_each(|&(column, at)| {
            inputs.push((&pk.fixed_polys[column.index()], domain.rotate_omega(x, at)))
        });

        inputs.push((&vanishing.committed.random_poly, x));

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
            inputs.push((z, x));
            inputs.push((z, x_next));
            inputs.push((permuted_input, x));
            inputs.push((permuted_input, x_inv));
            inputs.push((permuted_table, x));
        }

        let timer = start_timer!(|| format!("compute eval {}", inputs.len()));
        for eval in inputs
            .into_par_iter()
            .map(|(a, b)| halo2_proofs::arithmetic::eval_polynomial_st(a, b))
            .collect::<Vec<_>>()
        {
            transcript.write_scalar(eval).unwrap();
        }
        end_timer!(timer);

        let timer = start_timer!(|| "multi open");
        let h_pieces = vanishing
            .h_pieces
            .iter()
            .rev()
            .fold(domain.empty_coeff(), |acc, eval| acc * xn + eval);

        let advices_arr = [advices];
        let permutation_products_arr = [permutation_products];
        let lookups_arr = [lookups];

        let instances = instance
            .iter()
            .zip(advices_arr.iter())
            .zip(permutation_products_arr.iter())
            .zip(lookups_arr.iter())
            .flat_map(|(((instance, advice), permutation), lookups)| {
                iter::empty()
                    .chain((&pk).vk.cs.instance_queries.iter().map(|&(column, at)| {
                        ProverQueryGeneral {
                            point: domain.rotate_omega(x, at),
                            rotation: at,
                            poly: &instance.instance_polys[column.index()].values[..],
                        }
                    }))
                    .chain((&pk).vk.cs.advice_queries.iter().map(|&(column, at)| {
                        ProverQueryGeneral {
                            point: domain.rotate_omega(x, at),
                            rotation: at,
                            poly: &advice[column.index()],
                        }
                    }))
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
                    .map(|&(column, at)| ProverQueryGeneral {
                        point: domain.rotate_omega(x, at),
                        rotation: at,
                        poly: &pk.fixed_polys[column.index()],
                    }),
            )
            .chain(
                (&pk)
                    .permutation
                    .polys
                    .iter()
                    .map(move |poly| ProverQueryGeneral {
                        point: x,
                        rotation: Rotation::cur(),
                        poly: &poly.values[..],
                    }),
            )
            // We query the h(X) polynomial at x
            .chain(
                iter::empty()
                    .chain(Some(ProverQueryGeneral {
                        point: x,
                        rotation: Rotation::cur(),
                        poly: &h_pieces,
                    }))
                    .chain(Some(ProverQueryGeneral {
                        point: x,
                        rotation: Rotation::cur(),
                        poly: &vanishing.committed.random_poly,
                    })),
            );

        multiopen(&device, params, transcript, instances)?;
        end_timer!(timer);

        Ok(())
    })
}

fn multiopen<'a, I, C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    device: &CudaDevice,
    params: &Params<C>,
    transcript: &mut T,
    queries: I,
) -> DeviceResult<()>
where
    I: IntoIterator<Item = ProverQueryGeneral<'a, C>>,
{
    let v: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
    let commitment_data = halo2_proofs::poly::multiopen::construct_intermediate_sets(queries);

    // Sort by len to compute large batch first
    let ws = commitment_data
        .into_par_iter()
        .map(|commitment_at_a_point| -> DeviceResult<_> {
            //println!("queries {}", commitment_at_a_point.queries.len());
            let poly_batch_buf = device.alloc_device_buffer::<C::Scalar>(params.n as usize)?;
            let tmp_buf = device.alloc_device_buffer::<C::Scalar>(params.n as usize)?;
            let c_buf = device.alloc_device_buffer_from_slice(&[v][..])?;

            let mut poly_batch = Vec::new_in(HugePageAllocator);
            poly_batch.resize(params.n as usize, C::Scalar::zero());

            let z = commitment_at_a_point.point;

            device.copy_from_host_to_device(
                &poly_batch_buf,
                commitment_at_a_point.queries[0].get_commitment(),
            )?;

            for query in commitment_at_a_point.queries.iter().skip(1) {
                assert_eq!(query.get_point(), z);
                device.copy_from_host_to_device(&tmp_buf, query.get_commitment())?;

                field_op_v3(
                    device,
                    &poly_batch_buf,
                    Some(&poly_batch_buf),
                    Some(&c_buf),
                    Some(&tmp_buf),
                    None,
                    params.n as usize,
                    FieldOp::Sum,
                    None,
                )?;
            }

            device.copy_from_device_to_host(&mut poly_batch[..], &poly_batch_buf)?;
            let eval_batch = halo2_proofs::arithmetic::eval_polynomial_st(
                &poly_batch,
                commitment_at_a_point.queries[0].get_point(),
            );
            poly_batch[0] -= eval_batch;
            let witness_poly = halo2_proofs::arithmetic::kate_division(&poly_batch, z);
            Ok(witness_poly)
        })
        .collect::<DeviceResult<Vec<_>>>()?;

    let timer = start_timer!(|| "msm");
    let s_buf = device.alloc_device_buffer::<C::Scalar>(params.n as usize)?;
    let p_buf = device.alloc_device_buffer_from_slice(&params.g[..])?;
    for witness_poly in ws {
        device.copy_from_host_to_device(&s_buf, &witness_poly[..])?;
        let commitment = msm_with_groups(&device, &p_buf, &s_buf, params.n as usize, 1)?;
        transcript.write_point(commitment).unwrap();
    }
    end_timer!(timer);

    Ok(())
}

fn permutation_product_open<'a, C: CurveAffine>(
    pk: &'a ProvingKey<C>,
    products: &'a [Vec<C::Scalar, HugePageAllocator>],
    x: C::Scalar,
) -> impl Iterator<Item = ProverQueryGeneral<'a, C>> + Clone {
    let blinding_factors = pk.vk.cs.blinding_factors();
    let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());
    let x_last = pk
        .vk
        .domain
        .rotate_omega(x, Rotation(-((blinding_factors + 1) as i32)));

    iter::empty()
        .chain(products.iter().flat_map(move |product| {
            iter::empty()
                .chain(Some(ProverQueryGeneral {
                    point: x,
                    rotation: Rotation::cur(),
                    poly: &product,
                }))
                .chain(Some(ProverQueryGeneral {
                    point: x_next,
                    rotation: Rotation::next(),
                    poly: &product,
                }))
        }))
        .chain(products.iter().rev().skip(1).flat_map(move |product| {
            Some(ProverQueryGeneral {
                point: x_last,
                rotation: Rotation(-((blinding_factors + 1) as i32)),
                poly: &product,
            })
        }))
}

pub fn lookup_open<'a, C: CurveAffine>(
    pk: &'a ProvingKey<C>,
    lookup: (&'a [C::Scalar], &'a [C::Scalar], &'a [C::Scalar]),
    x: C::Scalar,
) -> impl Iterator<Item = ProverQueryGeneral<'a, C>> + Clone {
    let x_inv = pk.vk.domain.rotate_omega(x, Rotation::prev());
    let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());

    let (permuted_input, permuted_table, z) = lookup;

    iter::empty()
        // Open lookup product commitments at x
        .chain(Some(ProverQueryGeneral {
            point: x,
            rotation: Rotation::cur(),
            poly: z,
        }))
        // Open lookup input commitments at x
        .chain(Some(ProverQueryGeneral {
            point: x,
            rotation: Rotation::cur(),
            poly: permuted_input,
        }))
        // Open lookup table commitments at x
        .chain(Some(ProverQueryGeneral {
            point: x,
            rotation: Rotation::cur(),
            poly: permuted_table,
        }))
        // Open lookup input commitments at x_inv
        .chain(Some(ProverQueryGeneral {
            point: x_inv,
            rotation: Rotation::prev(),
            poly: permuted_input,
        }))
        // Open lookup product commitments at x_next
        .chain(Some(ProverQueryGeneral {
            point: x_next,
            rotation: Rotation::next(),
            poly: z,
        }))
}
