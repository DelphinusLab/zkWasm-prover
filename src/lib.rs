#![feature(allocator_api)]
#![feature(get_mut_unchecked)]

use std::rc::Rc;
use std::sync::Arc;
use std::thread;

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use cache::Cache;
use cache::CacheAction;
use cuda::bn254::buffer_copy_with_shift;
use cuda::bn254::extended_prepare;
use cuda::bn254::field_op;
use cuda::bn254::FieldOp;
use device::cuda::CudaDeviceBufRaw;
use halo2_proofs::arithmetic::BaseExt as _;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::pairing::group::ff::BatchInvert as _;
use halo2_proofs::pairing::group::Group as _;
use halo2_proofs::plonk::evaluation_gpu::Bop;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::lookup;
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

use crate::cuda::bn254::field_op_v2;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::msm;
use crate::cuda::bn254::msm_with_groups;
use crate::cuda::bn254::ntt;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254::ntt_raw;
use crate::cuda::bn254::permutation_eval_h_l;
use crate::cuda::bn254::permutation_eval_h_p1;
use crate::cuda::bn254::permutation_eval_h_p2;
use crate::cuda::bn254::permutation_eval_h_r;
use crate::device::cuda::CudaDevice;
use crate::device::Device as _;
use crate::device::DeviceResult;

mod cache;
pub mod cuda;
pub mod device;
mod hugetlb;

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
        for cell in &mut input[unusable_rows_start..] {
            *cell = F::random(&mut OsRng);
        }
        for cell in &mut table[unusable_rows_start..] {
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
                    handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
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
                    &pk.vk.cs.lookups[i].input_expressions[0],
                    &mut input[0..unusable_rows_start],
                );
                f(
                    &pk.vk.cs.lookups[i].table_expressions[0],
                    &mut table[0..unusable_rows_start],
                );
                let (permuted_input, permuted_table) =
                    handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
                (i, (permuted_input, permuted_table, input, table, z))
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        return (single_unit_lookups, single_comp_lookups, tuple_lookups);
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
                        rot_scale as i32,
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
                    handle_lookup_pair(&mut input, &mut table, unusable_rows_start);
                (i, (permuted_input, permuted_table, input, table, z))
            })
            .collect::<Vec<_>>();
        //end_timer!(timer);

        tuple_lookups
    });

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
        lookup_permuted_commitments[i * 2] =
            msm(&device, &g_lagrange_buf, &permuted_input_buf, size)?;
        lookup_permuted_commitments[i * 2 + 1] =
            msm(&device, &g_lagrange_buf, &permuted_table_buf, size)?;
    }

    for (i, permuted_input_buf, permuted_table_buf) in single_comp_buffers.iter() {
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

    for commitment in lookup_permuted_commitments {
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
        .for_each(|(_, (permuted_input, permuted_table, z, input, table))| {
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
            for i in 0..unusable_rows_start {
                std::mem::swap(&mut tmp, &mut z[i]);
                tmp = tmp * z[i];
            }

            if true {
                for cell in &mut z[unusable_rows_start..] {
                    *cell = C::Scalar::random(&mut OsRng);
                }
            }
        });

    let mut lookups = lookups
        .into_iter()
        .map(|(_, (permuted_input, permuted_table, z, _, _))| (permuted_input, permuted_table, z))
        .collect::<Vec<_>>();
    end_timer!(timer);

    let timer = start_timer!(|| "prepare ntt");
    let (omegas_buf, pq_buf) = ntt_prepare::<C::ScalarExt>(&device, k)?;
    let divisor_buf = device.alloc_device_buffer_from_slice(&[pk.get_vk().domain.ifft_divisor])?;
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
        (&pk)
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

                let mut tmp = C::ScalarExt::one();
                for i in 0..unusable_rows_start {
                    std::mem::swap(&mut tmp, &mut modified_values[i]);
                    tmp = tmp * modified_values[i];
                }

                modified_values
            })
            .collect::<Vec<_>>()
    });
    end_timer!(timer);

    let timer = start_timer!(|| "lookup intt and z msm");
    let mut tmp_buf = device.alloc_device_buffer::<C::ScalarExt>(size)?;
    let mut ntt_buf = device.alloc_device_buffer::<C::ScalarExt>(size)?;
    for (permuted_input, permuted_table, z) in lookups.iter_mut() {
        device.copy_from_host_to_device(&ntt_buf, &z[..])?;
        let commitment = msm_with_groups(&device, &g_lagrange_buf, &ntt_buf, size, 1)?;
        transcript.write_point(commitment).unwrap();
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &pq_buf,
            &omegas_buf,
            &divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut z[..], &ntt_buf)?;

        device.copy_from_host_to_device(&ntt_buf, &permuted_input[..])?;
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &pq_buf,
            &omegas_buf,
            &divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut permuted_input[..], &ntt_buf)?;

        device.copy_from_host_to_device(&ntt_buf, &permuted_table[..])?;
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &pq_buf,
            &omegas_buf,
            &divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut permuted_table[..], &ntt_buf)?;
    }
    end_timer!(timer);

    let timer = start_timer!(|| "wait permutation_products");
    let mut permutation_products = permutation_products_handler.join().unwrap();
    end_timer!(timer);

    let timer = start_timer!(|| "permutation z msm and ntt");
    for z in permutation_products.iter_mut() {
        device.copy_from_host_to_device(&ntt_buf, &z[..])?;
        let commitment = msm_with_groups(&device, &g_lagrange_buf, &ntt_buf, size, 1)?;
        transcript.write_point(commitment).unwrap();
        intt_raw(
            &device,
            &mut ntt_buf,
            &mut tmp_buf,
            &pq_buf,
            &omegas_buf,
            &divisor_buf,
            k,
        )?;
        device.copy_from_device_to_host(&mut z[..], &ntt_buf)?;
    }
    end_timer!(timer);
    let vanishing =
        halo2_proofs::plonk::vanishing::Argument::commit(params, &pk.vk.domain, OsRng, transcript)
            .unwrap();

    let y: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("y is {:?}", y);

    let timer = start_timer!(|| "h_poly");
    let fixed_ref = &pk.fixed_values.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let advice_ref = &advices.iter().map(|x| &x[..]).collect::<Vec<_>>()[..];
    let instance_ref = &instance[0]
        .instance_values
        .iter()
        .map(|x| &x[..])
        .collect::<Vec<_>>()[..];
    let h_poly = evaluate_h_gates(
        &device,
        &pk,
        fixed_ref,
        advice_ref,
        instance_ref,
        &permutation_products
            .iter()
            .map(|x| &x[..])
            .collect::<Vec<_>>()[..],
        y,
        beta,
        gamma,
        theta,
    )?;
    end_timer!(timer);

    Ok(())
}

struct EvalHContext<F: FieldExt> {
    y: Vec<F>,
    allocator: Vec<Rc<CudaDeviceBufRaw>>,
    extended_allocator: Vec<Rc<CudaDeviceBufRaw>>,
    extended_k: usize,
    size: usize,
    extended_size: usize,
    extended_omegas_buf: CudaDeviceBufRaw,
    extended_pq_buf: CudaDeviceBufRaw,
    coset_powers_buf: CudaDeviceBufRaw,
}

fn evaluate_h_gates<C: CurveAffine>(
    device: &CudaDevice,
    pk: &ProvingKey<C>,
    fixed: &[&[C::ScalarExt]],
    advice: &[&[C::ScalarExt]],
    instance: &[&[C::ScalarExt]],
    permutation_products: &[&[C::ScalarExt]],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
) -> DeviceResult<Vec<C::ScalarExt, HugePageAllocator>> {
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
    let extended_k = pk.get_vk().domain.extended_k() as usize;
    let extended_size = 1 << extended_k;
    let extended_omega = pk.vk.domain.get_extended_omega();

    let (extended_omegas_buf, extended_pq_buf) = ntt_prepare::<C::ScalarExt>(device, extended_k)?;
    let coset_powers_buf = device.alloc_device_buffer_from_slice(&[
        pk.get_vk().domain.g_coset,
        pk.get_vk().domain.g_coset_inv,
    ])?;
    let mut ctx = EvalHContext {
        y: vec![C::ScalarExt::one(), y],
        allocator: vec![],
        extended_allocator: vec![],
        extended_k,
        size,
        extended_size,
        extended_omegas_buf,
        extended_pq_buf,
        coset_powers_buf,
    };

    let mut res = Vec::new_in(HugePageAllocator);
    res.resize(extended_size, C::ScalarExt::zero());

    let timer = start_timer!(|| "prepare buffer");
    let fixed_buf = fixed
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<Result<Vec<_>, _>>()?;
    let advice_buf = advice
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<Result<Vec<_>, _>>()?;
    let instance_buf = instance
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<Result<Vec<_>, _>>()?;
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h gates");
    let ((buf, c), _) = evaluate_prove_expr(
        device,
        &pk.ev.gpu_gates_expr[0],
        &fixed_buf[..],
        &advice_buf[..],
        &instance_buf[..],
        &mut ctx,
    )?;
    let h_buf = buf.unwrap();
    println!("reach here 1");
    assert!(c.is_none());
    println!("reach here 2");
    //device.copy_from_device_to_host(&mut res[..], buf.as_ref().unwrap())?;
    end_timer!(timer);

    println!("reach here");
    //analysis(&pk.ev.gpu_gates_expr[0]);

    let y_buf = device.alloc_device_buffer_from_slice(&[y][..])?;
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..])?;
    let gamma_buf = device.alloc_device_buffer_from_slice(&[gamma][..])?;
    let theta_buf = device.alloc_device_buffer_from_slice(&[theta][..])?;

    let timer = start_timer!(|| "evaluate_h permutation");
    if permutation_products.len() > 0 {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let last_rotation = (ctx.size - (blinding_factors + 1)) << (extended_k - k);
        let chunk_len = pk.vk.cs.degree() - 2;

        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;

        let l0_buf = do_extended_fft_v2(device, &mut ctx, &l0.values[..])?;
        let l_last_buf = do_extended_fft_v2(device, &mut ctx, &l_last.values[..])?;
        let l_active_buf = device.alloc_device_buffer_from_slice(&l_active_row.values[..])?;

        let extended_p_buf = permutation_products
            .iter()
            .map(|x| do_extended_fft_v2(device, &mut ctx, x))
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
                .iter()
                .zip(pk.vk.cs.permutation.columns.chunks(chunk_len))
                .zip(pk.permutation.polys.chunks(chunk_len))
            {
                let buf = ctx.extended_allocator.pop();
                let l = if buf.is_none() {
                    device.alloc_device_buffer::<C::ScalarExt>(ctx.extended_size)?
                } else {
                    Rc::try_unwrap(buf.unwrap()).unwrap()
                };
                buffer_copy_with_shift::<C::ScalarExt>(
                    &device,
                    &l,
                    extended_p_buf,
                    1 << (extended_k - k),
                    ctx.extended_size,
                )?;

                let buf = ctx.extended_allocator.pop();
                let r = if buf.is_none() {
                    device.alloc_device_buffer::<C::ScalarExt>(ctx.extended_size)?
                } else {
                    Rc::try_unwrap(buf.unwrap()).unwrap()
                };
                buffer_copy_with_shift::<C::ScalarExt>(
                    &device,
                    &l,
                    extended_p_buf,
                    0,
                    ctx.extended_size,
                )?;

                for (value_buf, permutation) in columns
                    .iter()
                    .map(|&column| match column.column_type() {
                        Any::Advice => &advice_buf[column.index()],
                        Any::Fixed => &fixed_buf[column.index()],
                        Any::Instance => &instance_buf[column.index()],
                    })
                    .zip(polys.iter())
                {
                    let buf = ctx.extended_allocator.pop();
                    let tmp = if buf.is_none() {
                        device.alloc_device_buffer::<C::ScalarExt>(ctx.extended_size)?
                    } else {
                        Rc::try_unwrap(buf.unwrap()).unwrap()
                    };

                    let p_coset_buf =
                        device.alloc_device_buffer_from_slice(&permutation.values[..])?;
                    permutation_eval_h_l(
                        &device,
                        &tmp,
                        &beta_buf,
                        &gamma_buf,
                        &p_coset_buf,
                        ctx.size,
                    )?;
                    do_extended_fft(&device, &mut ctx, &tmp)?;
                    field_op_v2::<C::ScalarExt>(
                        &device,
                        &l,
                        Some(&l),
                        None,
                        Some(&tmp),
                        None,
                        ctx.extended_size,
                        FieldOp::Mul,
                    )?;

                    let curr_delta_buf =
                        device.alloc_device_buffer_from_slice(&[curr_delta][..])?;
                    permutation_eval_h_r(
                        &device,
                        &tmp,
                        &curr_delta_buf,
                        &gamma_buf,
                        &value_buf,
                        ctx.size,
                    )?;
                    do_extended_fft(&device, &mut ctx, &tmp)?;
                    field_op_v2::<C::ScalarExt>(
                        &device,
                        &r,
                        Some(&r),
                        None,
                        Some(&tmp),
                        None,
                        ctx.extended_size,
                        FieldOp::Mul,
                    )?;

                    curr_delta *= &C::Scalar::DELTA;
                    field_op_v2::<C::ScalarExt>(
                        &device,
                        &l,
                        Some(&l),
                        None,
                        Some(&r),
                        None,
                        ctx.extended_size,
                        FieldOp::Sub,
                    )?;
                    field_op_v2::<C::ScalarExt>(
                        &device,
                        &l,
                        Some(&l),
                        None,
                        Some(&l_active_buf),
                        None,
                        ctx.extended_size,
                        FieldOp::Mul,
                    )?;
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
                    field_op_v2::<C::ScalarExt>(
                        &device,
                        &h_buf,
                        Some(&h_buf),
                        None,
                        Some(&l),
                        None,
                        ctx.extended_size,
                        FieldOp::Sum,
                    )?;
                }

                ctx.extended_allocator.push(Rc::new(l));
                ctx.extended_allocator.push(Rc::new(r));
            }
        }
    }
    end_timer!(timer);

    Ok(res)
}

fn do_extended_fft_v2<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<CudaDeviceBufRaw> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        Rc::try_unwrap(buf.unwrap()).unwrap()
    };
    let tmp = ctx.extended_allocator.pop();
    let mut tmp = if tmp.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        Rc::try_unwrap(tmp.unwrap()).unwrap()
    };
    device.copy_from_host_to_device::<F>(&buf, data)?;
    extended_prepare(
        device,
        &buf,
        &ctx.coset_powers_buf,
        2,
        ctx.size,
        ctx.extended_size,
    )?;
    ntt_raw(
        device,
        &mut buf,
        &mut tmp,
        &ctx.extended_pq_buf,
        &ctx.extended_omegas_buf,
        ctx.extended_k,
    )?;
    ctx.allocator.push(Rc::new(tmp));

    Ok(buf)
}

fn do_extended_fft<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &CudaDeviceBufRaw,
) -> DeviceResult<CudaDeviceBufRaw> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        Rc::try_unwrap(buf.unwrap()).unwrap()
    };
    let tmp = ctx.extended_allocator.pop();
    let mut tmp = if tmp.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        Rc::try_unwrap(tmp.unwrap()).unwrap()
    };
    device.copy_from_device_to_device::<F>(&buf, 0, data, 0, ctx.size)?;
    extended_prepare(
        device,
        &buf,
        &ctx.coset_powers_buf,
        2,
        ctx.size,
        ctx.extended_size,
    )?;
    ntt_raw(
        device,
        &mut buf,
        &mut tmp,
        &ctx.extended_pq_buf,
        &ctx.extended_omegas_buf,
        ctx.extended_k,
    )?;
    ctx.allocator.push(Rc::new(tmp));

    Ok(buf)
}

fn evaluate_prove_expr<F: FieldExt>(
    device: &CudaDevice,
    expr: &ProveExpression<F>,
    fixed_buf: &[CudaDeviceBufRaw],
    advice_buf: &[CudaDeviceBufRaw],
    instance_buf: &[CudaDeviceBufRaw],
    ctx: &mut EvalHContext<F>,
) -> DeviceResult<((Option<CudaDeviceBufRaw>, Option<F>), usize)> {
    match expr {
        ProveExpression::Unit(u) => {
            let (src, rotation) = match u {
                ProveExpressionUnit::Fixed {
                    column_index,
                    rotation,
                } => (&fixed_buf[*column_index], rotation),
                ProveExpressionUnit::Advice {
                    column_index,
                    rotation,
                } => (&advice_buf[*column_index], rotation),
                ProveExpressionUnit::Instance {
                    column_index,
                    rotation,
                } => (&instance_buf[*column_index], rotation),
            };

            let rot = rotation.0 as isize;

            let res = ctx.allocator.pop();
            let res = if res.is_none() {
                device.alloc_device_buffer::<F>(ctx.size)?
            } else {
                Rc::try_unwrap(res.unwrap()).unwrap()
            };
            device.synchronize()?;
            buffer_copy_with_shift::<F>(device, &res, src, rot, ctx.size)?;
            Ok(((Some(res), None), 1))
        }
        ProveExpression::Op(l, r, op) => {
            let ((mut l_buf, l_c), mut l_deg) =
                evaluate_prove_expr(device, l, fixed_buf, advice_buf, instance_buf, ctx)?;
            let ((mut r_buf, r_c), mut r_deg) =
                evaluate_prove_expr(device, r, fixed_buf, advice_buf, instance_buf, ctx)?;

            if l_deg != r_deg || *op == Bop::Product {
                if l_deg == 1 && l_buf.is_some() {
                    let l_extended_buf = do_extended_fft(device, ctx, l_buf.as_ref().unwrap())?;
                    ctx.allocator.push(Rc::new(l_buf.unwrap()));
                    l_buf = Some(l_extended_buf);
                    l_deg = 4;
                }
                if r_deg == 1 && r_buf.is_some() {
                    let r_extended_buf = do_extended_fft(device, ctx, l_buf.as_ref().unwrap())?;
                    ctx.allocator.push(Rc::new(r_buf.unwrap()));
                    r_buf = Some(r_extended_buf);
                    r_deg = 4;
                }
            }

            let res = if l_buf.is_some() {
                l_buf.as_ref().unwrap()
            } else {
                r_buf.as_ref().unwrap()
            };
            field_op::<F>(
                device,
                res,
                l_buf.as_ref(),
                0,
                l_c,
                r_buf.as_ref(),
                0,
                r_c,
                if l_deg == 1 {
                    ctx.size
                } else {
                    ctx.extended_size
                },
                if *op == Bop::Sum {
                    FieldOp::Sum
                } else {
                    FieldOp::Mul
                },
            )?;
            let (res, other) = if l_buf.is_some() {
                (l_buf.unwrap(), r_buf)
            } else {
                (r_buf.unwrap(), l_buf)
            };
            if other.is_some() {
                ctx.allocator.push(Rc::new(other.unwrap()));
            }
            Ok(((Some(res), None), l_deg))
        }
        ProveExpression::Y(ys) => {
            let max_y_order = *ys.keys().max().unwrap();
            for _ in (ctx.y.len() as u32)..=max_y_order {
                ctx.y.push(ctx.y[1] * ctx.y.last().unwrap());
            }
            let c = ys.iter().fold(F::zero(), |acc, (y_order, f)| {
                acc + ctx.y[*y_order as usize] * f
            });

            Ok(((None, Some(c)), 1))
        }
        ProveExpression::Scale(l, ys) => {
            let ((l_buf, l_c), l_deg) =
                evaluate_prove_expr(device, l, fixed_buf, advice_buf, instance_buf, ctx)?;
            assert!(l_c.is_none());
            let max_y_order = *ys.keys().max().unwrap();
            for _ in (ctx.y.len() as u32)..=max_y_order {
                ctx.y.push(ctx.y[1] * ctx.y.last().unwrap());
            }
            let c = ys.iter().fold(F::zero(), |acc, (y_order, f)| {
                acc + ctx.y[*y_order as usize] * f
            });
            field_op(
                device,
                l_buf.as_ref().unwrap(),
                l_buf.as_ref(),
                0,
                None,
                None,
                0,
                Some(c),
                if l_deg == 1 {
                    ctx.size
                } else {
                    ctx.extended_size
                },
                FieldOp::Mul,
            )?;
            Ok(((l_buf, None), l_deg))
        }
    }
}

fn analysis<F: FieldExt>(expr: &ProveExpression<F>) -> usize {
    match expr {
        ProveExpression::Unit(u) => {
            let rotation = match u {
                ProveExpressionUnit::Fixed {
                    column_index,
                    rotation,
                } => rotation,
                ProveExpressionUnit::Advice {
                    column_index,
                    rotation,
                } => rotation,
                ProveExpressionUnit::Instance {
                    column_index,
                    rotation,
                } => rotation,
            };
            println!("handle unit {:?}", rotation);
            return 1;
        }
        ProveExpression::Op(l, r, op) => {
            let l_dep = analysis(l);
            let r_dep = analysis(r);

            if l_dep != r_dep {
                println!(
                    "handle deep upgrade {} {}",
                    l_dep.min(r_dep),
                    l_dep.max(r_dep)
                );
            }

            match op {
                Bop::Sum => {
                    println!("handle sum {}", l_dep.max(r_dep));
                    return l_dep.max(r_dep);
                }
                Bop::Product => {
                    println!("handle mul {}", l_dep.max(r_dep));
                    if l_dep == 1 && r_dep == 1 {
                        println!("handle deep upgrade 2 from 1");
                        return 2;
                    } else {
                        if l_dep < 4 {
                            println!("handle deep upgrade 4 from {}", 1);
                        }
                        if r_dep < 4 {
                            println!("handle deep upgrade 4 from {}", 1);
                        }
                        return 4;
                    }
                }
            }
        }
        ProveExpression::Y(_) => {
            println!("handle y");
            return 1;
        }
        ProveExpression::Scale(l, _) => {
            let l_dep = analysis(l);
            println!("handle scale {}", l_dep);
            return 1;
        }
    }
}
