use std::collections::BTreeMap;

use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;

use halo2_proofs::plonk::ProvingKey;
use rayon::iter::*;
use rayon::prelude::*;

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;

use crate::cuda::ntt::generate_ntt_buffers;
use crate::cuda::ntt::ntt_sync;
use crate::device::cuda::CudaDevice;
use crate::device::Device;
use crate::device::DeviceResult;
use crate::fill_random;
use crate::hugetlb::HugePageAllocator;
use log::debug;

#[allow(dead_code)]
pub(crate) fn lookup_compute_multiplicity<F: FieldExt>(
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

#[allow(dead_code)]
pub fn lookup_z_verifiy<C: CurveAffine>(
    device: &CudaDevice,
    pk: &ProvingKey<C>,
    lookups: &[(
        Vec<Vec<Vec<C::Scalar, HugePageAllocator>>>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<Vec<C::Scalar, HugePageAllocator>>,
    )],
    beta: &C::Scalar,
    unusable_rows_start: usize,
) -> DeviceResult<()> {
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << k;
    let (ntt_omegas_buf, ntt_pq_buf) =
        generate_ntt_buffers(&device, pk.get_vk().domain.get_omega(), k)?;
    // While in Lagrange basis, check that grand sum is correctly constructed
    /*
         sets[0] includes inputs, m and table
         φ_i(X) = f_i(X) + α
         τ(X) = t(X) + α
         LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
         RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))

         extended sets include only inputs, without m and table :
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

        input_set_sums[0]
            .par_iter_mut()
            .zip(table.par_iter())
            .zip(m_lagrange.par_iter())
            .for_each(|((input, table), m)| {
                *input = *input - *m * &(*table + beta).invert().unwrap();
            });

        //verify m correct
        let sum = (0..u)
            .into_par_iter()
            .map(|i| {
                input_set_sums
                    .iter()
                    .fold(C::Scalar::zero(), |acc, set| acc + set[i])
            })
            .reduce(|| C::Scalar::zero(), |acc, v| acc + v);
        assert_eq!(sum, C::Scalar::zero());

        //verify z correct
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
    Ok(())
}
