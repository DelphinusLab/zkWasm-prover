use ark_std::end_timer;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::ProvingKey;
use rand::rngs::OsRng;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator as _;
use rayon::prelude::ParallelSliceMut as _;
use rayon::slice::ParallelSlice as _;

use crate::device::DeviceResult;
use crate::hugetlb::HugePageAllocator;
use crate::ADD_RANDOM;

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
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
    advices
}

pub fn prepare_fixed_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> [Vec<Vec<C::Scalar, HugePageAllocator>>; 2] {
    [
        pk.fixed_values
            .par_iter()
            .map(|x| {
                let mut buf = Vec::new_in(HugePageAllocator);
                buf.extend_from_slice(&x[..]);
                buf
            })
            .collect::<Vec<_>>(),
        pk.fixed_polys
            .par_iter()
            .map(|x| {
                let mut buf = Vec::new_in(HugePageAllocator);
                buf.extend_from_slice(&x[..]);
                buf
            })
            .collect::<Vec<_>>(),
    ]
}

pub fn prepare_permutation_poly_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Vec<Vec<C::Scalar, HugePageAllocator>> {
    pk.permutation
        .polys
        .iter()
        .map(|x| {
            let mut buf = Vec::new_in(HugePageAllocator);
            buf.extend_from_slice(&x[..]);
            buf
        })
        .collect::<Vec<_>>()
}

pub(crate) fn prepare_lookup_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> DeviceResult<
    Vec<(
        Vec<Vec<Vec<C::Scalar, HugePageAllocator>>>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<Vec<C::Scalar, HugePageAllocator>>,
    )>,
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
            // multiplicity need init
            let mut multiplicity = Vec::new_in(HugePageAllocator);
            multiplicity.resize(size, C::Scalar::zero());

            let mut inputs_sets = argument
                .input_expressions_sets
                .iter()
                .map(|set| {
                    set.0
                        .iter()
                        .map(|_| Vec::new_in(HugePageAllocator))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut z_set: Vec<_> = (0..argument.input_expressions_sets.len())
                .map(|_| Vec::new_in(HugePageAllocator))
                .collect();

            for buf in std::iter::empty()
                .chain(Some(&mut table))
                .chain(inputs_sets.iter_mut().flat_map(|set| set.iter_mut()))
                .chain(z_set.iter_mut())
            {
                buf.reserve(size);
                unsafe {
                    buf.set_len(size);
                }
            }

            (inputs_sets, table, multiplicity, z_set)
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(lookups)
}

pub(crate) fn prepare_permutation_buffers<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> DeviceResult<Vec<Vec<C::Scalar, HugePageAllocator>>> {
    let size = 1 << pk.get_vk().domain.k();
    let chunk_len = &pk.vk.cs.degree() - 2;
    let timer = start_timer!(|| format!(
        "prepare permutation buffer, count {}",
        pk.vk.cs.permutation.columns.chunks(chunk_len).len()
    ));
    let buffers = pk
        .vk
        .cs
        .permutation
        .columns
        .par_chunks(chunk_len)
        .map(|_| {
            let mut z = Vec::new_in(HugePageAllocator);
            z.reserve(size);
            unsafe {
                z.set_len(size);
            }
            z.par_chunks_mut(size / 2)
                .for_each(|c| c.fill(C::Scalar::one()));
            z
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(buffers)
}

pub(crate) fn prepare_shuffle_buffers<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> DeviceResult<Vec<Vec<C::Scalar, HugePageAllocator>>> {
    let size = 1 << pk.get_vk().domain.k();
    let timer =
        start_timer!(|| format!("prepare shuffle buffer, count {}", pk.vk.cs.shuffles.len()));
    let buffers = pk
        .vk
        .cs
        .shuffles
        .par_iter()
        .map(|_| {
            let mut z = Vec::new_in(HugePageAllocator);
            z.reserve(size);
            unsafe {
                z.set_len(size);
            }
            z.par_chunks_mut(size / 2)
                .for_each(|c| c.fill(C::Scalar::one()));
            z
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(buffers)
}

pub(crate) fn generate_random_poly<F: FieldExt>(size: usize) -> Vec<F, HugePageAllocator> {
    use rand::thread_rng;
    use rand::RngCore;

    let random_nr = 32;
    let mut random_poly = Vec::new_in(HugePageAllocator);
    random_poly.resize(size, F::zero());

    let random = vec![0; 32usize]
        .iter()
        .map(|_| F::random(&mut OsRng))
        .collect::<Vec<_>>();

    random_poly.par_iter_mut().for_each(|coeff| {
        if ADD_RANDOM {
            let mut rng = thread_rng();
            *coeff = (F::random(&mut rng) + random[rng.next_u64() as usize % random_nr])
                * (F::random(&mut rng) + random[rng.next_u64() as usize % random_nr])
        }
    });

    random_poly
}
