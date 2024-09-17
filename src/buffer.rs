use ark_std::end_timer;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::ProvingKey;
use rand::rngs::OsRng;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator as _;
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

pub(crate) fn prepare_lookup_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> DeviceResult<
    Vec<(
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
        Vec<C::Scalar, HugePageAllocator>,
    )>,
> {
    let size = 1 << pk.get_vk().domain.k();
    let timer = start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
    let lookups = pk
        .vk
        .cs
        .lookups
        .par_iter()
        .map(|_| {
            let mut input = Vec::new_in(HugePageAllocator);
            input.resize(size, C::Scalar::zero());
            let mut table = Vec::new_in(HugePageAllocator);
            table.resize(size, C::Scalar::zero());
            let mut permuted_input = Vec::new_in(HugePageAllocator);
            permuted_input.resize(size, C::Scalar::zero());
            let mut permuted_table = Vec::new_in(HugePageAllocator);
            permuted_table.resize(size, C::Scalar::zero());
            let mut z = Vec::new_in(HugePageAllocator);
            z.resize(size, C::Scalar::zero());

            (input, table, permuted_input, permuted_table, z)
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
