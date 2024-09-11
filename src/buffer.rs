use ark_std::end_timer;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use rayon::iter::*;
use rayon::prelude::*;

use crate::device::cuda::CudaDevice;
use crate::device::Device;
use crate::hugetlb::PinnedPageAllocator;
use crate::Error;
use crate::ProvingKey;

pub(crate) fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Vec<Vec<C::Scalar, PinnedPageAllocator>> {
    let rows = 1 << pk.get_vk().domain.k();
    let columns = pk.get_vk().cs.num_advice_columns;
    let zero = C::Scalar::zero();
    let advices = (0..columns)
        .into_par_iter()
        .map(|_| {
            let mut buf = Vec::new_in(PinnedPageAllocator);
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

pub(crate) fn prepare_lookup_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Result<
    Vec<(
        Vec<C::Scalar, PinnedPageAllocator>,
        Vec<C::Scalar, PinnedPageAllocator>,
        Vec<C::Scalar, PinnedPageAllocator>,
        Vec<C::Scalar, PinnedPageAllocator>,
        Vec<C::Scalar, PinnedPageAllocator>,
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
        .map(|_| {
            let mut input = Vec::new_in(PinnedPageAllocator);
            input.resize(size, C::Scalar::zero());
            let mut table = Vec::new_in(PinnedPageAllocator);
            table.resize(size, C::Scalar::zero());
            let mut permuted_input = Vec::new_in(PinnedPageAllocator);
            permuted_input.resize(size, C::Scalar::zero());
            let mut permuted_table = Vec::new_in(PinnedPageAllocator);
            permuted_table.resize(size, C::Scalar::zero());
            let mut z = Vec::new_in(PinnedPageAllocator);
            z.resize(size, C::Scalar::zero());

            if false {
                let device = CudaDevice::get_device(0).unwrap();
                device.pin_memory(&permuted_input[..]).unwrap();
                device.pin_memory(&permuted_table[..]).unwrap();
                device.pin_memory(&z[..]).unwrap();
            }

            (input, table, permuted_input, permuted_table, z)
        })
        .collect::<Vec<_>>();
    end_timer!(timer);
    Ok(lookups)
}

pub(crate) fn prepare_permutation_buffers<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Result<Vec<Vec<C::Scalar, PinnedPageAllocator>>, Error> {
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
            let mut z = Vec::new_in(PinnedPageAllocator);
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
) -> Result<Vec<Vec<C::Scalar, PinnedPageAllocator>>, Error> {
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
            let mut z = Vec::new_in(PinnedPageAllocator);
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
