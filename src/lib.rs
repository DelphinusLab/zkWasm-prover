#![feature(allocator_api)]

use ark_std::end_timer;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::pairing::group::Group as _;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use std::alloc::Allocator;

use crate::cuda::bn254::msm;
use crate::device::cuda::CudaDevice;
use crate::device::Device as _;
use crate::device::DeviceResult;

pub mod cuda;
pub mod device;
mod hugetlb;

#[macro_use]
extern crate lazy_static;

pub fn prepare_advice_buffer<C: CurveAffine>(pk: &ProvingKey<C>) -> Vec<Vec<C::Scalar>> {
    let rows = 1 << pk.get_vk().domain.k();
    let columns = pk.get_vk().cs.num_advice_columns;
    let zero = C::Scalar::zero();
    let mut advices = vec![];
    for _ in 0..columns {
        advices.push(vec![zero; rows]);
    }
    advices
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

pub fn create_proof_from_advices<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    mut advices: Vec<&mut [C::Scalar]>,
    transcript: &mut T,
) -> Result<(), Error> {
    let size = 1 << pk.get_vk().domain.k();

    let timer = start_timer!(|| "create single instances");
    let instance = halo2_proofs::plonk::create_single_instances(params, pk, &[instances], transcript).unwrap();
    end_timer!(timer);

    let device = CudaDevice::get_device(0).unwrap();

    let timer = start_timer!(|| "pin advice memory to gpu");
    advices
        .iter_mut()
        .map(|x| -> Result<(), Error> {
            device.pin_memory(*x)?;
            Ok(())
        })
        .collect::<Result<_, _>>()?;
    end_timer!(timer);

    // TODO add random value
    if false {
        unimplemented!();
    }

    let timer = start_timer!(|| "copy advices to gpu");
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

    let timer = start_timer!(|| format!("advices msm, count {}", advices.len()));
    let msm_result = [C::Curve::identity()];
    let msm_result_buf = device.alloc_device_buffer_from_slice(&msm_result[..])?;
    for s_buf in advices_device_buf.iter() {
        msm(&device, &msm_result_buf, &g_lagrange_buf, s_buf, size)?;
    }
    end_timer!(timer);

    Ok(())
}
