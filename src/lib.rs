#![feature(allocator_api)]

use ark_std::end_timer;
use ark_std::start_timer;
use device::cuda::CudaDevice;
use device::Device;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;
use std::alloc::Allocator;
use std::mem::size_of;

use crate::cuda_allocator::CudaHostAllocator;

pub mod device;
pub mod cuda_allocator;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref CUDA_ALLOCATOR: CudaHostAllocator = CudaHostAllocator::new(0);
}

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Vec<Vec<C::Scalar, &CudaHostAllocator>> {
    let mut advices = vec![];
    let size = 1 << pk.get_vk().domain.k();
    for _ in 0..pk.get_vk().cs.num_advice_columns {
        let mut buf: Vec<_, &CudaHostAllocator> = Vec::new_in(&CUDA_ALLOCATOR);
        buf.reserve_exact(size);
        // set len
        unsafe {
            let end = size_of::<Vec<C::Scalar, &CudaHostAllocator>>() / size_of::<usize>();
            *(&mut buf as *mut _ as *mut usize).offset(end as isize - 1) = size;
        }
        println!("buf len is {}", buf.len());
        advices.push(buf)
    }
    advices
}

pub enum Error {}

#[test]
fn test_cuda() {
    let device = CudaDevice::get_device(0).unwrap();
    let a = vec![1, 2, 3];
    let mut b = vec![0, 0, 0];
    let buf = device.alloc_device_buffer_from_slice(&a[..]).unwrap();
    device.copy_from_device_to_host(&mut b, &buf).unwrap();

    assert_eq!(a, b);
}

pub fn create_proof_from_advices<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
    A: Allocator,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[C::Scalar]],
    advices: Vec<Vec<C::Scalar, A>>,
    transcript: &mut T,
) -> Result<(), Error> {
    let size = 1 << pk.get_vk().domain.k();
 
    let timer = start_timer!(|| "copy params to gpu");
    end_timer!(timer);

    unimplemented!();

    /*
    let mut cfg = get_default_msm_config::<CurveCfg>();
    let timer = start_timer!(|| "copy advices to gpu");
    let advices_device_buf = advices
        .iter()
        .map(|x| -> CudaResult<_> {
            let mut device_buf = HostOrDeviceSlice::cuda_malloc(size)?;
            let buf = unsafe { std::mem::transmute::<&[_], &[_]>(&x[..]) };
            device_buf.copy_from_host(buf)?;
            Ok(device_buf)
        })
        .collect::<CudaResult<Vec<_>>>()
        .map_err(|e| IcicleError::from_cuda_error(e))?;
    end_timer!(timer);

    let mut affine = G1Affine::zero();
    for i in 0..10 { //advices_device_buf.len() {
        let timer = start_timer!(|| format!("single advice msm {}", i));
        let mut msm_results = HostOrDeviceSlice::Host(vec![G1Projective::zero()]);
        msm(
            &advices_device_buf[i],
            &g_lagrange_device_buf,
            &mut cfg,
            &mut msm_results,
        )?;
        end_timer!(timer);
        
        CurveCfg::to_affine(&msm_results.as_slice()[0], &mut affine);
        println!(
            "single advice msm {} result is {:?}",
            i,
            affine
        );
    }
 */
    Ok(())
}
