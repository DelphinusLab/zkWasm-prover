#![feature(allocator_api)]

use ark_std::end_timer;
use ark_std::rand::rngs::OsRng;
use ark_std::start_timer;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::pairing::group::Group as _;
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

use crate::cuda::bn254::msm;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBuf;
use crate::device::Device as _;
use crate::device::DeviceResult;

pub mod cuda;
pub mod device;
mod hugetlb;

#[macro_use]
extern crate lazy_static;

pub fn prepare_advice_buffer<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> Vec<Vec<C::Scalar, &HugePageAllocator>> {
    let rows = 1 << pk.get_vk().domain.k();
    let columns = pk.get_vk().cs.num_advice_columns;
    let zero = C::Scalar::zero();
    (0..columns)
        .into_par_iter()
        .map(|_| {
            let mut buf = Vec::new_in(&HugePageAllocator);
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
    let meta = &pk.vk.cs;
    let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

    let timer = start_timer!(|| "create single instances");
    let instance =
        halo2_proofs::plonk::create_single_instances(params, pk, &[instances], transcript).unwrap();
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

    // add random value
    if true {
        let named = &pk.vk.cs.named_advices;
        advices.par_iter_mut().enumerate().for_each(|(i, advice)| {
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

    let timer =
        start_timer!(|| format!("copy fixed columns to gpu, count {}", pk.fixed_values.len()));
    let fixed_device_buf = pk
        .fixed_values
        .iter()
        .map(|x| device.alloc_device_buffer_from_slice(x))
        .collect::<DeviceResult<Vec<_>>>()?;
    end_timer!(timer);

    let timer = start_timer!(|| "copy g_lagrange buffer");
    let g_lagrange_buf = device
        .alloc_device_buffer_from_slice(&params.g_lagrange[..])
        .unwrap();
    end_timer!(timer);

    let timer = start_timer!(|| "advices msm");
    let msm_result = [C::Curve::identity()];
    let msm_result_buf = device.alloc_device_buffer_from_slice(&msm_result[..])?;
    for s_buf in advices_device_buf.iter() {
        let commitment = msm(&device, &msm_result_buf, &g_lagrange_buf, s_buf, size)?;
        transcript.write_point(commitment).unwrap();
    }
    end_timer!(timer);

    let theta: C::ScalarExt = *transcript.squeeze_challenge_scalar::<()>();
    println!("theta is {:?}", theta);

    let timer = start_timer!(|| format!("prepare lookup buffer, count {}", pk.vk.cs.lookups.len()));
    let mut lookups = pk
        .vk
        .cs
        .lookups
        .par_iter()
        .map(|_| {
            let mut permuted_input = Vec::new_in(&HugePageAllocator);
            permuted_input.resize(size, C::ScalarExt::zero());
            let mut permuted_table = Vec::new_in(&HugePageAllocator);
            permuted_table.resize(size, C::ScalarExt::zero());
            let mut product = Vec::new_in(&HugePageAllocator);
            product.resize(size, C::ScalarExt::zero());
            (permuted_input, permuted_table, product)
        })
        .collect::<Vec<_>>();
    end_timer!(timer);

    let is_pure = |x: &Expression<_>| {
        x.is_constant().is_some()
            || x.is_pure_fixed().is_some()
            || x.is_pure_advice().is_some()
            || x.is_pure_instance().is_some()
    };

    let mut single_unit_lookups = pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookups.iter_mut())
        .filter(|(l, _)| {
            l.input_expressions.len() == 1
                && l.table_expressions.len() == 1
                && is_pure(&l.input_expressions[0])
                && is_pure(&l.table_expressions[0])
        })
        .collect::<Vec<_>>();

    let timer = start_timer!(|| format!("permute lookup pure {}", single_unit_lookups.len()));
    single_unit_lookups.par_iter_mut().for_each(|(l, lookup)| {
        let f = |expr: &Expression<_>, target: &mut [_]| {
            if let Some(v) = expr.is_constant() {
                target.fill(v);
            } else if let Some(idx) = expr.is_pure_fixed() {
                target.clone_from_slice(&pk.fixed_values[idx].values[0..unusable_rows_start]);
            }
        };

        if true {
            for cell in &mut lookup.0[unusable_rows_start..] {
                *cell = C::Scalar::random(&mut OsRng);
            }
            for cell in &mut lookup.1[unusable_rows_start..] {
                *cell = C::Scalar::random(&mut OsRng);
            }
        }

        f(
            &l.input_expressions[0],
            &mut lookup.0[0..unusable_rows_start],
        );
        f(
            &l.table_expressions[0],
            &mut lookup.1[0..unusable_rows_start],
        );
    });

    single_unit_lookups.par_iter_mut().for_each(|(_, lookup)| {
        lookup.0[0..unusable_rows_start].sort_unstable_by(|a, b| unsafe {
            let a: &[u64; 4] = std::mem::transmute(a);
            let b: &[u64; 4] = std::mem::transmute(b);
            a.cmp(b)
        });
    });
    end_timer!(timer);

    let timer = start_timer!(|| format!("permute lookup pure {}", single_unit_lookups.len()));
    let mut single_expr_lookups = pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookups.iter_mut())
        .filter(|(l, _)| {
            l.input_expressions.len() == 1
                && l.table_expressions.len() == 1
                && !is_pure(&l.input_expressions[0])
                && !is_pure(&l.table_expressions[0])
        })
        .collect::<Vec<_>>();

    let mut tmp_buf = vec![];

    let eval_expr = |device, tmp_buf: &mut Vec<CudaDeviceBuf<C>>, expr: &Expression<_>| -> CudaDeviceBuf<_> {
        let res_buf = tmp_buf.pop().unwrap();
        match expr {
            Expression::Constant(_) => todo!(),
            Expression::Fixed { query_index, column_index, rotation } => todo!(),
            Expression::Advice { query_index, column_index, rotation } => todo!(),
            Expression::Instance { query_index, column_index, rotation } => todo!(),
            Expression::Negated(_) => todo!(),
            Expression::Sum(_, _) => todo!(),
            Expression::Product(_, _) => todo!(),
            Expression::Scaled(_, _) => todo!(),
            Expression::Selector(_) => unreachable!(),
        }
        res_buf
    };

    single_expr_lookups.iter_mut().for_each(|(l, lookup)| {
        if true {
            for cell in &mut lookup.0[unusable_rows_start..] {
                *cell = C::Scalar::random(&mut OsRng);
            }
            for cell in &mut lookup.1[unusable_rows_start..] {
                *cell = C::Scalar::random(&mut OsRng);
            }
        }

        let buf1 = eval_expr(
            &device,
            &mut tmp_buf,
            &l.input_expressions[0],
        );

        let buf = eval_expr(
            &device,
            &mut tmp_buf,
            &l.table_expressions[0],
        );
    });
    end_timer!(timer);

    Ok(())
}
