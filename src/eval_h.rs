use std::collections::BTreeMap;
use std::mem::ManuallyDrop;

use ark_std::end_timer;
use ark_std::iterable::Iterable;
use ark_std::start_timer;
use cuda_runtime_sys::cudaMemset;
use cuda_runtime_sys::cudaStream_t;
use cuda_runtime_sys::CUstream_st;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::circuit::Expression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;

use crate::analyze::analyze_expr_tree;
use crate::cuda::bn254::buffer_copy_with_shift;
use crate::cuda::bn254::extended_intt_after;
use crate::cuda::bn254::extended_prepare;
use crate::cuda::bn254::field_mul;
use crate::cuda::bn254::field_op_v2;
use crate::cuda::bn254::field_op_v3;
use crate::cuda::bn254::field_sub;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::intt_raw_async;
use crate::cuda::bn254::ntt_prepare;
use crate::cuda::bn254::ntt_raw;
use crate::cuda::bn254::permutation_eval_h_l;
use crate::cuda::bn254::permutation_eval_h_p1;
use crate::cuda::bn254::permutation_eval_h_p2;
use crate::cuda::bn254::pick_from_buf;
use crate::cuda::bn254::FieldOp;
use crate::cuda::bn254_c;
use crate::cuda::bn254_c::field_op_batch_mul_sum;
use crate::cuda::bn254_c::lookup_eval_h;
use crate::cuda::bn254_c::shuffle_eval_h;
use crate::cuda::bn254_c::shuffle_eval_h_v2;
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::cuda::CudaStreamWrapper;
use crate::device::Device as _;
use crate::device::DeviceResult;
use crate::expr::flatten_lookup_expression;
use crate::expr::flatten_shuffle_expression;
use crate::expr::is_expr_unit;
use crate::hugetlb::HugePageAllocator;

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
            device.alloc_device_buffer_non_zeroed::<F>(self.extended_size)
        } else {
            Ok(buf.unwrap())
        }
    }
}

pub fn _export_evaluate_h_gates<C: CurveAffine>(
    pk: &ProvingKey<C>,
    fixed: &[&[C::Scalar]],
    advice: &[&[C::Scalar]],
    instance: &[&[C::Scalar]],
    permutation_products: &[&[C::Scalar]],
    lookup_products: &mut [(
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
    )],
    shuffle_products: &[&[C::Scalar]],
    y: C::Scalar,
    beta: C::Scalar,
    gamma: C::Scalar,
    theta: C::Scalar,
    res: &mut [C::Scalar],
) {
    let device = CudaDevice::get_device(0).unwrap();
    let (intt_omegas_buf, intt_pq_buf) = ntt_prepare(
        &device,
        pk.get_vk().domain.get_omega_inv(),
        pk.vk.domain.k() as usize,
    )
    .unwrap();
    let intt_divisor_buf = device
        .alloc_device_buffer_from_slice::<C::Scalar>(&[pk.get_vk().domain.ifft_divisor])
        .unwrap();

    let (_, h_buf) = evaluate_h_gates_core(
        &device,
        pk,
        fixed,
        advice,
        instance,
        permutation_products,
        lookup_products,
        shuffle_products,
        y,
        beta,
        gamma,
        theta,
        intt_pq_buf,
        intt_omegas_buf,
        intt_divisor_buf,
    )
    .unwrap();

    device.copy_from_device_to_host(res, &h_buf).unwrap();
}

pub(crate) fn evaluate_h_gates_and_vanishing_construct<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    device: &CudaDevice,
    pk: &ProvingKey<C>,
    fixed: &[&[C::Scalar]],
    advice: &[&[C::Scalar]],
    instance: &[&[C::Scalar]],
    permutation_products: &[&[C::Scalar]],
    lookup_products: &mut [(
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
    )],
    shuffle_products: &[&[C::Scalar]],
    y: C::Scalar,
    beta: C::Scalar,
    gamma: C::Scalar,
    theta: C::Scalar,
    intt_pq_buf: CudaDeviceBufRaw,
    intt_omegas_buf: CudaDeviceBufRaw,
    intt_divisor_buf: CudaDeviceBufRaw,
    g_buf: &CudaDeviceBufRaw,
    transcript: &mut T,
) -> DeviceResult<(C::Scalar, C::Scalar, Vec<C::Scalar, HugePageAllocator>)> {
    let domain = &pk.vk.domain;
    let k = &pk.vk.domain.k();
    let size = 1 << k;

    let (mut ctx, mut h_buf) = evaluate_h_gates_core(
        &device,
        pk,
        fixed,
        advice,
        instance,
        permutation_products,
        lookup_products,
        shuffle_products,
        y,
        beta,
        gamma,
        theta,
        intt_pq_buf,
        intt_omegas_buf,
        intt_divisor_buf,
    )
    .unwrap();

    // do vanishing construct

    // divide zH
    {
        let t_evalutions_buf =
            device.alloc_device_buffer_from_slice::<C::Scalar>(&domain.t_evaluations[..])?;

        let err = unsafe {
            bn254_c::field_mul_zip(
                h_buf.ptr(),
                t_evalutions_buf.ptr(),
                domain.t_evaluations.len() as i32,
                domain.extended_len() as i32,
            )
        };

        to_result((), err, "failed to run field_mul_zip")?;
    }

    // intt
    {
        let intt_divisor_buf = device
            .alloc_device_buffer_from_slice::<C::Scalar>(&[domain.extended_ifft_divisor])
            .unwrap();

        let (extended_intt_omegas_buf, extended_intt_pq_buf) = ntt_prepare(
            &device,
            domain.extended_omega_inv,
            pk.vk.domain.extended_k() as usize,
        )?;
        let mut tmp = ctx.alloc(&device)?;

        intt_raw(
            &device,
            &mut h_buf,
            &mut tmp,
            &extended_intt_pq_buf,
            &extended_intt_omegas_buf,
            &intt_divisor_buf,
            pk.vk.domain.extended_k() as usize,
        )?;

        let coset_powers_buf =
            device.alloc_device_buffer_from_slice(&[domain.g_coset_inv, domain.g_coset])?;

        extended_intt_after(
            &device,
            &h_buf,
            &coset_powers_buf,
            3,
            ctx.size,
            ctx.extended_size,
            None,
        )?;

        if ctx.size >= 1 << 23 {
            ctx.extended_allocator.clear();
        }

        let timer = start_timer!(|| format!("vanishing msm {}", domain.quotient_poly_degree));
        let mut buffers = vec![];
        for i in 0..domain.quotient_poly_degree as usize {
            let s_buf = unsafe {
                ManuallyDrop::new(CudaDeviceBufRaw {
                    ptr: h_buf
                        .ptr()
                        .offset((i * size * core::mem::size_of::<C::Scalar>()) as isize),
                    device: device.clone(),
                    size: size * core::mem::size_of::<C::Scalar>(),
                })
            };
            buffers.push(s_buf);
        }

        let commitments = crate::cuda::bn254::batch_msm_v2(
            &g_buf,
            buffers.iter().map(|x| x as &CudaDeviceBufRaw).collect(),
            size,
        )?;
        for commitment in commitments {
            transcript.write_point(commitment).unwrap();
        }
        end_timer!(timer);
    }

    let x: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
    let xn = x.pow_vartime(&[1u64 << k]);

    let mut h_pieces = Vec::new_in(HugePageAllocator);
    h_pieces.resize(size, C::Scalar::zero());
    // pre-compute h_pieces for multi open
    {
        let last_ptr = unsafe {
            ManuallyDrop::new(CudaDeviceBufRaw {
                ptr: h_buf.ptr().offset(
                    ((domain.quotient_poly_degree as usize - 1)
                        * size
                        * core::mem::size_of::<C::Scalar>()) as isize,
                ),
                device: device.clone(),
                size: size * core::mem::size_of::<C::Scalar>(),
            })
        };
        let xn_buf = device.alloc_device_buffer_from_slice(&[xn][..])?;
        for i in (0..(domain.quotient_poly_degree - 1) as usize).rev() {
            let curr_ptr = unsafe {
                ManuallyDrop::new(CudaDeviceBufRaw {
                    ptr: h_buf
                        .ptr()
                        .offset((i * size * core::mem::size_of::<C::Scalar>()) as isize),
                    device: device.clone(),
                    size: size * core::mem::size_of::<C::Scalar>(),
                })
            };
            field_op_v3(
                device,
                &last_ptr,
                Some(&last_ptr),
                Some(&xn_buf),
                Some(&curr_ptr),
                None,
                size,
                FieldOp::Add,
                None,
            )?;
        }
        device.copy_from_device_to_host(&mut h_pieces[..], &last_ptr)?;
    }

    Ok((x, xn, h_pieces))
}

fn evaluate_h_gates_core<C: CurveAffine>(
    device: &CudaDevice,
    pk: &ProvingKey<C>,
    fixed: &[&[C::Scalar]],
    advice: &[&[C::Scalar]],
    instance: &[&[C::Scalar]],
    permutation_products: &[&[C::Scalar]],
    lookup_products: &mut [(
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
        &mut [C::Scalar],
    )],
    shuffle_products: &[&[C::Scalar]],
    y: C::Scalar,
    beta: C::Scalar,
    gamma: C::Scalar,
    theta: C::Scalar,
    intt_pq_buf: CudaDeviceBufRaw,
    intt_omegas_buf: CudaDeviceBufRaw,
    intt_divisor_buf: CudaDeviceBufRaw,
) -> DeviceResult<(EvalHContext<C::Scalar>, CudaDeviceBufRaw)> {
    let timer = start_timer!(|| "evaluate_h setup");
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
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
        y: vec![C::Scalar::one(), y],
        extended_allocator: vec![],
        k,
        extended_k,
        size,
        extended_size,
        extended_ntt_omegas_buf,
        extended_ntt_pq_buf,
        coset_powers_buf,
    };
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h gates");
    let exprs = analyze_expr_tree(&pk.ev.gpu_gates_expr[0], k);
    let h_buf =
        evaluate_prove_expr_with_async_ntt(device, &exprs, fixed, advice, instance, &mut ctx)?;
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
    let l_active_buf = ctx.alloc(device)?;
    device.copy_from_host_to_device(&l_active_buf, &l_active_row.values[..])?;
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
                buffer_copy_with_shift::<C::Scalar>(
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
                    field_mul::<C::Scalar>(&device, &l, &l_res, ctx.extended_size)?;

                    do_extended_prepare(device, &mut ctx, &mut r_res, None)?;
                    let coeff =
                        pick_from_buf::<C::Scalar>(device, &r_res, 0, 1, ctx.extended_size)?;
                    let short = vec![value[0] + gamma, coeff + curr_delta];
                    device.copy_from_host_to_device(&r_res, &short[..])?;
                    do_extended_ntt_pure(device, &mut ctx, &mut r_res)?;

                    field_mul::<C::Scalar>(&device, &r, &r_res, ctx.extended_size)?;
                    curr_delta *= &C::Scalar::DELTA;

                    ctx.extended_allocator.push(l_res);
                    ctx.extended_allocator.push(r_res);
                    ctx.extended_allocator.push(p_coset_buf);
                }

                field_sub::<C::Scalar>(&device, &l, &r, ctx.extended_size)?;
                field_mul::<C::Scalar>(&device, &l, &l_active_buf, ctx.extended_size)?;
                field_op_v2::<C::Scalar>(
                    &device,
                    &h_buf,
                    Some(&h_buf),
                    Some(y),
                    Some(&l),
                    None,
                    ctx.extended_size,
                    FieldOp::Add,
                )?;

                ctx.extended_allocator.push(l);
                ctx.extended_allocator.push(r);
            }
        }
    }
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h lookup");
    let gamma_buf = device.alloc_device_buffer_from_slice(&[gamma][..])?;
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..])?;
    for (_i, (lookup, (permuted_input, permuted_table, input, table, z))) in pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookup_products.into_iter())
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

        let (input_buf, stream_input) = if input_deg > 1 {
            (
                evaluate_prove_expr(device, &vec![e1], fixed, advice, instance, &mut ctx)?,
                None,
            )
        } else {
            unsafe {
                let mut stream = std::mem::zeroed();
                let _ = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                let mut buf = ctx.alloc(device)?;
                device.copy_from_host_to_device_async(&buf, &input, stream)?;

                let mut tmp_buf = ctx.alloc(device)?;
                field_op_v3(
                    device,
                    &buf,
                    Some(&buf),
                    None,
                    None,
                    Some(&beta_buf),
                    size,
                    FieldOp::Add,
                    Some(stream),
                )?;
                intt_raw_async(
                    &device,
                    &mut buf,
                    &mut tmp_buf,
                    &intt_pq_buf,
                    &intt_omegas_buf,
                    &intt_divisor_buf,
                    k,
                    Some(stream),
                )?;
                do_extended_prepare(device, &mut ctx, &mut buf, Some(stream))?;
                ntt_raw(
                    device,
                    &mut buf,
                    &mut tmp_buf,
                    &ctx.extended_ntt_pq_buf,
                    &ctx.extended_ntt_omegas_buf,
                    ctx.extended_k,
                    Some(stream),
                )?;

                (buf, Some((stream, tmp_buf)))
            }
        };

        let (table_buf, stream_table) = if table_deg > 1 {
            (
                evaluate_prove_expr(device, &vec![e2], fixed, advice, instance, &mut ctx)?,
                None,
            )
        } else {
            unsafe {
                let mut stream = std::mem::zeroed();
                let _ = cuda_runtime_sys::cudaStreamCreate(&mut stream);
                let mut buf = ctx.alloc(device)?;
                device.copy_from_host_to_device_async(&buf, &table, stream)?;

                let mut tmp_buf = ctx.alloc(device)?;
                field_op_v3(
                    device,
                    &buf,
                    Some(&buf),
                    None,
                    None,
                    Some(&gamma_buf),
                    size,
                    FieldOp::Add,
                    Some(stream),
                )?;
                intt_raw_async(
                    &device,
                    &mut buf,
                    &mut tmp_buf,
                    &intt_pq_buf,
                    &intt_omegas_buf,
                    &intt_divisor_buf,
                    k,
                    Some(stream),
                )?;
                do_extended_prepare(device, &mut ctx, &mut buf, Some(stream))?;
                ntt_raw(
                    device,
                    &mut buf,
                    &mut tmp_buf,
                    &ctx.extended_ntt_pq_buf,
                    &ctx.extended_ntt_omegas_buf,
                    ctx.extended_k,
                    Some(stream),
                )?;

                (buf, Some((stream, tmp_buf)))
            }
        };

        if let Some((stream, buf)) = stream_input {
            unsafe {
                cuda_runtime_sys::cudaStreamSynchronize(stream);
                cuda_runtime_sys::cudaStreamDestroy(stream);
            }
            ctx.extended_allocator.push(buf);
        }

        let (z_buf, tmp, stream) = if let Some((stream, tmp)) = stream_table {
            do_extended_ntt_v3_async(device, &mut ctx, stream, tmp, *z)?
        } else {
            do_extended_ntt_v2_async(device, &mut ctx, *z)?
        };

        let (permuted_input_buf, tmp, stream) =
            do_extended_ntt_v3_async(device, &mut ctx, stream, tmp, permuted_input)?;
        let (permuted_table_buf, tmp, stream) =
            do_extended_ntt_v3_async(device, &mut ctx, stream, tmp, permuted_table)?;

        unsafe {
            cuda_runtime_sys::cudaStreamSynchronize(stream);
            cuda_runtime_sys::cudaStreamDestroy(stream);
            ctx.extended_allocator.push(tmp);
        }

        unsafe {
            let mut stream = std::mem::zeroed();
            let _ = cuda_runtime_sys::cudaStreamCreate(&mut stream);
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
                stream,
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;

            {
                cuda_runtime_sys::cudaStreamSynchronize(stream);
                cuda_runtime_sys::cudaStreamDestroy(stream);
                ctx.extended_allocator.append(&mut vec![
                    input_buf,
                    table_buf,
                    permuted_input_buf,
                    permuted_table_buf,
                    z_buf,
                ])
            }
        }
    }
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h shuffle");

    let pick_host_buf_and_do_ntt_async =
        |ctx: &mut _,
         expr: &Expression<_>,
         tmp_buf: &mut Vec<_>,
         tmp_stream: &mut Vec<CudaStreamWrapper>,
         device_buf_vec: &mut Vec<_>| {
            let host_buf = match expr {
                Expression::Fixed { column_index, .. } => &*fixed[*column_index],
                Expression::Advice { column_index, .. } => &*advice[*column_index],
                Expression::Instance { column_index, .. } => &*instance[*column_index],
                _ => unreachable!(),
            };
            let (device_buf, tmp, stream) = do_extended_ntt_async(device, ctx, host_buf).unwrap();
            tmp_buf.push(tmp);
            tmp_stream.push(stream);
            device_buf_vec.push(device_buf);
        };

    let shuffle_group = pk.vk.cs.shuffles.group(pk.vk.cs.degree());
    for (_i, (shuffle, z)) in shuffle_group
        .iter()
        .zip(shuffle_products.iter())
        .enumerate()
    {
        if shuffle
            .0
            .iter()
            .find(|x| {
                x.input_expressions.len() > 1
                    || !is_expr_unit(&x.input_expressions[0])
                    || x.shuffle_expressions.len() > 1
                    || !is_expr_unit(&x.shuffle_expressions[0])
            })
            .is_none()
        {
            let mut inputs = vec![];
            let mut shuffles = vec![];

            let mut tmps = vec![];
            let mut streams = vec![];

            let (z_buf, tmp, stream) = do_extended_ntt_async(device, &mut ctx, z)?;
            tmps.push(tmp);
            streams.push(stream);

            for x in shuffle.0.iter() {
                pick_host_buf_and_do_ntt_async(
                    &mut ctx,
                    &x.input_expressions[0],
                    &mut tmps,
                    &mut streams,
                    &mut inputs,
                );
                pick_host_buf_and_do_ntt_async(
                    &mut ctx,
                    &x.shuffle_expressions[0],
                    &mut tmps,
                    &mut streams,
                    &mut shuffles,
                );
            }

            drop(streams);

            let mut betas = vec![beta];
            for _ in 1..inputs.len() {
                betas.push(*betas.last().unwrap() * beta);
            }

            let betas_buf = device.alloc_device_buffer_from_slice(&betas[..])?;
            let inputs_buf = device.alloc_device_buffer_from_slice(
                &inputs.iter().map(|x| x.ptr()).collect::<Vec<_>>()[..],
            )?;
            let shuffles_buf = device.alloc_device_buffer_from_slice(
                &shuffles.iter().map(|x| x.ptr()).collect::<Vec<_>>()[..],
            )?;

            unsafe {
                let err = shuffle_eval_h_v2(
                    h_buf.ptr(),
                    inputs_buf.ptr(),
                    shuffles_buf.ptr(),
                    betas_buf.ptr(),
                    inputs.len() as i32,
                    z_buf.ptr(),
                    l0_buf.ptr(),
                    l_last_buf.ptr(),
                    l_active_buf.ptr(),
                    y_buf.ptr(),
                    1 << (extended_k - k),
                    ctx.extended_size as i32,
                );

                to_result((), err, "fail to run field_op_batch_mul_sum")?;
                device.synchronize()?;
            }
        } else {
            let (input_expressions, table_expressions) = shuffle
                .0
                .iter()
                .map(|x| (x.input_expressions.clone(), x.shuffle_expressions.clone()))
                .collect::<Vec<_>>()
                .into_iter()
                .unzip();
            let [e1, e2] =
                flatten_shuffle_expression(&input_expressions, &table_expressions, beta, theta);

            let input_buf =
                evaluate_prove_expr(device, &vec![e1], fixed, advice, instance, &mut ctx)?;
            let table_buf =
                evaluate_prove_expr(device, &vec![e2], fixed, advice, instance, &mut ctx)?;
            let z_buf = do_extended_ntt_v2(device, &mut ctx, z)?;

            let timer = start_timer!(|| "shuffle_eval_h");
            unsafe {
                let err = shuffle_eval_h(
                    h_buf.ptr(),
                    input_buf.ptr(),
                    table_buf.ptr(),
                    z_buf.ptr(),
                    l0_buf.ptr(),
                    l_last_buf.ptr(),
                    l_active_buf.ptr(),
                    y_buf.ptr(),
                    1 << (extended_k - k),
                    ctx.extended_size as i32,
                );

                to_result((), err, "fail to run field_op_batch_mul_sum")?;
                device.synchronize()?;
            }
            end_timer!(timer);
        }
    }
    end_timer!(timer);

    Ok((ctx, h_buf))
}

fn get_expr_degree<F: FieldExt>(expr: &Vec<Expression<F>>) -> usize {
    let mut deg = 0;
    for expr in expr {
        deg = deg.max(expr.degree());
    }
    deg
}

fn do_extended_ntt_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw, CudaStreamWrapper)> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer_non_zeroed::<F>(ctx.extended_size)?
    } else {
        buf.unwrap()
    };

    let (stream_wrapper, stream) = CudaStreamWrapper::new_with_inner();
    device.copy_from_host_to_device_async::<F>(&buf, data, stream)?;
    do_extended_prepare(device, ctx, &mut buf, Some(stream))?;
    let tmp = do_extended_ntt_pure_async(device, ctx, &mut buf, Some(stream))?;

    Ok((buf, tmp, stream_wrapper))
}

fn do_extended_ntt_v2<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<CudaDeviceBufRaw> {
    let mut buf = ctx.alloc(device)?;
    device.copy_from_host_to_device::<F>(&buf, data)?;
    do_extended_ntt(device, ctx, &mut buf)?;

    Ok(buf)
}

fn do_extended_ntt_v2_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw, *mut CUstream_st)> {
    let mut buf = ctx.alloc(device)?;
    let (tmp, stream) = unsafe {
        let mut stream = std::mem::zeroed();
        let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
        assert_eq!(err, cuda_runtime_sys::cudaError::cudaSuccess);
        device.copy_from_host_to_device_async::<F>(&buf, data, stream)?;
        do_extended_prepare(device, ctx, &mut buf, Some(stream))?;
        (
            do_extended_ntt_pure_async(device, ctx, &mut buf, Some(stream))?,
            stream,
        )
    };

    Ok((buf, tmp, stream))
}

fn do_extended_ntt_v3_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    last_stream: *mut CUstream_st,
    last_tmp: CudaDeviceBufRaw,
    data: &[F],
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw, *mut CUstream_st)> {
    let mut buf = ctx.alloc(device)?;
    let (tmp, stream) = unsafe {
        let mut stream = std::mem::zeroed();
        let err = cuda_runtime_sys::cudaStreamCreate(&mut stream);
        assert_eq!(err, cuda_runtime_sys::cudaError::cudaSuccess);
        device.copy_from_host_to_device_async::<F>(&buf, data, stream)?;
        cuda_runtime_sys::cudaStreamSynchronize(last_stream);
        cuda_runtime_sys::cudaStreamDestroy(last_stream);
        ctx.extended_allocator.push(last_tmp);
        do_extended_prepare(device, ctx, &mut buf, Some(stream))?;
        (
            do_extended_ntt_pure_async(device, ctx, &mut buf, Some(stream))?,
            stream,
        )
    };

    Ok((buf, tmp, stream))
}

fn do_extended_prepare<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
    stream: Option<cudaStream_t>,
) -> DeviceResult<()> {
    extended_prepare(
        device,
        data,
        &ctx.coset_powers_buf,
        3,
        ctx.size,
        ctx.extended_size,
        stream,
    )
}

fn do_extended_ntt<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
) -> DeviceResult<()> {
    do_extended_prepare(device, ctx, data, None)?;
    do_extended_ntt_pure(device, ctx, data)?;
    Ok(())
}

fn _do_extended_ntt_pure_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
    tmp: Option<CudaDeviceBufRaw>,
    stream: Option<cudaStream_t>,
) -> DeviceResult<CudaDeviceBufRaw> {
    let tmp = tmp.or_else(|| ctx.extended_allocator.pop());
    let mut tmp = if tmp.is_none() {
        device.alloc_device_buffer_non_zeroed::<F>(ctx.extended_size)?
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
        stream,
    )?;
    Ok(tmp)
}

fn do_extended_ntt_pure_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
    stream: Option<cudaStream_t>,
) -> DeviceResult<CudaDeviceBufRaw> {
    _do_extended_ntt_pure_async(device, ctx, data, None, stream)
}

fn do_extended_ntt_pure<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
) -> DeviceResult<()> {
    let tmp = do_extended_ntt_pure_async(device, ctx, data, None)?;
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

            let err = field_op_batch_mul_sum(
                res.ptr(),
                group_buf.ptr(),
                rots_buf.ptr(),
                group.len() as i32,
                ctx.extended_size as i32,
                0 as _,
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;

            last_bufs = bufs;
        }
    }

    Ok(res)
}

fn evaluate_prove_expr_with_async_ntt<F: FieldExt>(
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

            let mut last_tmp = None;
            let mut last_stream = None;
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
                        let (buf, tmp, stream) = do_extended_ntt_v2_async(device, ctx, src)?;
                        if let Some(last_stream) = last_stream {
                            cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                            cuda_runtime_sys::cudaStreamDestroy(last_stream);
                            ctx.extended_allocator.push(last_tmp.unwrap());
                            last_tmp = Some(tmp);
                        } else {
                            last_tmp = Some(tmp);
                        }
                        last_stream = Some(stream);
                        bufs.insert(id, buf);
                    }
                    for _ in 0..*exp {
                        group.push(bufs.get(&id).unwrap().ptr());
                        rots.push(rot.0 << (ctx.extended_k - ctx.k));
                    }
                }

                group.push(0usize as _);
            }

            if let Some(last_stream) = last_stream {
                cuda_runtime_sys::cudaStreamSynchronize(last_stream);
                cuda_runtime_sys::cudaStreamDestroy(last_stream);
                ctx.extended_allocator.push(last_tmp.unwrap());
            }

            let group_buf = device.alloc_device_buffer_from_slice(&group[..])?;
            let rots_buf = device.alloc_device_buffer_from_slice(&rots[..])?;

            let err = field_op_batch_mul_sum(
                res.ptr(),
                group_buf.ptr(),
                rots_buf.ptr(),
                group.len() as i32,
                ctx.extended_size as i32,
                0 as _,
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;

            last_bufs = bufs;
        }
    }

    Ok(res)
}
