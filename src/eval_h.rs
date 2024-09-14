use std::collections::BTreeMap;
use std::mem::ManuallyDrop;

use ark_std::end_timer;
use ark_std::iterable::Iterable;
use ark_std::start_timer;
use cuda_runtime_sys::cudaMemset;
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
use crate::cuda::bn254::field_mul;
use crate::cuda::bn254::field_op_v2;
use crate::cuda::bn254::field_op_v3;
use crate::cuda::bn254::field_sub;
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
use crate::cuda::msm::batch_msm;
use crate::cuda::ntt::extended_prepare;
use crate::cuda::ntt::generate_ntt_buffers;
use crate::cuda::ntt::ntt_raw;
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
use crate::pinned_page::PinnedPageAllocator;

struct EvalHContext<'a, F: FieldExt> {
    device: &'a CudaDevice,
    y: Vec<F>,
    extended_allocator: Vec<CudaDeviceBufRaw>,
    extended_k: usize,
    k: usize,
    size: usize,
    extended_size: usize,
    extended_ntt_omegas_buf: CudaDeviceBufRaw,
    extended_ntt_pq_buf: CudaDeviceBufRaw,
    coset_powers_buf: CudaDeviceBufRaw,
    to_coset: bool,
}

impl<'a, F: FieldExt> EvalHContext<'a, F> {
    fn alloc(&mut self) -> DeviceResult<CudaDeviceBufRaw> {
        if let Some(buf) = self.extended_allocator.pop() {
            Ok(buf)
        } else {
            self.device
                .alloc_device_buffer_non_zeroed::<F>(self.extended_size)
        }
    }

    fn free(&mut self, buf: CudaDeviceBufRaw) {
        self.extended_allocator.push(buf);
    }
}

impl<'a, F: FieldExt> EvalHContext<'a, F> {
    fn extended_ntt_prepare(
        &mut self,
        data: &mut CudaDeviceBufRaw,
        stream: Option<&CudaStreamWrapper>,
    ) -> DeviceResult<()> {
        extended_prepare(
            &self.device,
            data,
            &self.coset_powers_buf,
            3,
            self.size,
            self.extended_size,
            self.to_coset,
            stream.map(|x| x.into()),
        )
    }

    fn extended_ntt_pure(
        &mut self,
        data: &mut CudaDeviceBufRaw,
        stream: Option<&CudaStreamWrapper>,
    ) -> DeviceResult<CudaDeviceBufRaw> {
        let mut tmp = self.alloc()?;
        ntt_raw(
            &self.device,
            data,
            &mut tmp,
            &self.extended_ntt_pq_buf,
            &self.extended_ntt_omegas_buf,
            self.extended_k,
            None,
            stream,
        )?;
        Ok(tmp)
    }

    fn extended_ntt_async(
        &mut self,
        data: &mut CudaDeviceBufRaw,
    ) -> DeviceResult<(CudaStreamWrapper, CudaDeviceBufRaw)> {
        let stream = CudaStreamWrapper::new();
        self.extended_ntt_prepare(data, Some(&stream))?;
        let tmp = self.extended_ntt_pure(data, Some(&stream))?;

        Ok((stream, tmp))
    }

    fn extended_ntt_sync(&mut self, data: &mut CudaDeviceBufRaw) -> DeviceResult<()> {
        let (stream, tmp) = self.extended_ntt_async(data)?;
        self.extended_ntt_wait((stream, tmp))?;
        Ok(())
    }

    fn copy_and_extended_ntt_async(
        &mut self,
        data: &[F],
    ) -> DeviceResult<(CudaDeviceBufRaw, (CudaStreamWrapper, CudaDeviceBufRaw))> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let mut buf = self.alloc()?;
        self.device
            .copy_from_host_to_device_async(&buf, data, stream)?;
        self.extended_ntt_prepare(&mut buf, Some(&sw))?;
        let tmp = self.extended_ntt_pure(&mut buf, Some(&sw))?;

        Ok((buf, (sw, tmp)))
    }

    fn copy_and_extended_ntt_sync(&mut self, data: &[F]) -> DeviceResult<CudaDeviceBufRaw> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let mut buf = self.alloc()?;
        self.device
            .copy_from_host_to_device_async(&buf, data, stream)?;
        self.extended_ntt_prepare(&mut buf, Some(&sw))?;
        let tmp = self.extended_ntt_pure(&mut buf, Some(&sw))?;
        self.extended_ntt_wait((sw, tmp))?;

        Ok(buf)
    }

    fn batch_copy_and_extended_ntt_async<const P: usize>(
        &mut self,
        batch_data: &[&[F]],
        pre_stream_and_tmps: [Option<(CudaStreamWrapper, CudaDeviceBufRaw)>; P],
    ) -> DeviceResult<Vec<CudaDeviceBufRaw>> {
        const BATCH_SIZE: usize = 4;

        let mut stream_and_tmp_queue = [0; BATCH_SIZE].map(|_| None);
        let mut index = 0;

        let mut push_stream_and_tmp = |ctx: &mut Self, mut stream_and_tmp| {
            std::mem::swap(&mut stream_and_tmp, &mut stream_and_tmp_queue[index]);
            stream_and_tmp.map(|x| ctx.extended_ntt_wait(x));
            index = (index + 1) % BATCH_SIZE;
        };

        for pre_stream_and_tmp in pre_stream_and_tmps.into_iter().filter_map(|x| x) {
            push_stream_and_tmp(self, Some(pre_stream_and_tmp));
        }

        let mut res = vec![];
        for data in batch_data {
            let (buf, stream_and_tmp) = self.copy_and_extended_ntt_async(data)?;
            push_stream_and_tmp(self, Some(stream_and_tmp));
            res.push(buf)
        }

        for stream_and_tmp in stream_and_tmp_queue.into_iter().filter_map(|x| x) {
            self.extended_ntt_wait(stream_and_tmp)?;
        }

        Ok(res)
    }

    fn extended_ntt_wait(
        &mut self,
        last: (CudaStreamWrapper, CudaDeviceBufRaw),
    ) -> DeviceResult<()> {
        let (stream, tmp_buf) = last;
        stream.sync();
        self.free(tmp_buf);
        Ok(())
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
    let (intt_omegas_buf, intt_pq_buf) = generate_ntt_buffers(
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
) -> DeviceResult<(C::Scalar, C::Scalar, Vec<C::Scalar, PinnedPageAllocator>)> {
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

        let (extended_intt_omegas_buf, extended_intt_pq_buf) = generate_ntt_buffers(
            &device,
            domain.extended_omega_inv,
            pk.vk.domain.extended_k() as usize,
        )?;
        let mut tmp = ctx.alloc()?;

        ntt_raw(
            &device,
            &mut h_buf,
            &mut tmp,
            &extended_intt_pq_buf,
            &extended_intt_omegas_buf,
            pk.vk.domain.extended_k() as usize,
            Some(&intt_divisor_buf),
            None,
        )?;

        ctx.coset_powers_buf =
            device.alloc_device_buffer_from_slice(&[domain.g_coset_inv, domain.g_coset])?;
        ctx.to_coset = true;
        ctx.extended_ntt_prepare(&mut h_buf, None)?;

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

        let commitments = batch_msm(
            device,
            &g_buf,
            buffers
                .iter()
                .map(|x| x as &CudaDeviceBufRaw)
                .collect::<Vec<_>>(),
            None,
            size,
        )?
        .0;
        for commitment in commitments {
            transcript.write_point(commitment).unwrap();
        }
        end_timer!(timer);
    }

    let x: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
    let xn = x.pow_vartime(&[1u64 << k]);

    let mut h_pieces = Vec::new_in(PinnedPageAllocator);
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

fn evaluate_h_gates_core<'a, C: CurveAffine>(
    device: &'a CudaDevice,
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
) -> DeviceResult<(EvalHContext<'a, C::Scalar>, CudaDeviceBufRaw)> {
    let timer = start_timer!(|| "evaluate_h setup");
    let k = pk.get_vk().domain.k() as usize;
    let size = 1 << pk.get_vk().domain.k();
    let extended_k = pk.get_vk().domain.extended_k() as usize;
    let extended_size = 1 << extended_k;
    let extended_omega = pk.vk.domain.get_extended_omega();

    let (extended_ntt_omegas_buf, extended_ntt_pq_buf) =
        generate_ntt_buffers(device, extended_omega, extended_k)?;
    let coset_powers_buf = device.alloc_device_buffer_from_slice(&[
        pk.get_vk().domain.g_coset,
        pk.get_vk().domain.g_coset_inv,
    ])?;

    let mut ctx = EvalHContext {
        device,
        y: vec![C::Scalar::one(), y],
        extended_allocator: vec![],
        k,
        extended_k,
        size,
        extended_size,
        extended_ntt_omegas_buf,
        extended_ntt_pq_buf,
        coset_powers_buf,
        to_coset: false,
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
    let (l0_buf, (l0_stream, l0_tmp)) = ctx.copy_and_extended_ntt_async(&l0.values[..])?;
    let (l_last_buf, l_last_stream_buf) = ctx.copy_and_extended_ntt_async(&l_last.values[..])?;
    let l_active_buf = ctx.alloc()?;
    device.copy_from_host_to_device_async(
        &l_active_buf,
        &l_active_row.values[..],
        (&l0_stream).into(),
    )?;
    ctx.extended_ntt_wait((l0_stream, l0_tmp))?;
    ctx.extended_ntt_wait(l_last_stream_buf)?;
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h permutation");
    if permutation_products.len() > 0 {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let last_rotation = (ctx.size - (blinding_factors + 1)) << (extended_k - k);
        let chunk_len = pk.vk.cs.degree() - 2;

        let extended_p_buf = ctx.batch_copy_and_extended_ntt_async(permutation_products, [])?;

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
                let l = ctx.alloc()?;
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
                    let mut l_res = ctx.alloc()?;
                    let mut r_res = ctx.alloc()?;
                    let p_coset_buf = ctx.alloc()?;
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
                    ctx.extended_ntt_sync(&mut l_res)?;
                    field_mul::<C::Scalar>(&device, &l, &l_res, ctx.extended_size)?;

                    ctx.extended_ntt_prepare(&mut r_res, None)?;
                    let coeff =
                        pick_from_buf::<C::Scalar>(device, &r_res, 0, 1, ctx.extended_size)?;
                    let short = vec![value[0] + gamma, coeff + curr_delta];
                    device.copy_from_host_to_device(&r_res, &short[..])?;
                    ctx.extended_ntt_pure(&mut r_res, None)?;

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
            let (sw, stream) = CudaStreamWrapper::new_with_inner();
            let mut buf = ctx.alloc()?;
            device.copy_from_host_to_device_async(&buf, &input, stream)?;

            let mut tmp_buf = ctx.alloc()?;
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
            ntt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                k,
                Some(&intt_divisor_buf),
                Some(&sw),
            )?;
            ctx.extended_ntt_prepare(&mut buf, Some(&sw))?;
            ntt_raw(
                device,
                &mut buf,
                &mut tmp_buf,
                &ctx.extended_ntt_pq_buf,
                &ctx.extended_ntt_omegas_buf,
                ctx.extended_k,
                None,
                Some(&sw),
            )?;

            (buf, Some((sw, tmp_buf)))
        };

        let (table_buf, stream_table) = if table_deg > 1 {
            (
                evaluate_prove_expr(device, &vec![e2], fixed, advice, instance, &mut ctx)?,
                None,
            )
        } else {
            let (sw, stream) = CudaStreamWrapper::new_with_inner();
            let mut buf = ctx.alloc()?;
            device.copy_from_host_to_device_async(&buf, &table, stream)?;

            let mut tmp_buf = ctx.alloc()?;
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
            ntt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                k,
                Some(&intt_divisor_buf),
                Some(&sw),
            )?;
            ctx.extended_ntt_prepare(&mut buf, Some(&sw))?;
            ntt_raw(
                device,
                &mut buf,
                &mut tmp_buf,
                &ctx.extended_ntt_pq_buf,
                &ctx.extended_ntt_omegas_buf,
                ctx.extended_k,
                None,
                Some(&sw),
            )?;

            (buf, Some((sw, tmp_buf)))
        };

        let bufs = ctx.batch_copy_and_extended_ntt_async(
            &[*z, permuted_input, permuted_table],
            [stream_input, stream_table],
        )?;
        let [z_buf, permuted_input_buf, permuted_table_buf]: [_; 3] = bufs.try_into().unwrap();

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
        |ctx: &mut EvalHContext<'_, _>,
         expr: &Expression<_>,
         stream_and_tmps: &mut Vec<(CudaStreamWrapper, CudaDeviceBufRaw)>,
         device_buf_vec: &mut Vec<_>| {
            let host_buf = match expr {
                Expression::Fixed { column_index, .. } => &*fixed[*column_index],
                Expression::Advice { column_index, .. } => &*advice[*column_index],
                Expression::Instance { column_index, .. } => &*instance[*column_index],
                _ => unreachable!(),
            };
            let (device_buf, stream_and_tmp) = ctx.copy_and_extended_ntt_async(host_buf).unwrap();
            stream_and_tmps.push(stream_and_tmp);
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

            let mut stream_and_tmps = vec![];

            let (z_buf, stream_and_tmp) = ctx.copy_and_extended_ntt_async(z)?;
            stream_and_tmps.push(stream_and_tmp);

            for x in shuffle.0.iter() {
                pick_host_buf_and_do_ntt_async(
                    &mut ctx,
                    &x.input_expressions[0],
                    &mut stream_and_tmps,
                    &mut inputs,
                );
                pick_host_buf_and_do_ntt_async(
                    &mut ctx,
                    &x.shuffle_expressions[0],
                    &mut stream_and_tmps,
                    &mut shuffles,
                );
            }

            drop(stream_and_tmps);

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
            let z_buf = ctx.copy_and_extended_ntt_sync(z)?;

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
    let res = ctx.alloc()?;
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
                        let buf = ctx.copy_and_extended_ntt_sync(src)?;
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
    let res = ctx.alloc()?;
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

            let mut last_stream_and_tmp = None;
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
                        let (buf, stream_and_tmp) = ctx.copy_and_extended_ntt_async(src)?;
                        if let Some(stream_and_tmp) = last_stream_and_tmp {
                            ctx.extended_ntt_wait(stream_and_tmp)?;
                        }
                        last_stream_and_tmp = Some(stream_and_tmp);
                        bufs.insert(id, buf);
                    }
                    for _ in 0..*exp {
                        group.push(bufs.get(&id).unwrap().ptr());
                        rots.push(rot.0 << (ctx.extended_k - ctx.k));
                    }
                }

                group.push(0usize as _);
            }

            last_stream_and_tmp.map(|stream_and_tmp| ctx.extended_ntt_wait(stream_and_tmp));

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
