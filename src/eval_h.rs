use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem::swap;
use std::mem::ManuallyDrop;

use ark_std::end_timer;
use ark_std::iterable::Iterable;
use ark_std::start_timer;
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
use crate::cuda::bn254::permutation_eval_h_l;
use crate::cuda::bn254::permutation_eval_h_p1;
use crate::cuda::bn254::permutation_eval_h_p2;
use crate::cuda::bn254::pick_from_buf;
use crate::cuda::bn254::FieldOp;
use crate::cuda::bn254::{logup_eval_h_inputs_product_sum, logup_eval_h_z_set};
use crate::cuda::bn254_c;
use crate::cuda::bn254_c::field_op_batch_mul_sum;
use crate::cuda::bn254_c::logup_eval_h;
use crate::cuda::bn254_c::logup_eval_h_extra_inputs;
use crate::cuda::bn254_c::shuffle_eval_h;
use crate::cuda::bn254_c::shuffle_eval_h_v2;
use crate::cuda::field_op::field_op;
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
use crate::expr::get_expr_degree;
use crate::expr::is_expression_pure_unit;
use crate::hugetlb::HugePageAllocator;

struct EvalHContext<'a, F: FieldExt> {
    device: &'a CudaDevice,
    y: Vec<F>,
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
        self.device
            .alloc_device_buffer_non_zeroed::<F>(self.extended_size)
    }

    fn alloc_zeroed(&mut self, stream: &CudaStreamWrapper) -> DeviceResult<CudaDeviceBufRaw> {
        self.device
            .alloc_device_buffer_async::<F>(self.extended_size, stream)
    }

    fn free(&mut self, buf: CudaDeviceBufRaw) {
        drop(buf)
    }
}

impl<'a, F: FieldExt> EvalHContext<'a, F> {
    fn eval_ys_unit(&mut self, index: usize) -> F {
        while self.y.len() <= index {
            self.y.push(self.y[1] * self.y.last().unwrap())
        }

        self.y[index]
    }

    fn eval_ys<'b>(&mut self, ys_iter: impl Iterator<Item = &'b BTreeMap<u32, F>>) -> Vec<F> {
        let mut res = vec![];
        for ys in ys_iter {
            let mut acc = F::zero();
            for (ys, f) in ys.iter() {
                acc = acc + self.eval_ys_unit(*ys as usize) * f
            }
            res.push(acc);
        }
        res
    }

    fn evaluate_prove_expr_with_async_ntt<'b>(
        &mut self,
        exprs: &'b [Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>],
        fixed: &[&[F]],
        advice: &[&[F]],
        instance: &[&[F]],
    ) -> DeviceResult<(CudaDeviceBufRaw, (CudaStreamWrapper, Vec<CudaDeviceBufRaw>))> {
        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let res = self.alloc_zeroed(&sw)?;

        let mut last_bufs: HashMap<_, CudaDeviceBufRaw> = HashMap::new();
        let mut pending_bufs = vec![];
        for expr in exprs.iter() {
            let coeffs = self.eval_ys(expr.iter().map(|(_, coeff)| coeff));
            let coeffs_buf = self
                .device
                .alloc_device_buffer_from_slice_async(&coeffs[..], stream)?;

            let mut bufs = HashMap::new();
            let mut group = vec![];
            let mut rots = vec![];

            for (unit, _) in expr.iter().flat_map(|(units, _)| units.iter()) {
                let id = unit.get_group();
                last_bufs.remove(&id).map(|buf| bufs.insert(id, buf));
            }
            sw.sync();
            drop(last_bufs);
            drop(pending_bufs);

            let pick = |unit: &'b _| match unit {
                ProveExpressionUnit::Fixed {
                    column_index,
                    rotation,
                } => (&fixed[column_index.clone()], rotation),
                ProveExpressionUnit::Advice {
                    column_index,
                    rotation,
                } => (&advice[column_index.clone()], rotation),
                ProveExpressionUnit::Instance {
                    column_index,
                    rotation,
                } => (&instance[column_index.clone()], rotation),
            };

            let mut stream_and_tmp_queue = vec![];
            let mut index = 0;
            for (unit, _) in expr.iter().flat_map(|(units, _)| units.iter()) {
                let id = unit.get_group();
                let (src, _) = pick(unit);
                if !bufs.contains_key(&id) {
                    let (buf, mut stream_and_tmp) = self.copy_and_extended_ntt_async(src)?;
                    bufs.insert(id, buf);
                    if stream_and_tmp_queue.len() >= 2 {
                        swap(&mut stream_and_tmp, &mut stream_and_tmp_queue[index & 1]);
                        index += 1;
                        self.extended_ntt_wait(stream_and_tmp)?;
                    } else {
                        stream_and_tmp_queue.push(stream_and_tmp)
                    }
                }
            }

            for (i, (units, _)) in expr.iter().enumerate() {
                group.push(unsafe {
                    coeffs_buf
                        .ptr()
                        .offset((i * core::mem::size_of::<F>()) as isize)
                });

                for (unit, exp) in units {
                    let id = unit.get_group();
                    let (_, rot) = pick(unit);

                    for _ in 0..*exp {
                        group.push(bufs.get(&id).unwrap().ptr());
                        rots.push(rot.0 << (self.extended_k - self.k));
                    }
                }

                group.push(0usize as _);
            }

            let group_buf = self
                .device
                .alloc_device_buffer_from_slice_async(&group[..], stream)?;
            let rots_buf = self
                .device
                .alloc_device_buffer_from_slice_async(&rots[..], stream)?;
            for stream_and_tmp in stream_and_tmp_queue {
                self.extended_ntt_wait(stream_and_tmp)?;
            }

            unsafe {
                let err = field_op_batch_mul_sum(
                    res.ptr(),
                    group_buf.ptr(),
                    rots_buf.ptr(),
                    group.len() as i32,
                    self.extended_size as i32,
                    stream,
                );

                to_result((), err, "fail to run field_op_batch_mul_sum")?;
            }

            last_bufs = bufs;
            pending_bufs = vec![group_buf, rots_buf];
        }

        Ok((res, (sw, pending_bufs)))
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
        stream: &CudaStreamWrapper,
    ) -> DeviceResult<CudaDeviceBufRaw> {
        self.extended_ntt_prepare(data, Some(&stream))?;
        let tmp = self.extended_ntt_pure(data, Some(&stream))?;

        Ok(tmp)
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

    fn batch_copy_and_extended_ntt_async(
        &mut self,
        batch_data: &[&[F]],
        pre_stream_and_tmps: Vec<(CudaStreamWrapper, CudaDeviceBufRaw)>,
    ) -> DeviceResult<Vec<CudaDeviceBufRaw>> {
        const BATCH_SIZE: usize = 4;

        let mut stream_and_tmp_queue = [0; BATCH_SIZE].map(|_| None);
        let mut index = 0;

        let mut push_stream_and_tmp = |ctx: &mut Self, mut stream_and_tmp| {
            std::mem::swap(&mut stream_and_tmp, &mut stream_and_tmp_queue[index]);
            stream_and_tmp.map(|x| ctx.extended_ntt_wait(x));
            index = (index + 1) % BATCH_SIZE;
        };

        for pre_stream_and_tmp in pre_stream_and_tmps {
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
    //todo check the inner mut to remove with outer mut
    lookup_products: &mut Vec<(
        &mut Vec<Vec<Vec<C::Scalar, HugePageAllocator>>>,
        &mut Vec<C::Scalar, HugePageAllocator>,
        &mut Vec<C::Scalar, HugePageAllocator>,
        &mut Vec<Vec<C::Scalar, HugePageAllocator>>,
    )>,
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
            &device,
            &g_buf,
            buffers
                .iter()
                .map(|x| &x as &CudaDeviceBufRaw)
                .collect::<Vec<_>>(),
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
            field_op::<C::Scalar>(
                device,
                &last_ptr,
                (&last_ptr as &CudaDeviceBufRaw, &xn_buf),
                &curr_ptr as &CudaDeviceBufRaw,
                size,
                FieldOp::Add,
                None,
            )?;
        }
        device.copy_from_device_to_host(&mut h_pieces[..], &last_ptr)?;
    }

    Ok((x, xn, h_pieces))
}

fn logup_transform_extend_coset_async<F: FieldExt>(
    device: &CudaDevice,
    values: &[F],
    beta_buf: &CudaDeviceBufRaw,
    pq_buf: &CudaDeviceBufRaw,
    omegas_buf: &CudaDeviceBufRaw,
    divisor: &CudaDeviceBufRaw,
    ctx: &mut EvalHContext<F>,
) -> DeviceResult<(CudaDeviceBufRaw, (CudaStreamWrapper, CudaDeviceBufRaw))> {
    let (sw, stream) = CudaStreamWrapper::new_with_inner();
    let mut buf = ctx.alloc()?;
    device.copy_from_host_to_device_async(&buf, values, stream)?;

    let mut tmp_buf = ctx.alloc()?;
    field_op::<F>(
        device,
        &buf,
        &buf,
        ((), beta_buf),
        ctx.size,
        FieldOp::Add,
        Some(stream),
    )?;
    ntt_raw(
        &device,
        &mut buf,
        &mut tmp_buf,
        &pq_buf,
        &omegas_buf,
        ctx.k,
        Some(divisor),
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

    Ok((buf, (sw, tmp_buf)))
}

fn evaluate_h_gates_core<'a, C: CurveAffine>(
    device: &'a CudaDevice,
    pk: &ProvingKey<C>,
    fixed: &[&[C::Scalar]],
    advice: &[&[C::Scalar]],
    instance: &[&[C::Scalar]],
    permutation_products: &[&[C::Scalar]],
    lookup_products: &mut Vec<(
        &mut Vec<Vec<Vec<C::Scalar, HugePageAllocator>>>,
        &mut Vec<C::Scalar, HugePageAllocator>,
        &mut Vec<C::Scalar, HugePageAllocator>,
        &mut Vec<Vec<C::Scalar, HugePageAllocator>>,
    )>,
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
    let (h_buf, _) = ctx.evaluate_prove_expr_with_async_ntt(&exprs, fixed, advice, instance)?;
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h prepare buffers for constants");
    let y_buf = device.alloc_device_buffer_from_slice(&[y][..])?;
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..])?;
    let gamma_buf = device.alloc_device_buffer_from_slice(&[gamma][..])?;

    let l0 = &pk.l0;
    let l_last = &pk.l_last;
    let l_active_row = &pk.l_active_row;
    let blinding_factors = pk.vk.cs.blinding_factors();
    let last_rotation = (ctx.size - (blinding_factors + 1)) << (extended_k - k);
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
        let chunk_len = pk.vk.cs.degree() - 2;

        let extended_p_buf = ctx.batch_copy_and_extended_ntt_async(permutation_products, vec![])?;
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
                let l_acc = ctx.alloc()?;
                let r_acc = extended_p_buf;
                buffer_copy_with_shift::<C::Scalar>(
                    &device,
                    &l_acc,
                    &r_acc,
                    1 << (extended_k - k),
                    ctx.extended_size,
                )?;

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

                    device.copy_from_host_to_device(&l_res, value)?;
                    device
                        .copy_from_device_to_device::<C::Scalar>(&r_res, 0, &l_res, 0, ctx.size)?;

                    let (l_sw, l_stream) = CudaStreamWrapper::new_with_inner();
                    let (r_sw, r_stream) = CudaStreamWrapper::new_with_inner();

                    let r_tmp = {
                        ctx.extended_ntt_prepare(&mut r_res, Some(&r_sw))?;
                        r_sw.sync();
                        let coeff =
                            pick_from_buf::<C::Scalar>(device, &r_res, 0, 1, ctx.extended_size)?;
                        let short = vec![value[0] + gamma, coeff + curr_delta];
                        device.copy_from_host_to_device_async(&r_res, &short[..], r_stream)?;
                        let tmp = ctx.extended_ntt_pure(&mut r_res, Some(&r_sw))?;
                        field_op::<C::Scalar>(
                            &device,
                            &r_acc,
                            &r_acc,
                            &r_res,
                            ctx.extended_size,
                            FieldOp::Mul,
                            Some(r_stream),
                        )?;
                        tmp
                    };

                    let p_coset_buf = ctx.alloc()?;
                    let l_tmp = {
                        device.copy_from_host_to_device_async(
                            &p_coset_buf,
                            &permutation.values[..],
                            l_stream,
                        )?;
                        permutation_eval_h_l(
                            &device,
                            &l_res,
                            &beta_buf,
                            &gamma_buf,
                            &p_coset_buf,
                            ctx.size,
                            Some(l_stream),
                        )?;
                        let tmp = ctx.extended_ntt_async(&mut l_res, &l_sw)?;
                        field_op::<C::Scalar>(
                            &device,
                            &l_acc,
                            &l_acc,
                            &l_res,
                            ctx.extended_size,
                            FieldOp::Mul,
                            Some(l_stream),
                        )?;
                        tmp
                    };

                    r_sw.sync();
                    l_sw.sync();

                    curr_delta *= &C::Scalar::DELTA;

                    ctx.free(l_res);
                    ctx.free(r_res);
                    ctx.free(p_coset_buf);
                    ctx.free(r_tmp);
                    ctx.free(l_tmp);
                }

                field_op::<C::Scalar>(
                    &device,
                    &l_acc,
                    &l_acc,
                    &r_acc,
                    ctx.extended_size,
                    FieldOp::Sub,
                    None,
                )?;
                field_op::<C::Scalar>(
                    &device,
                    &l_acc,
                    &l_acc,
                    &l_active_buf,
                    ctx.extended_size,
                    FieldOp::Mul,
                    None,
                )?;
                field_op::<C::Scalar>(
                    &device,
                    &h_buf,
                    (&h_buf, 0, y),
                    &l_acc,
                    ctx.extended_size,
                    FieldOp::Add,
                    None,
                )?;

                ctx.free(l_acc);
                ctx.free(r_acc);
            }
        }
    }
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h lookup");
    let beta_buf = device.alloc_device_buffer_from_slice(&[beta][..])?;

    let mut last_round_stream_and_buffers: Option<(CudaStreamWrapper, _)> = None;
    for (_i, (lookup, (inputs_sets_host, table_host, multiplicity_host, z_set_host))) in pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookup_products.into_iter())
        .enumerate()
    {
        let sets_len = lookup.input_expressions_sets.len();
        let mut input_deg = 2;
        lookup.input_expressions_sets.iter().for_each(|set| {
            set.0.iter().for_each(|input| {
                input_deg = input_deg.max(get_expr_degree(input));
            })
        });
        let table_deg = get_expr_degree(&lookup.table_expressions);

        let (inputs_product_expr, inputs_product_sum_expr, table_expr) = flatten_lookup_expression(
            &lookup
                .input_expressions_sets
                .iter()
                .map(|set| set.0.clone())
                .collect::<Vec<_>>(),
            &lookup.table_expressions,
            beta,
            theta,
        );

        // calculate inputs_host+beta in advance for input_deg=1
        let inputs_beta_sets = if input_deg == 1 {
            let (inputs_buf_sets, streams): (Vec<_>, Vec<_>) = inputs_sets_host
                .iter()
                .map(|set| -> Result<(Vec<_>, Vec<_>), _> {
                    let rst = set
                        .iter()
                        .map(|input| -> DeviceResult<_> {
                            logup_transform_extend_coset_async(
                                device,
                                input,
                                &beta_buf,
                                &intt_pq_buf,
                                &intt_omegas_buf,
                                &intt_divisor_buf,
                                &mut ctx,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .unzip();
                    Ok(rst)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();

            drop(streams);
            Some(inputs_buf_sets)
        } else {
            None
        };

        let (multiplicity_buf, m_stream_and_tmp) =
            ctx.copy_and_extended_ntt_async(multiplicity_host)?;

        last_round_stream_and_buffers.map(|(sw, buffers)| {
            sw.sync();
            for buffer in buffers {
                ctx.free(buffer);
            }
        });

        let mut inputs_products_bufs = vec![];
        let mut inputs_products_sum_bufs = vec![];
        let mut zs_bufs = vec![];
        let mut last_waiting = m_stream_and_tmp;

        for i in 0..inputs_product_expr.len() {
            if input_deg > 1 {
                let (buf, pendings_product) = ctx.evaluate_prove_expr_with_async_ntt(
                    &inputs_product_expr[i..i + 1],
                    fixed,
                    advice,
                    instance,
                )?;
                inputs_products_bufs.push(buf);
                let (buf, pendings_sum) = ctx.evaluate_prove_expr_with_async_ntt(
                    &inputs_product_sum_expr[i..i + 1],
                    fixed,
                    advice,
                    instance,
                )?;
                inputs_products_sum_bufs.push(buf);

                ctx.extended_ntt_wait(last_waiting)?;
                let (z_buf, stream_and_tmp) =
                    ctx.copy_and_extended_ntt_async(&z_set_host[i][..])?;
                last_waiting = stream_and_tmp;
                zs_bufs.push(z_buf);

                drop([pendings_product, pendings_sum]);
            } else {
                let (sw, stream) = CudaStreamWrapper::new_with_inner();
                let product_buf = ctx.alloc()?;
                let product_sum_buf = ctx.alloc()?;
                logup_eval_h_inputs_product_sum(
                    device,
                    &product_buf,
                    &product_sum_buf,
                    &inputs_beta_sets.as_ref().unwrap()[i],
                    ctx.extended_size,
                    Some(stream),
                )?;
                inputs_products_bufs.push(product_buf);
                inputs_products_sum_bufs.push(product_sum_buf);

                ctx.extended_ntt_wait(last_waiting)?;
                let (z_buf, stream_and_tmp) =
                    ctx.copy_and_extended_ntt_async(&z_set_host[i][..])?;

                drop(sw);
                last_waiting = stream_and_tmp;
                zs_bufs.push(z_buf);
            }
        }

        let table_buf = if table_deg > 1 {
            let (buf, pendings) =
                ctx.evaluate_prove_expr_with_async_ntt(&vec![table_expr], fixed, advice, instance)?;
            ctx.extended_ntt_wait(last_waiting)?;
            drop(pendings);
            buf
        } else {
            let (buf, (sw, tmp_buf)) = logup_transform_extend_coset_async(
                device,
                &table_host[..],
                &beta_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                &mut ctx,
            )?;
            ctx.extended_ntt_wait(last_waiting)?;
            ctx.extended_ntt_wait((sw, tmp_buf))?;
            buf
        };

        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        unsafe {
            let err = logup_eval_h(
                h_buf.ptr(),
                inputs_products_bufs[0].ptr(),
                inputs_products_sum_bufs[0].ptr(),
                table_buf.ptr(),
                multiplicity_buf.ptr(),
                zs_bufs.first().unwrap().ptr(),
                zs_bufs.last().unwrap().ptr(),
                l0_buf.ptr(),
                l_last_buf.ptr(),
                l_active_buf.ptr(),
                y_buf.ptr(),
                1 << (extended_k - k),
                ctx.extended_size as i32,
                stream,
            );
            to_result((), err, "fail to run logup_eval_h")?;

            if sets_len > 1 {
                logup_eval_h_z_set(
                    device,
                    &h_buf,
                    &zs_bufs[..],
                    &l0_buf,
                    &l_last_buf,
                    &y_buf,
                    last_rotation,
                    ctx.extended_size,
                    Some(stream),
                )?;

                for ((input_product, input_product_sum), z) in inputs_products_bufs
                    .iter()
                    .zip(inputs_products_sum_bufs.iter())
                    .zip(zs_bufs.iter())
                    .skip(1)
                {
                    let err = logup_eval_h_extra_inputs(
                        h_buf.ptr(),
                        input_product.ptr(),
                        input_product_sum.ptr(),
                        z.ptr(),
                        l_active_buf.ptr(),
                        y_buf.ptr(),
                        1 << (extended_k - k),
                        ctx.extended_size as i32,
                        stream,
                    );
                    to_result((), err, "fail to run logup_eval_h_extra_inputs")?;
                }
            }

            last_round_stream_and_buffers = Some((
                sw,
                vec![
                    vec![table_buf],
                    vec![multiplicity_buf],
                    inputs_products_bufs,
                    inputs_products_sum_bufs,
                    zs_bufs,
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
            ));
        }
    }
    end_timer!(timer);

    let timer = start_timer!(|| "evaluate_h shuffle");

    let pick_host_buf_and_do_ntt_async =
        |ctx: &mut EvalHContext<'_, _>,
         expr: &Expression<_>,
         stream_and_tmps: &mut Vec<(CudaStreamWrapper, CudaDeviceBufRaw)>,
         device_buf_vec: &mut Vec<_>|
         -> DeviceResult<()> {
            let host_buf = match expr {
                Expression::Fixed { column_index, .. } => &*fixed[*column_index],
                Expression::Advice { column_index, .. } => &*advice[*column_index],
                Expression::Instance { column_index, .. } => &*instance[*column_index],
                _ => unreachable!(),
            };
            let (device_buf, stream_and_tmp) = ctx.copy_and_extended_ntt_async(host_buf).unwrap();
            while stream_and_tmps.len() >= 2 {
                ctx.extended_ntt_wait(stream_and_tmps.pop().unwrap())?;
            }
            stream_and_tmps.push(stream_and_tmp);
            device_buf_vec.push(device_buf);
            Ok(())
        };

    let mut betas = vec![beta];
    for _ in 1..16 {
        betas.push(*betas.last().unwrap() * beta);
    }
    let betas_buf = device.alloc_device_buffer_from_slice(&betas[..])?;

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
                    || !is_expression_pure_unit(&x.input_expressions[0])
                    || x.shuffle_expressions.len() > 1
                    || !is_expression_pure_unit(&x.shuffle_expressions[0])
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
                )?;
                pick_host_buf_and_do_ntt_async(
                    &mut ctx,
                    &x.shuffle_expressions[0],
                    &mut stream_and_tmps,
                    &mut shuffles,
                )?;
            }

            drop(stream_and_tmps);

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

            let (input_buf, _) =
                ctx.evaluate_prove_expr_with_async_ntt(&vec![e1], fixed, advice, instance)?;
            let (table_buf, _) =
                ctx.evaluate_prove_expr_with_async_ntt(&vec![e2], fixed, advice, instance)?;
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
