use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
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
        let pick_host_buffer = |unit: &'b _| match unit {
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

        let (sw, stream) = CudaStreamWrapper::new_with_inner();
        let res = self.alloc_zeroed(&sw)?;
        sw.sync();
        drop((sw, stream));

        let mut prev_stream_and_pending_pending_buffers: Option<(
            CudaStreamWrapper,
            Vec<CudaDeviceBufRaw>,
        )> = None;
        let mut prev_extended_buffers: HashMap<_, CudaDeviceBufRaw> = HashMap::new();

        for expr in exprs.iter() {
            let mut last_stream_and_tmp = None;
            let mut curr_extended_buffers = HashMap::new();

            let unit_ids = expr
                .iter()
                .flat_map(|(units, _)| units.iter().map(|x| x.0.get_group()))
                .collect::<HashSet<_>>();

            for unit_id in unit_ids {
                if let Some(buffer) = prev_extended_buffers.remove(&unit_id) {
                    curr_extended_buffers.insert(unit_id, buffer);
                }
            }

            let mut sync_prev = true;

            for (unit, _) in expr.iter().flat_map(|(units, _)| units.iter()) {
                let unit_id = unit.get_group();
                if !curr_extended_buffers.contains_key(&unit_id) {
                    let (src, _) = pick_host_buffer(unit);
                    let (buf, stream_and_tmp) = self.copy_and_extended_ntt_async(src)?;
                    curr_extended_buffers.insert(unit_id, buf);
                    self.extended_ntt_wait_opt(last_stream_and_tmp)?;
                    last_stream_and_tmp = Some(stream_and_tmp);

                    if sync_prev {
                        // Clear prev_extended_buffers after the first new extended buffer task.
                        // Sync previous round before reuse extended buffer.
                        prev_stream_and_pending_pending_buffers
                            .as_ref()
                            .map(|x: &(CudaStreamWrapper, _)| x.0.sync());
                        prev_stream_and_pending_pending_buffers
                            .as_mut()
                            .map(|(_, x)| x.clear());
                        prev_extended_buffers.clear();
                        sync_prev = false;
                    }
                }
            }

            // sync previous round before reuse extended buffer
            prev_stream_and_pending_pending_buffers
                .as_ref()
                .map(|x: &(CudaStreamWrapper, _)| x.0.sync());
            prev_extended_buffers.clear();
            drop(prev_stream_and_pending_pending_buffers);

            let (sw, stream) = CudaStreamWrapper::new_with_inner();
            let coeffs = self.eval_ys(expr.iter().map(|(_, coeff)| coeff));
            let coeffs_buf = self
                .device
                .alloc_device_buffer_from_slice_async(&coeffs[..], stream)?;

            let mut group = vec![];
            let mut rots = vec![];

            for (i, (units, _)) in expr.iter().enumerate() {
                group.push(unsafe {
                    coeffs_buf
                        .ptr()
                        .offset((i * core::mem::size_of::<F>()) as isize)
                });
                for (unit, exp) in units {
                    let id = unit.get_group();
                    let (_, rot) = pick_host_buffer(unit);
                    for _ in 0..*exp {
                        group.push(curr_extended_buffers.get(&id).unwrap().ptr());
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

            sw.sync();
            drop((sw, stream));

            let (sw, mut pending_queue) = if let Some((sw, tmp)) = last_stream_and_tmp {
                (sw, vec![tmp])
            } else {
                (CudaStreamWrapper::new(), vec![])
            };

            unsafe {
                let err = field_op_batch_mul_sum(
                    res.ptr(),
                    group_buf.ptr(),
                    rots_buf.ptr(),
                    group.len() as i32,
                    self.extended_size as i32,
                    (&sw).into(),
                );

                to_result((), err, "fail to run field_op_batch_mul_sum")?;
            }

            prev_extended_buffers = curr_extended_buffers;
            pending_queue.append(&mut vec![group_buf, rots_buf, coeffs_buf]);
            prev_stream_and_pending_pending_buffers = Some((sw, pending_queue));
        }

        prev_stream_and_pending_pending_buffers.as_mut().map(|x| {
            x.1.append(&mut prev_extended_buffers.into_values().collect())
        });

        Ok((res, prev_stream_and_pending_pending_buffers.unwrap()))
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
    ) -> DeviceResult<Vec<CudaDeviceBufRaw>> {
        let mut res = vec![];
        let mut prev_stream_and_tmp = None;
        for data in batch_data {
            let (buf, stream_and_tmp) = self.copy_and_extended_ntt_async(data)?;
            res.push(buf);
            self.extended_ntt_wait_opt(prev_stream_and_tmp)?;
            prev_stream_and_tmp = Some(stream_and_tmp);
        }
        self.extended_ntt_wait_opt(prev_stream_and_tmp)?;
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

    fn extended_ntt_wait_opt(
        &mut self,
        last: Option<(CudaStreamWrapper, CudaDeviceBufRaw)>,
    ) -> DeviceResult<()> {
        if let Some((stream, tmp_buf)) = last {
            stream.sync();
            self.free(tmp_buf);
        }
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
    permutation_poly: &[&[C::Scalar]],
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
        permutation_poly,
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
    permutation_poly: &[&[C::Scalar]],
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
    let (h_buf, (sw, buf)) =
        ctx.evaluate_prove_expr_with_async_ntt(&exprs, fixed, advice, instance)?;
    sw.sync();
    drop(sw);
    drop(buf);
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

        let extended_p_buf = ctx.batch_copy_and_extended_ntt_async(permutation_products)?;
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
                .zip(permutation_poly.chunks(chunk_len))
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

                let mut prev_sw = CudaStreamWrapper::new();
                let mut prev_buffer = vec![];

                for (value, permutation) in columns
                    .iter()
                    .map(|&column| match column.column_type() {
                        Any::Advice => &advice[column.index()],
                        Any::Fixed => &fixed[column.index()],
                        Any::Instance => &instance[column.index()],
                    })
                    .zip(polys)
                {
                    let mut l_res = ctx.alloc()?;
                    let mut r_res = ctx.alloc()?;
                    let p_coset_buf = ctx.alloc()?;

                    let (sw, stream) = CudaStreamWrapper::new_with_inner();
                    device.copy_from_host_to_device_async(&l_res, value, stream)?;
                    device.copy_from_device_to_device_async::<C::Scalar>(
                        &r_res, 0, &l_res, 0, ctx.size, stream,
                    )?;
                    sw.sync();

                    let (copy_sw, copy_stream) = CudaStreamWrapper::new_with_inner();
                    let (calc_sw, calc_stream) = CudaStreamWrapper::new_with_inner();

                    let diff_buffer = device
                        .alloc_device_buffer_from_slice_async(&[gamma, curr_delta], copy_stream)?;
                    copy_sw.sync();

                    prev_sw.sync();
                    for buffer in prev_buffer {
                        ctx.free(buffer);
                    }

                    device.copy_from_host_to_device_async(
                        &p_coset_buf,
                        &permutation[..],
                        copy_stream,
                    )?;

                    ctx.extended_ntt_prepare(&mut r_res, Some(&calc_sw))?;

                    field_op::<C::Scalar>(
                        &device,
                        &r_res,
                        &r_res,
                        &diff_buffer,
                        2,
                        FieldOp::Add,
                        Some(calc_stream),
                    )?;
                    let tmp1 = ctx.extended_ntt_pure(&mut r_res, Some(&calc_sw))?;
                    field_op::<C::Scalar>(
                        &device,
                        &r_acc,
                        &r_acc,
                        &r_res,
                        ctx.extended_size,
                        FieldOp::Mul,
                        Some(calc_stream),
                    )?;

                    copy_sw.sync();
                    permutation_eval_h_l(
                        &device,
                        &l_res,
                        &beta_buf,
                        &gamma_buf,
                        &p_coset_buf,
                        ctx.size,
                        Some(calc_stream),
                    )?;

                    let tmp2 = ctx.extended_ntt_async(&mut l_res, &calc_sw)?;
                    field_op::<C::Scalar>(
                        &device,
                        &l_acc,
                        &l_acc,
                        &l_res,
                        ctx.extended_size,
                        FieldOp::Mul,
                        Some(calc_stream),
                    )?;

                    curr_delta *= &C::Scalar::DELTA;

                    prev_buffer = vec![l_res, r_res, tmp1, tmp2, p_coset_buf];
                    prev_sw = calc_sw;
                }

                prev_sw.sync();
                for buffer in prev_buffer {
                    ctx.free(buffer);
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

    let mut prev_stream_and_buffers: Option<(CudaStreamWrapper, _)> = None;
    for (_i, (lookup, (inputs_sets_host, table_host, multiplicity_host, z_set_host))) in pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookup_products.into_iter())
        .enumerate()
    {
        let mut input_deg = 1;
        lookup.input_expressions_sets.iter().for_each(|set| {
            set.0.iter().for_each(|input| {
                input_deg = input_deg.max(get_expr_degree(&input));
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

        let (multiplicity_buf, m_stream_and_tmp) =
            ctx.copy_and_extended_ntt_async(multiplicity_host)?;

        prev_stream_and_buffers.map(|(sw, buffers)| {
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

                pendings_product.0.sync();
                pendings_sum.0.sync();
                drop([pendings_product, pendings_sum]);
            } else {
                let (sw, stream) = CudaStreamWrapper::new_with_inner();
                let product_buf = ctx.alloc()?;
                let product_sum_buf = ctx.alloc()?;

                let mut inputs_beta_sets = vec![];

                for input in inputs_sets_host[i].iter() {
                    let (buf, stream_and_tmp) = logup_transform_extend_coset_async(
                        device,
                        &input,
                        &beta_buf,
                        &intt_pq_buf,
                        &intt_omegas_buf,
                        &intt_divisor_buf,
                        &mut ctx,
                    )?;
                    inputs_beta_sets.push(buf);

                    ctx.extended_ntt_wait(last_waiting)?;
                    last_waiting = stream_and_tmp;
                }

                ctx.extended_ntt_wait(last_waiting)?;

                logup_eval_h_inputs_product_sum(
                    device,
                    &product_buf,
                    &product_sum_buf,
                    &inputs_beta_sets,
                    ctx.extended_size,
                    Some(stream),
                )?;
                inputs_products_bufs.push(product_buf);
                inputs_products_sum_bufs.push(product_sum_buf);

                let (z_buf, stream_and_tmp) =
                    ctx.copy_and_extended_ntt_async(&z_set_host[i][..])?;

                sw.sync();
                last_waiting = stream_and_tmp;
                zs_bufs.push(z_buf);
            }
        }

        let table_buf = if table_deg > 1 {
            let (buf, pendings) =
                ctx.evaluate_prove_expr_with_async_ntt(&vec![table_expr], fixed, advice, instance)?;
            ctx.extended_ntt_wait(last_waiting)?;
            pendings.0.sync();
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

            let sets_len = lookup.input_expressions_sets.len();
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

            prev_stream_and_buffers = Some((
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

    prev_stream_and_buffers.map(|(sw, buffers)| {
        sw.sync();
        for buffer in buffers {
            ctx.free(buffer);
        }
    });
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
