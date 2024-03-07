use std::collections::BTreeMap;
use std::collections::HashSet;
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
use halo2_proofs::plonk::evaluation_gpu::Bop;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::transcript::EncodedChallenge;
use halo2_proofs::transcript::TranscriptWrite;

use crate::cuda::bn254::buffer_copy_with_shift;
use crate::cuda::bn254::extended_intt_after;
use crate::cuda::bn254::extended_prepare;
use crate::cuda::bn254::field_mul;
use crate::cuda::bn254::field_op_v2;
use crate::cuda::bn254::field_op_v3;
use crate::cuda::bn254::field_sub;
use crate::cuda::bn254::intt_raw;
use crate::cuda::bn254::msm;
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
use crate::device::cuda::to_result;
use crate::device::cuda::CudaBuffer;
use crate::device::cuda::CudaDevice;
use crate::device::cuda::CudaDeviceBufRaw;
use crate::device::Device as _;
use crate::device::DeviceResult;
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
            device.alloc_device_buffer::<F>(self.extended_size)
        } else {
            Ok(buf.unwrap())
        }
    }
}

pub(crate) fn analyze_expr_tree<F: FieldExt>(
    expr: &ProveExpression<F>,
) -> Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>> {
    let tree = expr.clone().flatten();
    let tree = tree
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    let limit = 20;
    let mut v = HashSet::new();

    let mut expr_group = vec![];
    let mut expr_groups = vec![];
    for (_, (units, coeff)) in tree.iter().enumerate() {
        let mut v_new = v.clone();
        let mut v_new_clean = HashSet::new();
        let mut muls_new = 0;
        for (unit, exp) in units {
            v_new.insert(unit.get_group());
            v_new_clean.insert(unit.get_group());
            muls_new += exp;
        }

        if v_new.len() > limit {
            v = v_new_clean;

            expr_groups.push(expr_group);
            expr_group = vec![(units.clone(), coeff.clone())];
        } else {
            v = v_new;
            expr_group.push((units.clone(), coeff.clone()));
        }
    }

    expr_groups.push(expr_group);
    expr_groups
}

pub fn _export_evaluate_h_gates<C: CurveAffine>(
    pk: &ProvingKey<C>,
    fixed: &[&[C::Scalar]],
    advice: &[&[C::Scalar]],
    instance: &[&[C::Scalar]],
    permutation_products: &[&[C::Scalar]],
    lookup_products: &[(
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
    )],
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
    lookup_products: &[(
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
    )],
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

        let timer = start_timer!(|| format!("vanishing msm {}", domain.quotient_poly_degree));
        for i in 0..domain.quotient_poly_degree as usize {
            let s_buf = unsafe {
                ManuallyDrop::new(CudaDeviceBufRaw {
                    ptr: h_buf
                        .ptr()
                        .offset((i * size * core::mem::size_of::<C::Scalar>()) as isize),
                    device: device.clone(),
                })
            };
            let commitment = msm(&device, &g_buf, &s_buf, size)?;
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
    lookup_products: &[(
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
        &[C::Scalar],
    )],
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
    if pk.ev.gpu_gates_expr.len() != 1 {
        println!("Multi-GPU detected, please set CUDA_VISIBLE_DEVICES to use one GPU");
        assert!(false);
    }
    let exprs = analyze_expr_tree(&pk.ev.gpu_gates_expr[0]);
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
    let l_active_buf = device.alloc_device_buffer_from_slice(&l_active_row.values[..])?;
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
    for (_i, (lookup, (permuted_input, permuted_table, input, table, z))) in pk
        .vk
        .cs
        .lookups
        .iter()
        .zip(lookup_products.iter())
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

        let input_buf = if input_deg > 1 {
            evaluate_prove_expr(device, &vec![e1], fixed, advice, instance, &mut ctx)?
        } else {
            let mut buf = ctx.alloc(device)?;
            device.copy_from_host_to_device(&buf, &input)?;

            let mut tmp_buf = ctx.alloc(device)?;
            intt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            ctx.extended_allocator.push(tmp_buf);

            let coeff = pick_from_buf::<C::Scalar>(device, &buf, 0, 0, ctx.size)?;
            let short = vec![coeff + beta];
            device.copy_from_host_to_device(&buf, &short[..])?;
            do_extended_ntt(device, &mut ctx, &mut buf)?;

            buf
        };
        let table_buf = if table_deg > 1 {
            evaluate_prove_expr(device, &vec![e2], fixed, advice, instance, &mut ctx)?
        } else {
            let mut buf = ctx.alloc(device)?;
            device.copy_from_host_to_device(&buf, &table)?;

            let mut tmp_buf = ctx.alloc(device)?;
            intt_raw(
                &device,
                &mut buf,
                &mut tmp_buf,
                &intt_pq_buf,
                &intt_omegas_buf,
                &intt_divisor_buf,
                k,
            )?;
            ctx.extended_allocator.push(tmp_buf);

            let coeff = pick_from_buf::<C::Scalar>(device, &buf, 0, 0, ctx.size)?;
            let short = vec![coeff + gamma];
            device.copy_from_host_to_device(&buf, &short[..])?;
            do_extended_ntt(device, &mut ctx, &mut buf)?;

            buf
        };

        let (permuted_input_buf, tmp0, stream0) =
            do_extended_ntt_v2_async(device, &mut ctx, permuted_input)?;
        let (permuted_table_buf, tmp1, stream1) =
            do_extended_ntt_v2_async(device, &mut ctx, permuted_table)?;
        let (z_buf, tmp2, stream2) = do_extended_ntt_v2_async(device, &mut ctx, z)?;

        unsafe {
            cuda_runtime_sys::cudaStreamSynchronize(stream0);
            cuda_runtime_sys::cudaStreamDestroy(stream0);
            ctx.extended_allocator.push(tmp0);
            cuda_runtime_sys::cudaStreamSynchronize(stream1);
            cuda_runtime_sys::cudaStreamDestroy(stream1);
            ctx.extended_allocator.push(tmp1);
            cuda_runtime_sys::cudaStreamSynchronize(stream2);
            cuda_runtime_sys::cudaStreamDestroy(stream2);
            ctx.extended_allocator.push(tmp2);
        }

        unsafe {
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
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;
            device.synchronize()?;
        }

        ctx.extended_allocator.push(input_buf);
        ctx.extended_allocator.push(table_buf);
        ctx.extended_allocator.push(permuted_input_buf);
        ctx.extended_allocator.push(permuted_table_buf);
        ctx.extended_allocator.push(z_buf);
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

fn flatten_lookup_expression<F: FieldExt>(
    input: &Vec<Expression<F>>,
    table: &Vec<Expression<F>>,
    beta: F,
    gamma: F,
    theta: F,
) -> [Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>; 2] {
    let mut expr_input = ProveExpression::<F>::from_expr(&input[0]);
    for input in input.iter().skip(1) {
        expr_input = ProveExpression::Scale(
            Box::new(expr_input),
            BTreeMap::from_iter([(0, theta)].into_iter()),
        );
        expr_input = ProveExpression::Op(
            Box::new(expr_input),
            Box::new(ProveExpression::<F>::from_expr(input)),
            Bop::Sum,
        );
    }

    expr_input = ProveExpression::Op(
        Box::new(expr_input),
        Box::new(ProveExpression::Y(BTreeMap::from_iter(
            [(0, beta)].into_iter(),
        ))),
        Bop::Sum,
    );

    let expr_input = expr_input
        .flatten()
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    let mut expr_table = ProveExpression::<F>::from_expr(&table[0]);
    for table in table.iter().skip(1) {
        expr_table = ProveExpression::Scale(
            Box::new(expr_table),
            BTreeMap::from_iter([(0, theta)].into_iter()),
        );
        expr_table = ProveExpression::Op(
            Box::new(expr_table),
            Box::new(ProveExpression::<F>::from_expr(table)),
            Bop::Sum,
        );
    }

    expr_table = ProveExpression::Op(
        Box::new(expr_table),
        Box::new(ProveExpression::Y(BTreeMap::from_iter(
            [(0, gamma)].into_iter(),
        ))),
        Bop::Sum,
    );

    let expr_table = expr_table
        .flatten()
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    [expr_input, expr_table]
}

fn do_extended_ntt_v2<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<CudaDeviceBufRaw> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        buf.unwrap()
    };
    device.copy_from_host_to_device::<F>(&buf, data)?;
    do_extended_ntt(device, ctx, &mut buf)?;

    Ok(buf)
}

fn do_extended_ntt_v2_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &[F],
) -> DeviceResult<(CudaDeviceBufRaw, CudaDeviceBufRaw, *mut CUstream_st)> {
    let buf = ctx.extended_allocator.pop();
    let mut buf = if buf.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
    } else {
        buf.unwrap()
    };
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

fn do_extended_ntt_pure_async<F: FieldExt>(
    device: &CudaDevice,
    ctx: &mut EvalHContext<F>,
    data: &mut CudaDeviceBufRaw,
    stream: Option<cudaStream_t>,
) -> DeviceResult<CudaDeviceBufRaw> {
    let tmp = ctx.extended_allocator.pop();
    let mut tmp = if tmp.is_none() {
        device.alloc_device_buffer::<F>(ctx.extended_size)?
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
            );

            to_result((), err, "fail to run field_op_batch_mul_sum")?;

            last_bufs = bufs;
        }
    }

    Ok(res)
}
