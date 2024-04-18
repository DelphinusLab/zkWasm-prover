# Introduction
This repository serves as an alternative backend for zkcircuits written in the halo2 frontend. It assumes GPU utilization as the main proving unit and implements the KZG backend for GPU arithmetic. This tool is compatible with the frontend of [DelphinusLab's halo2-gpu-specific](https://github.com/DelphinusLab/halo2-gpu-specific), which is derived from halo2. Thus, any circuits written in the halo2 frontend should be able to be proven in this prover (but much faster). Currently, this prover is mainly used for generating proofs for [ZKWASM](https://github.com/DelphinusLab/zkWasm) and its continuation, batcher.

# Usage
To use this prover, you need to prepare two things: the pkey of your circuit and the synthesizer. Prepare all your advisors and then feed them to the prover. Currently, both GWC and Shplonk modes are supported.
## Preparing your advices
```

#[cfg(feature = "perf")]
let advices = {
    use halo2_proofs::plonk::generate_advice_from_synthesize;
    use std::sync::Arc;
    use zkwasm_prover::prepare_advice_buffer;

    let mut advices = Arc::new(prepare_advice_buffer(pkey, false));

    generate_advice_from_synthesize(
        &params,
        pkey,
        c,
        &instances,
        &unsafe { Arc::get_mut_unchecked(&mut advices) }
            .iter_mut()
            .map(|x| (&mut x[..]) as *mut [_])
            .collect::<Vec<_>>()[..],
    );

    advices
};
```


## Generate Proofs via prepared advices
```
#[cfg(feature = "perf")]
macro_rules! perf_gen_proof {
    ($transcript: expr, $schema: expr) => {{
        use zkwasm_prover::create_proof_from_advices_with_gwc;
        use zkwasm_prover::create_proof_from_advices_with_shplonk;

        match $schema {
            OpenSchema::GWC => create_proof_from_advices_with_gwc(
                &params,
                pkey,
                &instances,
                advices,
                &mut $transcript,
            )
            .expect("proof generation should not fail"),
            OpenSchema::Shplonk => create_proof_from_advices_with_shplonk(
                &params,
                pkey,
                &instances,
                advices,
                &mut $transcript,
            )
            .expect("proof generation should not fail"),
        }
    }};
}
```
