#ifndef BN254_CUH
#define BN254_CUH

#include "common.cuh"
#include "ff.cuh"
#include "ec.cuh"

__device__ const ulong BN254_FR_MODULUS[4] = {
    0x43e1f593f0000001ul,
    0x2833e84879b97091ul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FR_NEG_TWO[4] = {
    0x43e1f593effffffful,
    0x2833e84879b97091ul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FR_R[4] = {
    0xac96341c4ffffffbul,
    0x36fc76959f60cd29ul,
    0x666ea36f7879462eul,
    0x0e0a77c19a07df2ful,
};

__device__ const ulong BN254_FR_R2[4] = {
    0x1bb8e645ae216da7ul,
    0x53fe3ab1e35c59e3ul,
    0x8c49833d53bb8085ul,
    0x0216d0b17f4e44a5ul,
};

__device__ const ulong Bn254_FR_INV = 0xc2e1f593effffffful;

typedef Field<4, BN254_FR_MODULUS, BN254_FR_NEG_TWO, BN254_FR_R, BN254_FR_R2, Bn254_FR_INV> Bn254FrField;

__device__ const ulong BN254_FP_MODULUS[4] = {
    0x3c208c16d87cfd47ul,
    0x97816a916871ca8dul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FP_NEG_TWO[4] = {
    0x3c208c16d87cfd45ul,
    0x97816a916871ca8dul,
    0xb85045b68181585dul,
    0x30644e72e131a029ul,
};

__device__ const ulong BN254_FP_R[4] = {
    0xd35d438dc58f0d9dul,
    0x0a78eb28f5c70b3dul,
    0x666ea36f7879462cul,
    0x0e0a77c19a07df2ful,
};

__device__ const ulong BN254_FP_R2[4] = {
    0xf32cfc5b538afa89ul,
    0xb5e71911d44501fbul,
    0x47ab1eff0a417ff6ul,
    0x06d89f71cab8351ful,
};

__device__ const ulong Bn254_FP_INV = 0x87d20782e4866389ul;

typedef Field<4, BN254_FP_MODULUS, BN254_FP_NEG_TWO, BN254_FP_R, BN254_FP_R2, Bn254_FP_INV> Bn254FpField;

typedef CurveAffine<Bn254FpField> Bn254G1Affine;
typedef Curve<Bn254FpField> Bn254G1;

#endif