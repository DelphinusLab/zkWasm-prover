#ifndef COMMON_CUH
#define COMMON_CUH

#include <assert.h>

__forceinline__ __device__ void add_u64x4(ulong *out, const ulong *a, const ulong *b)
{
  asm("add.cc.u64  %0, %4, %8; \n\t"
      "addc.cc.u64 %1, %5, %9; \n\t"
      "addc.cc.u64 %2, %6, %10;\n\t"
      "addc.u64    %3, %7, %11;\n\t" :                         //
      "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3]) : //
      "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

__forceinline__ __device__ void sub_u64x4(ulong *out, const ulong *a, const ulong *b)
{
  asm("sub.cc.u64  %0, %4, %8; \n\t"
      "subc.cc.u64 %1, %5, %9; \n\t"
      "subc.cc.u64 %2, %6, %10;\n\t"
      "subc.u64    %3, %7, %11;\n\t" :                       //
      "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3]) //
      : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
        "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

__forceinline__ __device__ uint sub_u64x4_with_borrow(ulong *out, const ulong *a, const ulong *b)
{
  uint ret;
  asm("sub.cc.u64  %0, %5, %9; \n\t"
      "subc.cc.u64 %1, %6, %10;\n\t"
      "subc.cc.u64 %2, %7, %11;\n\t"
      "subc.cc.u64 %3, %8, %12;\n\t"
      "subc.u32    %4,  0,   0;\n\t" :                                    //
      "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3]), "=r"(ret) : //
      "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
  return ret;
}

// a[0..5] += u * m[0..4]
__device__ void mul_add_u64x4(
    ulong *a0, ulong *a1, ulong *a2, ulong *a3, ulong *a4,
    const ulong *m, ulong u, ulong *last_carry)
{
  asm("{\n\t"
      ".reg .u64 t1;\n\t"
      ".reg .u64 t2;\n\t"
      "mad.lo.cc.u64   %0, %10, %6, %0;  \n\t"    //
      "madc.hi.u64     t1, %10, %6, 0;   \n\t"    // (carry, a[i + 0]) = u * m[0] + a[i + 0]
                                                  //
      "mad.lo.cc.u64   %1, %10, %7, %1;  \n\t"    //
      "madc.hi.u64     t2, %10, %7, 0;   \n\t"    //
      "add.cc.u64      %1, %1,  t1;      \n\t"    // (carry, a[i + 1]) = u * m[1] + a[i + 1] + carry
                                                  //
      "madc.lo.cc.u64  %2, %10, %8, %2;  \n\t"    //
      "madc.hi.u64     t1, %10, %8, 0;   \n\t"    //
      "add.cc.u64      %2, %2,  t2;      \n\t"    // (carry, a[i + 2]) = u * m[2] + a[i + 2] + carry
                                                  //
      "madc.lo.cc.u64  %3, %10, %9, %3;  \n\t"    //
      "madc.hi.u64     t2, %10, %9, 0;   \n\t"    //
      "add.cc.u64      %3, %3,  t1;      \n\t"    // (carry, a[i + 3]) = u * m[3] + a[i + 3] + carry
                                                  //
      "addc.cc.u64     %4, %4,  %5;      \n\t"    // 
      "addc.u64        %5, 0,   0;       \n\t"    //
      "add.cc.u64      %4, %4,  t2;      \n\t"    // (carry, a[i + 4]) += carry + last_carray
                                                  //
      "addc.u64        %5, %5,  0;        \n\t"     // return carry
      "}" :                                       //
      "+l"(*a0),                                  // 0
      "+l"(*a1), "+l"(*a2), "+l"(*a3), "+l"(*a4), // 1, 2, 3, 4
      "+l"(*last_carry) :                         // 5
      "l"(m[0]), "l"(m[1]), "l"(m[2]), "l"(m[3]), // 6, 7, 8, 9
      "l"(u)                                      // 10
  );
}

__forceinline__ __device__ void mul_u64x4(
    ulong *a0, ulong *a1, ulong *a2, ulong *a3,
    ulong *a4, ulong *a5, ulong *a6, ulong *a7,
    const ulong *l, const ulong *r)
{
  ulong carry = 0;
  mul_add_u64x4(a0, a1, a2, a3, a4, l, r[0], &carry);
  mul_add_u64x4(a1, a2, a3, a4, a5, l, r[1], &carry);
  mul_add_u64x4(a2, a3, a4, a5, a6, l, r[2], &carry);
  mul_add_u64x4(a3, a4, a5, a6, a7, l, r[3], &carry);
}

// u = a[i] * inv
// a += u * a[0..4] * (2 ** 64 ** i)
__forceinline__ __device__ void mont_reduce_u64x8_round(
    ulong *a0, ulong *a1, ulong *a2, ulong *a3, ulong *a4,
    const ulong *m, ulong inv, ulong *last_carry)
{
  ulong u = *a0 * inv;
  mul_add_u64x4(a0, a1, a2, a3, a4, m, u, last_carry);
}

__forceinline__ __device__ void mont_reduce_u64x8(
    ulong *a0, ulong *a1, ulong *a2, ulong *a3,
    ulong *a4, ulong *a5, ulong *a6, ulong *a7,
    const ulong *m, ulong inv)
{
  ulong carry = 0;
  mont_reduce_u64x8_round(a0, a1, a2, a3, a4, m, inv, &carry);
  mont_reduce_u64x8_round(a1, a2, a3, a4, a5, m, inv, &carry);
  mont_reduce_u64x8_round(a2, a3, a4, a5, a6, m, inv, &carry);
  mont_reduce_u64x8_round(a3, a4, a5, a6, a7, m, inv, &carry);
}
#endif