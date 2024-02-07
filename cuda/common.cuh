#ifndef COMMON_CUH
#define COMMON_CUH
__device__ ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d)
{
  ulong lo, hi;
  asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
      "madc.hi.u64 %1, %2, %3, 0;\r\n"
      "add.cc.u64 %0, %0, %5;\r\n"
      "addc.u64 %1, %1, 0;\r\n"
      : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
  *d = hi;
  return lo;
}

__device__ void add_u64x4(const ulong *a, const ulong *b, ulong *c)
{
  asm("add.cc.u64  %0, %4, %8;\r\n"
      "addc.cc.u64 %1, %5, %9;\r\n"
      "addc.cc.u64 %2, %6, %10;\r\n"
      "addc.u64    %3, %7, %11;\r\n":
      "=l"(c[0]),"=l"(c[1]),"=l"(c[2]),"=l"(c[3]):
      "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
      "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3])
  );
}

__device__ void sub_u64x4(const ulong *a, const ulong *b, ulong *c)
{
  asm("sub.cc.u64  %0, %4, %8;\r\n"
      "subc.cc.u64 %1, %5, %9;\r\n"
      "subc.cc.u64 %2, %6, %10;\r\n"
      "subc.u64  %3, %7, %11;\r\n":
      "=l"(c[0]),"=l"(c[1]),"=l"(c[2]),"=l"(c[3]):
      "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
      "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3])
  );
}

__device__ uint sub_u64x4_with_borrow(const ulong *a, const ulong *b, ulong *c)
{
  uint ret;
  asm("sub.cc.u64  %0, %5, %9;\r\n"
      "subc.cc.u64 %1, %6, %10;\r\n"
      "subc.cc.u64 %2, %7, %11;\r\n"
      "subc.cc.u64 %3, %8, %12;\r\n"
      "subc.u32    %4,  0,   0;\r\n":
      "=l"(c[0]),"=l"(c[1]),"=l"(c[2]),"=l"(c[3]),"=r"(ret):
      "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
      "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3])
  );
  return ret;
}
#endif