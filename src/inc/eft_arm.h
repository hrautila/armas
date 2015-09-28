
// Copyright (c), Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_EFT_ARM_H
#define __ARMAS_EFT_ARM_H 1

#ifdef __cplusplus
extern "C" {
#endif


/*
 * gcc predefines: 'gcc -dM -E -c - </dev/null'
 *
 * RPi:
 *      __arm__ = 1, __VFP_FP__ = 1, __ARM_PCS_VFP = 1
 *
 * RPi2: gcc -march=native
 *      __arm__ = 1, __ARM_FP = 12 __ARM_ARCH = 7, __ARM_NEON_FP = 4,
 *      __ARM_FP = 12, __ARM_FEATURE_SIMD32 = 1, __VFP_FP__ = 1
 *
 * RPi2: gcc -mfpu=neon
 *      __arm__ = 1, __ARM_FP = 12 __ARM_ARCH = 7, __ARM_NEON_FP = 4, __ARM_NEON = 1,
 *      __ARM_FP = 12, __ARM_FEATURE_SIMD32 = 1, __VFP_FP__ = 1
 *
 * RPi2 gcc -mfpu=neon-vfpv4:
 *      __arm__ = 1, __ARM_ARCH = 7, __ARM_FP = 14, __ARM_NEON = 1,
 *      __ARM_NEON_FP = 6, __ARM_FEATURE_SIMD32 = 1, __ARM_FEATURE_FMA = 1, __VFP_FP = 1
 *
 */

#if defined(__arm__) && __VFP_FP__ > 0
#define __have_accelerated_eft_versions 1

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


#if ! defined(__SIMD_LENGTH)
#if defined(__ARM_NEON)
#define __SIMD_LENGTH 128

#ifndef __HAVE_SIMD32X4
#define __HAVE_SIMD32X4 1
#endif

/*
 * Division by reciprocal estimation and two refinements.
 */
extern inline __attribute__((__always_inline__))
float32x4_t vdivq_f32(float32x4_t a, float32x4_t b)
{
    float32x4_t recp = vrecpeq_f32(b);
    recp = vmulq_f32(vrecpsq_f32(b, recp), recp);
    recp = vmulq_f32(vrecpsq_f32(b, recp), recp);
    return vmulq_f32(a, recp);
}

#else
#define __SIMD_LENGTH 0
#endif
#endif

/*
 * first 3 parameter are:
 *   t   instruction operand '.f32', '.f64'
 *   co  GCC asm constraint;
 *       w = VFP registers d0-15, d0-d31
 *       x = VFP registers d0-d7
 *       t = VFP register  s0-s31
 *   op  GCC asm operand code
 *       P = double precision VFP register
 *       q = NEON quad register
 *
 * see: http://hardwarebug.org/2010/07/06/arm-inline-asm-secrets/
 */

// basic error free transformation: x + y = a + b
#define __twosum_base(t, co, op, _type, _x, _y, _a, _b)                 \
    do {                                                                \
        register _type z, a0, b0;                                       \
        asm __volatile__                                                \
            (                                                           \
             "vmov" t " %"op"[a0], %"op"[ah]\n\t"                       \
             "vmov" t " %"op"[b0], %"op"[bh]\n\t"                       \
             "vadd" t " %"op"[x], %"op"[ah], %"op"[bh]\n\t"             \
             "vsub" t " %"op"[z], %"op"[x],  %"op"[a0]\n\t"             \
             "vsub" t " %"op"[y], %"op"[x],  %"op"[z]\n\t"              \
             "vsub" t " %"op"[y], %"op"[a0], %"op"[y]\n\t"              \
             "vsub" t " %"op"[z], %"op"[b0], %"op"[z]\n\t"              \
             "vadd" t " %"op"[y], %"op"[z],  %"op"[y]\n\t"              \
             : [x] "+"co (_x), [y] "+"co (_y), [z] "+"co (z),           \
               [a0] "+"co (a0), [b0] "+"co (b0)                         \
             : [ah]   co (_a), [bh]   co (_b));                         \
    } while (0)

// basic error free transformation: x + y = a + b iff |a| > |b|
#define __fastsum_base(t, co, op, type, _x, _y, _a, _b)                 \
    do {                                                                \
        register type z, a0, b0;                                        \
        asm __volatile__                                                \
            (                                                           \
             "vmov" t " %"op"[a0], %"op"[ah]\n\t"                       \
             "vmov" t " %"op"[b0], %"op"[bh]\n\t"                       \
             "vadd" t " %"op"[x], %"op"[ah], %"op"[bh]\n\t"             \
             "vsub" t " %"op"[z], %"op"[a0], %"op"[x]\n\t"              \
             "vadd" t " %"op"[y], %"op"[b0], %"op"[z]\n\t"              \
             : [x] "+"co (_x), [y] "="co (_y), [z] "+"co (z),           \
               [a0] "+"co (a0), [b0] "+"co (b0)                         \
             : [ah]   co (_a), [bh]   co (_b));                         \
    } while (0)


#define __split_base(t, co, op, type, _x, _y, _a, _fct)         \
    do {                                                        \
        register type z;                                        \
        asm __volatile__                                        \
            (                                                   \
             "vmul" t " %"op"[z], %"op"[a], %"op"[f]\n\t"       \
             "vsub" t " %"op"[x], %"op"[z], %"op"[a]\n\t"       \
             "vsub" t " %"op"[x], %"op"[z], %"op"[x]\n\t"       \
             "vsub" t " %"op"[y], %"op"[a], %"op"[x]\n\t"       \
             : [x] "+"co (_x), [y] "="co (_y), [z] "+"co (z)    \
             : [a]    co (_a), [f]    co (_fct));               \
    } while (0)

#define __extract_scalar_base(t, co, op, type, _x, _y, _p, _r)  \
    do {                                                        \
        register type q;                                        \
        asm __volatile__                                        \
            (                                                   \
             "vadd" t " %"op"[q], %"op"[p], %"op"[r]\n\t"       \
             "vsub" t " %"op"[x], %"op"[q], %"op"[r]\n\t"       \
             "vsub" t " %"op"[y], %"op"[p], %"op"[x]\n\t"       \
             : [x] "+"co (_x), [y] "="co (_y), [q] "+"co (q)    \
             : [r]    co (_r), [p]    co (_p));                 \
    } while (0)

#if defined(__ARM_FEATURE_FMA)
#define __twoprod_base(t, co, op, type, x, y, a, b, fct)                \
    do {                                                                \
        register type z;                                                \
        asm __volatile__                                                \
            (                                                           \
             "vmul" t " %"op"[ch], %"op"[ah], %"op"[bh]\n\t"            \
             "vneg" t " %"op"[cl], %"op"[ch]\n\t"                       \
             "vfma" t " %"op"[cl], %"op"[ah], %"op"[bh]\n\t"            \
             : [ch] "+"co (x), [cl] "+"co (y)                           \
             : [ah]    co (a), [bh]    co (b));                         \
    } while (0)

#define __twoprod_vec(t, type, x, y, a, b, fct) \
    do {                                        \
        x = vmulq_f32(a, b);                    \
        y = vnegq_f32(x);                       \
        y = vfmaq_f32(a, b, y);                \
    } while (0)


/*
 * Approx twodiv (with fused-multiply-add)
 *
 *   x = a / b                    (line 1)
 *   -- [v, w] = twoprod(x, b)
 *   v = x * b                    (line 2)
 *   w = FMA(x*b - v)             (lines 3-4)
 *   --
 *   y = (a - v - w) / b          (lines 5-7)
 */
#define __approx_twodiv_base(t, co, op, type, _x, _y, _a, _b, _fct)     \
    do {                                                                \
         register type v, w, a1, a2, b1, b2, z;                         \
         asm __volatile__                                               \
             (                                                          \
              "vdiv" t " %"op"[x], %"op"[a], %"op"[b]\n\t"              \
              "vmul" t " %"op"[v], %"op"[x], %"op"[b]\n\t"              \
              "vneg" t " %"op"[w]  %"op"[v]\n\t"                        \
              "vfma" t " %"op"[w], %"op"[x], %"op"[b]\n\t"              \
              "vsub" t " %"op"[v], %"op"[a], %"op"[v]\n\t"              \
              "vsub" t " %"op"[v], %"op"[v], %"op"[w]\n\t"              \
              "vdiv" t " %"op"[y], %"op"[v], %"op"[b]\n\t"              \
              : [x] "+"co (_x), [y] "+"co (_y), [z]  "+"co (z),         \
                [v] "+"co (v),  [w] "+"co (w)                           \
              : [a]  co (_a), [b]  co (_b),  [f] co (_fct));            \
     } while (0)

#define __approx_twodiv_vec(t, type, x, y, a, b, fct)  \
    do {                                        \
        register type v, w;                     \
        x = vdivq_f32(a, b);                    \
        v = vmulq_f32(x, b);                    \
        w = vnegq_f32(w);                       \
        w = vfmaq_f32(x, b, w);                 \
        v = vsubq_f32(a, v);                    \
        v = vsubq_f32(v, w);                    \
        y = vdivq_f32(v, b);                    \
    } while (0)

#else 
// without Fused-Multiply-Add
#define __twoprod_base(t, co, op, type, _x, _y, _a, _b, _fct)           \
    do {                                                                \
        register type z, a2, a1, b2, b1;                                \
        asm __volatile__                                                \
            (                                                           \
             "vmul" t " %"op"[z],  %"op"[ah], %"op"[f]\n\t"             \
             "vsub" t " %"op"[a1], %"op"[z],  %"op"[ah]\n\t"            \
             "vsub" t " %"op"[a1], %"op"[z],  %"op"[a1]\n\t"            \
             "vsub" t " %"op"[a2], %"op"[ah], %"op"[a1]\n\t"            \
             "vmul" t " %"op"[z],  %"op"[bh], %"op"[f]\n\t"             \
             "vsub" t " %"op"[b1], %"op"[z],  %"op"[bh]\n\t"            \
             "vsub" t " %"op"[b1], %"op"[z],  %"op"[b1]\n\t"            \
             "vsub" t " %"op"[b2], %"op"[bh], %"op"[b1]\n\t"            \
             "vmul" t " %"op"[x0], %"op"[ah], %"op"[bh]\n\t"            \
             "vmul" t " %"op"[z],  %"op"[b1], %"op"[a1]\n\t"            \
             "vsub" t " %"op"[y0], %"op"[x0], %"op"[z]\n\t"             \
             "vmul" t " %"op"[z],  %"op"[a2], %"op"[b1]\n\t"            \
             "vsub" t " %"op"[y0], %"op"[y0], %"op"[z]\n\t"             \
             "vmul" t " %"op"[z],  %"op"[a1], %"op"[b2]\n\t"            \
             "vsub" t " %"op"[y0], %"op"[y0], %"op"[z]\n\t"             \
             "vmul" t " %"op"[z],  %"op"[a2], %"op"[b2]\n\t"            \
             "vsub" t " %"op"[y0], %"op"[z],  %"op"[y0]\n\t"            \
             : [x0] "+"co (_x), [y0] "+"co (_y), [z]  "+"co (z),        \
               [a1] "+"co (a1), [a2] "+"co (a2),                        \
               [b1] "+"co (b1), [b2] "+"co (b2)                         \
             : [ah]  "x" (_a), [bh]  "x" (_b),  [f] "x" (_fct));        \
    } while (0)

#define __twoprod_vec(t, type, _x, _y, _a, _b, _fct)                    \
    do {                                                                \
        register type z, f0, a2, a1, b2, b1;                            \
        f0 = _fct;                                                      \
        z  = vmulq_f32(_a, f0);						\
        a1 = vsubq_f32( z, _a);						\
        a1 = vsubq_f32( z, a1);						\
        a2 = vsubq_f32(_a, a1);						\
        z  = vmulq_f32(_b, f0);						\
        b1 = vsubq_f32( z, _b);						\
        b1 = vsubq_f32( z, b1);						\
        b2 = vsubq_f32(_b, b1);						\
        _x = vmulq_f32(_a, _b);						\
        z  = vmulq_f32(a1, b1);						\
        _y = vsubq_f32(_x, z);                                          \
        z  = vmulq_f32(a2, b1);						\
        _y = vsubq_f32(_y, z);                                          \
        z  = vmulq_f32(a1, b2);						\
        _y = vsubq_f32(_y, z);                                          \
        z  = vmulq_f32(a2, b2);						\
        _y = vsubq_f32(z, _y);						\
    } while (0)


/*
 * Approx twodiv (without fused-multiply-add)
 *
 *   x = a / b                          (line 1)
 *   -- [v, w] = twoprod(x, b)
 *   [a2, a1] = split(x)                (lines 2-5)
 *   [b2, b1] = split(b)                (lines 6-9)
 *   v = x*b                            (line  10)
 *   w = ((v - a1*b1) - a2*b1) - a1*b2  (lines 11-16)
 *   w = a2*b2 - w                      (lines 17-18)
 *   --
 *   y = (a - v - w) / b                (lines 19-21)
 */
#define __approx_twodiv_base(t, co, op, type, _x, _y, _a, _b, _fct)     \
         do {                                                           \
             register type v, w, a1, a2, b1, b2, z;                     \
             asm __volatile__                                           \
                 (                                                      \
                  "vdiv" t " %"op"[x],  %"op"[a], %"op"[b]\n\t"         \
                  "vmul" t " %"op"[z],  %"op"[x], %"op"[f]\n\t"         \
                  "vsub" t " %"op"[a1], %"op"[z], %"op"[a]\n\t"         \
                  "vsub" t " %"op"[a1], %"op"[z], %"op"[a1]\n\t"        \
                  "vsub" t " %"op"[a2], %"op"[x], %"op"[a1]\n\t"        \
                  "vmul" t " %"op"[z],  %"op"[b], %"op"[f]\n\t"         \
                  "vsub" t " %"op"[b1], %"op"[z], %"op"[b]\n\t"         \
                  "vsub" t " %"op"[b1], %"op"[z], %"op"[b1]\n\t"        \
                  "vsub" t " %"op"[b2], %"op"[b], %"op"[b1]\n\t"        \
                  "vmul" t " %"op"[v],  %"op"[b],  %"op"[a]\n\t"        \
                  "vmul" t " %"op"[z],  %"op"[b1], %"op"[a1]\n\t"       \
                  "vsub" t " %"op"[w],  %"op"[v],  %"op"[z]\n\t"        \
                  "vmul" t " %"op"[z],  %"op"[b1], %"op"[a2]\n\t"       \
                  "vsub" t " %"op"[w],  %"op"[w],  %"op"[z]\n\t"        \
                  "vmul" t " %"op"[z],  %"op"[b2], %"op"[a1]\n\t"       \
                  "vsub" t " %"op"[w],  %"op"[w],  %"op"[z]\n\t"        \
                  "vmul" t " %"op"[z],  %"op"[b2], %"op"[a2]\n\t"       \
                  "vsub" t " %"op"[w],  %"op"[z],  %"op"[w]\n\t"        \
                  "vsub" t " %"op"[v],  %"op"[a], %"op"[v]\n\t"         \
                  "vsub" t " %"op"[v],  %"op"[v], %"op"[w]\n\t"         \
                  "vdiv" t " %"op"[y],  %"op"[v], %"op"[b]\n\t"         \
                  : [x]  "+"co (_x), [y]  "+"co (_y), [z]  "+"co (z),   \
                    [a1] "+"co (a1), [a2] "+"co (a2),                   \
                    [b1] "+"co (b1), [b2] "+"co (b2),                   \
                    [v]  "+"co (v),  [w]  "+"co (w)                     \
                  : [a] co  (_a), [b] co  (_b),  [f] co (_fct));        \
         } while (0)

#define __approx_twodiv_vec(t, type, _x, _y, _a, _b, _fct)     \
    do {                                                \
        register type z, f0, a2, a1, b2, b1, v, w;      \
        f0 = _fct;                                      \
        _x = vdivq_f32(_a, _b);                         \
        z  = vmulq_f32(_x, f0);                         \
        a1 = vsubq_f32( z, _a);                         \
        a1 = vsubq_f32( z, a1);                         \
        a2 = vsubq_f32(_x, a1);                         \
        z  = vmulq_f32(_b, f0);                         \
        b1 = vsubq_f32( z, _b);                         \
        b1 = vsubq_f32( z, b1);                         \
        b2 = vsubq_f32(_b, b1);                         \
        v  = vmulq_f32(_a, _b);                         \
        z  = vmulq_f32(a1, b1);                         \
        w  = vsubq_f32( v, z);                          \
        z  = vmulq_f32(a2, b1);                         \
        w  = vsubq_f32( w, z);                          \
        z  = vmulq_f32(a1, b2);                         \
        w  = vsubq_f32( w, z);                          \
        z  = vmulq_f32(a2, b2);                         \
        w  = vsubq_f32( z,  w);                         \
        v  = vsubq_f32(_a, v);                          \
        v  = vsubq_f32( v, w);                          \
        _y = vdivq_f32( v, _b);                         \
    } while (0)


#endif

#define __twosum_base_f32(_x, _y, _a, _b) \
    __twosum_base(".f32", "x", "", float, _x, _y, _a, _b)

#define __fastsum_base_f32(_x, _y, _a, _b) \
    __fastsum_base(".f32", "x", "", float, _x, _y, _a, _b)

#define __twoprod_base_f32(_x, _y, _a, _b) \
    __twoprod_base(".f32", "x", "", float, _x, _y, _a, _b, __factor_f32)

#define __approx_twodiv_base_f32(_x, _y, _a, _b) \
    __approx_twodiv_base(".f32", "x", "", float, _x, _y, _a, _b, __factor_f32)

#define __split_base_f32(_x, _y, _a) \
    __split_base(".f32", "x", "", float, _x, _y, _a, __factor_f32)

#define __extract_scalar_base_f32(_x, _y, _p, _r)     \
    __extract_scalar_base(".f32", "x", "", float, _x, _y, _p, _r)

// Double precision float

#define __twosum_base_f64(_x, _y, _a, _b)                       \
             __twosum_base(".f64", "w", "P", double, _x, _y, _a, _b)

#define __fastsum_base_f64(_x, _y, _a, _b)                      \
             __fastsum_base(".f64", "w", "P", double, _x, _y, _a, _b)

#define __twoprod_base_f64(_x, _y, _a, _b)                              \
             __twoprod_base(".f64", "w", "P", double, _x, _y, _a, _b, __factor_f64)

#define __approx_twodiv_base_f64(_x, _y, _a, _b)                        \
             __approx_twodiv_base(".f64", "w", "P", double, _x, _y, _a, _b, __factor_f64)

#define __split_base_f64(_x, _y, _a)                                    \
             __split_base(".f64", "w", "P", double, _x, _y, _a, __factor_f64)

#define __extract_scalar_base_f64(_x, _y, _p, _r)                       \
             __extract_scalar_base(".f64", "w", "P", double, _x, _y, _p, _r)

#if __SIMD_LENGTH >= 128

#ifdef __HAVE_SIMD32X4

static const float32x4_t __factor_f32x4 = (const float32x4_t){
    __FACT32, __FACT32, __FACT32, __FACT32
};

#define __twosum_base_f32x4(_x, _y, _a, _b)                             \
             __twosum_base(".f32", "t", "q", float32x4_t, _x, _y, _a, _b)

#define __fastsum_base_f32x4(_x, _y, _a, _b)                            \
             __fastsum_base(".f32", "t", "q", float32x4_t, _x, _y, _a, _b)

#define __twoprod_base_f32x4(_x, _y, _a, _b)                            \
             __twoprod_vec("q_f32", float32x4_t, _x, _y, _a, _b, __factor_f32x4)

#define __approx_twodiv_base_f32x4(_x, _y, _a, _b)                      \
             __approx_twodiv_vec("q_f32", float32x4_t, _x, _y, _a, _b, __factor_f32x4);

#define __split_base_f32x4(_x, _y, _a)                                  \
             __split_base(".f32", "t", "q", float32x4_t, _x, _y, _a, __factor_f32x4);

#define __extract_scalar_base_f32x4(_x, _y, _p, _r)                     \
             __extract_scalar_base(".f32", "t", "q", float32x4_t, _x, _y, _p, _r)


#endif // __HAVE_SIMD32X4

#ifdef __HAVE_SIMD64X2
// no support
#endif

#endif // __SIMD_LENGTH >= 128

#if __SIMD_LENGTH >= 256

#ifdef __HAVE_SIMD32X8
#endif

#ifdef __HAVE_SIMD64X4
#endif

#endif // __SIMD_LENGTH >= 256

#endif  // __arm__

#ifdef __cplusplus
}
#endif

#endif  // __ARMAS_EFT_ARM_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

