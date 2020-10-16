
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_EFT_X86_H
#define __ARMAS_EFT_X86_H 1

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __SPLIT_FACTOR
#define __SPLIT_FACTOR 1
#define __FACT32 ((1 << 14) + 1)
#define __FACT64 ((1 << 27) + 1)
static const float  __factor_f32 = __FACT32;
static const double __factor_f64 = __FACT64;
#endif


#if defined(__x86_64__)

#include <immintrin.h>

#if defined(__AVX__)
#define __have_accelerated_eft_versions 1


static const __m128 __fact_f32x4 = (const __m128){
    __FACT32, __FACT32, __FACT32, __FACT32
};

static const __m128d __fact_f64x2 = (const __m128d){
    __FACT64, __FACT64
};

static const __m256 __fact_f32x8 = (const __m256){
    __FACT32, __FACT32, __FACT32, __FACT32, __FACT32, __FACT32, __FACT32, __FACT32
};

static const __m256d __fact_f64x4 = (const __m256d){
    __FACT64, __FACT64, __FACT64, __FACT64
};

/*
 * These x86_64 extended precision macros require at least AVX instruction set.
 * If fused-multiply-add is available it is used. 
 */

/*
 * error free summation; twosum
 *   x = fl(ah + bh)
 *   z = fl(ah - x)
 *   y = fl((ah - (x - z)) + (bh - z))
 *       y = fl(x - z)
 *       y = fl(ah - y)
 *       z = fl(bh - z)
 *       y = fl(y + z)
 */

/*
 * Standard error free transformation for summation: x + y = a + b;
 *
 * The additional registry moves are to protect against calls like
 *   x + y = x + b or x + y = b + x;
 */
#define __twosum_base(t, movi, type, _x, _y, _a, _b)                    \
    do {                                                                \
        register type z, a0, b0;                                        \
        asm __volatile__                                                \
            (                                                           \
             movi " %[ah], %[a0]\n\t"                                   \
             movi " %[bh], %[b0]\n\t"                                   \
             "vadd" t " %[ah], %[bh], %[x]\n\t"                         \
             "vsub" t " %[a0], %[x],  %[z]\n\t"                         \
             "vsub" t " %[z],  %[x],  %[y]\n\t"                         \
             "vsub" t " %[y],  %[a0], %[y]\n\t"                         \
             "vsub" t " %[z],  %[b0], %[z]\n\t"                         \
             "vadd" t " %[y],  %[z],  %[y]\n\t"                         \
             : [x]  "+x" (_x), [y]  "+x" (_y), [z] "+x" (z),            \
               [a0] "+x" (a0), [b0] "+x" (b0)                           \
             : [ah]  "x" (_a), [bh]  "x" (_b));                         \
    } while (0)


/*
 * Fast error free transformation for summation: x + y = a + b and |a| > |b|
 *
 * The additional registry moves are to protect against calls like
 *   x + y = x + b or x + y = a + x;
 */
#define __fastsum_base(t, movi, type, _x, _y, _a, _b)                   \
    do {                                                                \
         register type z, a0, b0;                                       \
         asm __volatile__                                               \
            (                                                           \
             movi " %[ah], %[a0]\n\t"                                   \
             movi " %[bh], %[b0]\n\t"                                   \
             "vadd" t " %[ah], %[bh], %[x]\n\t"                         \
             "vsub" t " %[x],  %[a0], %[z]\n\t"                         \
             "vadd" t " %[z],  %[b0], %[y]\n\t"                         \
             : [x]  "+x" (_x), [y]  "=x" (_y), [z] "+x" (z),            \
               [a0] "+x" (a0), [b0] "+x" (b0)                           \
             : [ah]  "x" (_a), [bh]  "x" (_b));                         \
    } while (0)


/*
 * Split value to high and low parts such that x + y = a. Spliting uses
 * constant _fct that has value ((1<< NBITS)+1) where NBITS is 14 for single
 * precision floats and 27 for double precision IEEE floating point numbers.
 * Algorithm originally from Dekker. 
 */
#define __split_base(t, movi, type, _x, _y, _a, _fct)           \
    do {                                                        \
        register type z, a0;                                    \
        asm __volatile__                                        \
            (                                                   \
             movi " %[a], %[a0]\n\t"                            \
             "vmul" t " %[f], %[a], %[z]\n\t"                   \
             "vsub" t " %[a], %[z], %[x]\n\t"                   \
             "vsub" t " %[x], %[z], %[x]\n\t"                   \
             "vsub" t " %[x], %[a0], %[y]\n\t"                  \
             : [x] "+x" (_x), [y] "=x" (_y),                    \
               [z] "+x" (z) , [a0] "+x" (a0)                    \
             : [a]  "x" (_a), [f]  "x" (_fct));                 \
    } while (0)

/*
 * Extract high part of scalar p such that x + y = p.
 * Constant r is some power of 2 and |p| < r. 
 */
#define __extract_scalar_base(t, type, _x, _y, _p, _r)          \
    do {                                                        \
        register type q;                                        \
        asm __volatile__                                        \
            (                                                   \
             "vadd" t " %[r], %[p], %[q]\n\t"                   \
             "vsub" t " %[r], %[q], %[x]\n\t"                   \
             "vsub" t " %[x], %[p], %[y]\n\t"                   \
             : [x] "+x" (_x), [y] "=x" (_y), [q] "+x" (q)       \
             : [r] "x" (_r), [p] "x" (_p));                     \
    } while (0)


#if defined(__FMA__)
// versions with fused-multiply-add instruction
/*
 * Standard error free transformation for multiplication: x + y = a*b 
 *
 * The additional registry moves are to protect against calls like
 *   x + y = x*b or x + y = a*x.
 */
#define __twoprod_base(t, movi, type, x, y, a, b, fct)                  \
    do {                                                                \
        register type a0, b0;                                           \
        asm __volatile__                                                \
            (                                                           \
             movi " %[ah], %[a0]\n\t"                                   \
             movi " %[bh], %[b0]\n\t"                                   \
             "vmul" t " %[ah], %[bh], %[ch]\n\t"                        \
             movi " %[ch], %[cl]\n\t"                                   \
             "vfmsub231" t " %[a0], %[b0], %[cl]\n\t"                   \
             : [ch] "+x" (x),  [cl] "+x" (y),                           \
               [a0] "+x" (a0), [b0] "+x" (b0)                           \
             : [ah]  "x"  (a), [bh]  "x"  (b));                         \
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
#define __approx_twodiv_base(t, movi, type, _x, _y, _a, _b, _fct)       \
     do {                                                               \
         register type v, w, a0, b0;                                    \
         asm __volatile__                                               \
             (                                                          \
              movi " %[a], %[a0]\n\t"                                   \
              movi " %[b], %[b0]\n\t"                                   \
              "vdiv" t " %[b], %[a], %[x]\n\t"                          \
              "vmul" t " %[b0], %[x], %[v]\n\t"                         \
              movi " %[v], %[w]\n\t"                                    \
              "vfmsub231" t "  %[b0], %[x], %[w]\n\t"                   \
              "vsub" t " %[v], %[a], %[v]\n\t"                          \
              "vsub" t " %[w], %[v], %[v]\n\t"                          \
              "vdiv" t " %[b0], %[v], %[y]\n\t"                         \
              : [x]  "+x" (_x), [y]  "+x" (_y),                         \
                [v]  "+x" (v),  [w]  "+x" (w),                          \
                [a0] "+x" (a0), [b0] "+x" (b0)                          \
              : [a]  "x"  (_a), [b]  "x"  (_b));                        \
     } while (0)


#else
// here without fused-multiply-add instruction

/*
 * twoprod base:
 *   [a2,a1] = split(ah)                 (lines 1-4)
 *   [b2,b1] = split(bh)                 (lines 5-8)
 *   x = ah*bh                           (line 9)
 *   y = ((x - a1*b1) - a2*b1) - a1*b2   (lines 10-15)
 *   y = a2*b2 - y                       (lines 16-17)
 */
#define __twoprod_base(t, movi, type, _x, _y, _a, _b, _fct)             \
        do {                                                            \
            register type z, a2, a1, b2, b1;                            \
            asm __volatile__                                            \
                (                                                       \
                 "vmul" t " %[f],  %[ah], %[z]\n\t"                     \
                 "vsub" t " %[ah], %[z],  %[a1]\n\t"                    \
                 "vsub" t " %[a1], %[z],  %[a1]\n\t"                    \
                 "vsub" t " %[a1], %[ah], %[a2]\n\t"                    \
                 "vmul" t " %[f],  %[bh], %[z]\n\t"                     \
                 "vsub" t " %[bh], %[z],  %[b1]\n\t"                    \
                 "vsub" t " %[b1], %[z],  %[b1]\n\t"                    \
                 "vsub" t " %[b1], %[bh], %[b2]\n\t"                    \
                 "vmul" t " %[ah], %[bh], %[x0]\n\t"                    \
                 "vmul" t " %[a1], %[b1], %[z]\n\t"                     \
                 "vsub" t " %[z],  %[x0], %[y0]\n\t"                    \
                 "vmul" t " %[a2], %[b1], %[z]\n\t"                     \
                 "vsub" t " %[z],  %[y0], %[y0]\n\t"                    \
                 "vmul" t " %[a1], %[b2], %[z]\n\t"                     \
                 "vsub" t " %[z],  %[y0], %[y0]\n\t"                    \
                 "vmul" t " %[a2], %[b2], %[z]\n\t"                     \
                 "vsub" t " %[y0], %[z],  %[y0]\n\t"                    \
                 : [x0] "+x" (_x), [y0] "+x" (_y), [z]  "+x" (z),       \
                   [a1] "+x" (a1), [a2] "+x" (a2),                      \
                   [b1] "+x" (b1), [b2] "+x" (b2)                       \
                 : [ah]  "x" (_a), [bh]  "x" (_b),  [f] "x" (_fct));    \
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
#define __approx_twodiv_base(t, movi, type, _x, _y, _a, _b, _fct)       \
     do {                                                               \
         register type v, w, a1, a2, b1, b2, z;                         \
         asm __volatile__                                               \
             (                                                          \
              "vdiv" t " %[b],  %[a], %[x]\n\t"                         \
              "vmul" t " %[f],  %[x], %[z]\n\t"                         \
              "vsub" t " %[x],  %[z], %[a1]\n\t"                        \
              "vsub" t " %[a1], %[z], %[a1]\n\t"                        \
              "vsub" t " %[a1], %[x], %[a2]\n\t"                        \
              "vmul" t " %[f],  %[b], %[z]\n\t"                         \
              "vsub" t " %[b],  %[z], %[b1]\n\t"                        \
              "vsub" t " %[b1], %[z], %[b1]\n\t"                        \
              "vsub" t " %[b1], %[b], %[b2]\n\t"                        \
              "vmul" t " %[a],  %[b],  %[v]\n\t"                        \
              "vmul" t " %[a1], %[b1], %[z]\n\t"                        \
              "vsub" t " %[z],  %[v],  %[w]\n\t"                        \
              "vmul" t " %[a2], %[b1], %[z]\n\t"                        \
              "vsub" t " %[z],  %[w],  %[w]\n\t"                        \
              "vmul" t " %[a1], %[b2], %[z]\n\t"                        \
              "vsub" t " %[z],  %[w],  %[w]\n\t"                        \
              "vmul" t " %[a2], %[b2], %[z]\n\t"                        \
              "vsub" t " %[w],  %[z],  %[w]\n\t"                        \
              "vsub" t " %[v], %[a], %[v]\n\t"                          \
              "vsub" t " %[w], %[v], %[v]\n\t"                          \
              "vdiv" t " %[b], %[v], %[y]\n\t"                          \
              : [x]  "+x" (_x), [y]  "+x" (_y), [z]  "+x" (z),          \
                [a1] "+x" (a1), [a2] "+x" (a2),                         \
                [b1] "+x" (b1), [b2] "+x" (b2),                         \
                [v]  "+x" (v),  [w]  "+x" (w)                           \
              : [a]   "x" (_a), [b]   "x" (_b),  [f] "x" (_fct));       \
     } while (0)

#endif    // !defined(__FMA__)



#define __twosum_base_f32(_x, _y, _a, _b) \
    __twosum_base("ss", "movss", float, _x, _y, _a, _b)

#define __fastsum_base_f32(_x, _y, _a, _b) \
    __fastsum_base("ss", "movss", float, _x, _y, _a, _b)

#define __twoprod_base_f32(_x, _y, _a, _b) \
    __twoprod_base("ss", "movss", float, _x, _y, _a, _b, __factor_f32)

#define __approx_twodiv_base_f32(_x, _y, _a, _b) \
    __approx_twodiv_base("ss", "movss", float, _x, _y, _a, _b, __factor_f32)

#define __split_base_f32(_x, _y, _a) \
    __split_base("ss", "movss", float, _x, _y, _a, __factor_f32)

#define __extract_scalar_base_f32(_x, _y, _p, _r)     \
    __extract_scalar_base("ss", float, _x, _y, _p, _r)

// Double precision float

#define __twosum_base_f64(_x, _y, _a, _b) \
    __twosum_base("sd", "movsd", double, _x, _y, _a, _b)

#define __fastsum_base_f64(_x, _y, _a, _b) \
    __fastsum_base("sd", "movsd", double, _x, _y, _a, _b)

#define __twoprod_base_f64(_x, _y, _a, _b) \
    __twoprod_base("sd", "movsd", double, _x, _y, _a, _b, __factor_f64)

#define __approx_twodiv_base_f64(_x, _y, _a, _b) \
    __approx_twodiv_base("sd", "movsd",double, _x, _y, _a, _b, __factor_f64)

#define __split_base_f64(_x, _y, _a) \
    __split_base("sd", "movsd", double, _x, _y, _a, __factor_f64)

#define __extract_scalar_base_f64(_x, _y, _p, _r)     \
    __extract_scalar_base("sd", double, _x, _y, _p, _r)

#endif    // defined(__AVX__)


#if __SIMD_LENGTH >= 128

#ifdef __HAVE_SIMD32X4

#define __twosum_base_f32x4(_x, _y, _a, _b)             \
    __twosum_base("ps", "vmovaps", float32x4_t, _x, _y, _a, _b)

#define __fastsum_base_f32x4(_x, _y, _a, _b)            \
    __fastsum_base("ps", "vmovaps", float32x4_t, _x, _y, _a, _b)

#define __twoprod_base_f32x4(_x, _y, _a, _b) \
    __twoprod_base("ps", "vmovaps", float32x4_t, _x, _y, _a, _b, __fact_f32x4)

#define __approx_twodiv_base_f32x4(_x, _y, _a, _b) \
    __approx_twodiv_base("ps", "vmovaps", float32x4_t, _x, _y, _a, _b, __fact_f32x4);

#define __split_base_f32x4(_x, _y, _a) \
    __split_base("ps", "vmovaps", float32x4_t, _x, _y, _a, __fact_f32x4);

#define __extract_scalar_base_f32x4(_x, _y, _p, _r)            \
    __extract_scalar_base("ps", float32x4_t, _x, _y, _p, _r)


#endif // __HAVE_SIMD32X4

#ifdef __HAVE_SIMD64X2

#define __twosum_base_f64x2(_x, _y, _a, _b)             \
    __twosum_base("pd", "vmovapd", float64x2_t, _x, _y, _a, _b)

#define __fastsum_base_f64x2(_x, _y, _a, _b)            \
    __fastsum_base("pd", "vmovapd", float64x2_t, _x, _y, _a, _b)

#define __twoprod_base_f64x2(_x, _y, _a, _b) \
    __twoprod_base("pd", "vmovapd", float64x2_t, _x, _y, _a, _b, __fact_f64x2)

#define __approx_twodiv_base_f64x2(_x, _y, _a, _b) \
    __approx_twodiv_base("pd", "vmovapd", float64x2_t, _x, _y, _a, _b, __fact_f64x2);

#define __split_base_f64x2(_x, _y, _a) \
    __split_base("pd", "vmovapd", float64x2_t, _x, _y, _a, __fact_f64x2);

#define __extract_scalar_base_f64x2(_x, _y, _p, _r)            \
        __extract_scalar_base("pd", float64x2_t, _x, _y, _p, _r)

#endif // __HAVE_SIMD64X2

#endif  // __SIMD_LENGTH >= 128

#if __SIMD_LENGTH >= 256

#ifdef __HAVE_SIMD32X8

#define __twosum_base_f32x8(_x, _y, _a, _b)             \
    __twosum_base("ps", "vmovaps", float32x8_t, _x, _y, _a, _b)

#define __fastsum_base_f32x8(_x, _y, _a, _b)            \
    __fastsum_base("ps", "vmovaps", float32x8_t, _x, _y, _a, _b)

#define __twoprod_base_f32x8(_x, _y, _a, _b) \
    __twoprod_base("ps", "vmovaps", float32x8_t, _x, _y, _a, _b, __fact_f32x8)

#define __approx_twodiv_base_f32x8(_x, _y, _a, _b) \
    __approx_twodiv_base("ps", "vmovaps", float32x8_t, _x, _y, _a, _b, __fact_f32x8);

#define __split_base_f32x8(_x, _y, _a) \
    __split_base("ps", "vmovaps", float32x8_t, _x, _y, _a, __fact_f32x8);

#define __extract_scalar_base_f32x8(_x, _y, _p, _r)            \
    __extract_scalar_base("ps", float32x8_t, _x, _y, _p, _r)


#endif // __HAVE_SIMD32X8

#ifdef __HAVE_SIMD64X4

#define __twosum_base_f64x4(_x, _y, _a, _b)             \
    __twosum_base("pd", "vmovapd", float64x4_t, _x, _y, _a, _b)

#define __fastsum_base_f64x4(_x, _y, _a, _b)            \
    __fastsum_base("pd", "vmovapd", float64x4_t, _x, _y, _a, _b)

#define __twoprod_base_f64x4(_x, _y, _a, _b) \
    __twoprod_base("pd", "vmovapd", float64x4_t, _x, _y, _a, _b, __fact_f64x4)

#define __approx_twodiv_base_f64x4(_x, _y, _a, _b) \
    __approx_twodiv_base("pd", "vmovapd", float64x4_t, _x, _y, _a, _b, __fact_f64x4);

#define __split_base_f64x4(_x, _y, _a) \
    __split_base("pd", "vmovapd", float64x4_t, _x, _y, _a, __fact_f64x4);

#define __extract_scalar_base_f64x4(_x, _y, _p, _r)            \
    __extract_scalar_base("pd", float64x4_t, _x, _y, _p, _r)

#endif  // __HAVE_SIMD64X4

#endif  // __SIMD_LENGTH >= 256

#endif    // defined(__x86_64__) 

#ifdef __cplusplus
}
#endif


#endif  // __ARMAS_EFTMACROS_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

