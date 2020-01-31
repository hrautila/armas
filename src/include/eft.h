
// Copyright (c), Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_EFTMACROS_H
#define ARMAS_EFTMACROS_H 1

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Error-free transformation of floating point operations
 *
 * References:
 * (1) T. Ogita, S. Rump, S. Oishi
 *     Accurate Sum and Dot Product,
 *     2005, SIAM Journal of Scientific Computing
 * (2) Ph. Langlois, N. Louvet
 *     Solving Triangular Systems More Accurately and Efficiently
 * (3) T.J. Dekker
 *     A Floating-Point Technique for Extending the Available Precission,
 *     1972, Numeriche Mathematik 18
 * (4) D.E. Knuth
 *     The Art of Computer Programming: Seminumerical Algorithms, Vol 2
 *     1969
 */

#include "simd.h"

#ifndef __SPLIT_FACTOR
#define __SPLIT_FACTOR 1
#define __FACT32 ((1 << 14) + 1)
#define __FACT64 ((1 << 27) + 1)
static const float  __factor_f32 = __FACT32;
static const double __factor_f64 = __FACT64;
#endif

#if defined(FLOAT32)
#define twosum  twosum_f32
#define twoprod twoprod_f32
#define approx_twodiv twodiv_f32
#define fastsum fastsum_f32
#else
#define twosum  twosum_f64
#define twoprod twoprod_f64
#define approx_twodiv twodiv_f64
#define fastsum fastsum_f64
#endif

/*
 * Macros in architecture spesific headers generate multiple "uninitialized" warnings
 * on internal register variables. Ignore these warnings.
 */
#ifndef __nopragma
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#if defined(__x86_64__)
#include "x86_64/eftops.h"

#elif defined(__arm__)
#include "arm/eftops.h"

#endif

#if ! defined(__have_accelerated_eft_versions)

#warning "Using non-accelerated EFT functions!"

// these the basic, non-optimized versions
#define __twosum_base(t, type, x, y, a, b)               \
    do {                                                 \
        volatile register type z, w;                     \
        x = a + b;                                       \
        z = x - a;                                       \
        w = x - z;                                       \
        w = a - w;                                       \
        z = b - z;                                       \
        y = w + z;                                       \
    } while (0)

#define __fastsum_base(t, type, x, y, a, b)              \
    do {                                                 \
        volatile register type z;                        \
        x = a + b;                                       \
        z = a - x;                                       \
        y = z + b;                                       \
    } while (0)

#define __twoprod_base(t, type, x, y, ah, bh, fct)             \
    do {                                                       \
        volatile register type x0, y0, z1, z2;                 \
        register type a1, a2, b1, b2;                          \
        z1 = fct*ah;                                           \
        a1 = z1 - ah;                                          \
        a1 = z1 - a1;                                          \
        a2 = ah - a1;                                          \
        z2 = fct*bh;                                           \
        b1 = z2 - bh;                                          \
        b1 = z2 - b1;                                          \
        b2 = bh - b1;                                          \
        x0 = ah*bh;                                            \
        y0 = (((x0 - a1*b1) - a2*b1) - a1*b2);                 \
        y0 = a2*b2 - y0;                                       \
        (x) = x0; (y) = y0;                                    \
    } while (0)

#define __approx_twodiv_base(t, type, x, y, a, b, fct)  \
    do {                                                \
        volatile register type v, w;                    \
        x = a/b;                                        \
        __twoprod_base(t, type, v, w, x, b, fct);       \
        y = a - v;                                      \
        y = y - w;                                      \
        y = y/b;                                        \
    } while (0)

#define __split_base(t, type, x, y, a, fct)        \
    do {                                           \
        volatile type z;                           \
        z = fct*a;                                 \
        x = z - a;                                 \
        x = z - x;                                 \
        y = a - x;                                 \
    } while (0)


#define __extract_scalar_base(t, type, x, y, p, r)   \
     do {                                            \
         volatile type q;                            \
         q = p - r;                                  \
         x = q - r;                                  \
         y = p - x;                                  \
     } while (0)


// Single precision float

#define __twosum_base_f32(_x, _y, _a, _b) \
    __twosum_base("", float, _x, _y, _a, _b)

#define __fastsum_base_f32(_x, _y, _a, _b) \
    __fastsum_base("", float, _x, _y, _a, _b)

#define __twoprod_base_f32(_x, _y, _a, _b) \
    __twoprod_base("", float, _x, _y, _a, _b, __factor_f32)

#define __approx_twodiv_base_f32(_x, _y, _a, _b) \
    __approx_twodiv_base("", float, _x, _y, _a, _b, __factor_f32)

#define __split_base_f32(_x, _y, _a) \
    __split_base("", float, _x, _y, _a, __factor_f32)

#define __extract_scalar_base_f32(_x, _y, _p, _r)     \
    __extract_scalar_base("", float, _x, _y, _p, _r)

// Double precision float

#define __twosum_base_f64(_x, _y, _a, _b) \
    __twosum_base("", double, _x, _y, _a, _b)

#define __fastsum_base_f64(_x, _y, _a, _b) \
    __fastsum_base("", double, _x, _y, _a, _b)

#define __twoprod_base_f64(_x, _y, _a, _b) \
    __twoprod_base("", double, _x, _y, _a, _b, __factor_f64)

#define __approx_twodiv_base_f64(_x, _y, _a, _b) \
    __approx_twodiv_base("", double, _x, _y, _a, _b, __factor_f64)

#define __split_base_f64(_x, _y, _a) \
    __split_base("", double, _x, _y, _a, __factor_f64)

#define __extract_scalar_base_f64(_x, _y, _p, _r)     \
    __extract_scalar_base("", double, _x, _y, _p, _r)

#endif  // non-optimized

/* ------------------------------------------------------------------------------
 *
 */

static inline void __attribute__((__always_inline__))
twosum_f32(float *x, float *y, float a, float b)
{
    __twosum_base_f32((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f32_ext(float *x, float *y, float ah, float al, float bh, float bl)
{
    __twosum_base_f32((*x), (*y), ah, bh);
    *y += al + bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f32(float *x, float *y, float a, float b)
{
    __fastsum_base_f32((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f32_ext(float *x, float *y, float ah, float al, float bh, float bl)
{
    __fastsum_base_f32((*x), (*y), ah, bh);
    *y += al + bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f32(float *x, float *y, float a, float b)
{
    __twoprod_base_f32((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f32_ext(float *x, float *y, float ah, float al, float bh, float bl)
{
    register float yl = ah*bl + bh*al;
    __twoprod_base_f32((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f32(float *x, float *y, float a, float b)
{
    __approx_twodiv_base_f32((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f32_ext(float *x, float *y, float ah, float al, float b)
{
    register float yl = al/b;
    __approx_twodiv_base_f32((*x), (*y), ah, b);
    (*y) += yl;
}

static inline void __attribute__((__always_inline__))
split_f32(float *x, float *y, float a)
{
    __split_base_f32((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f32(float *x, float *y, float p, float r)
{
    __extract_scalar_base_f32((*x), (*y), p, r);
}

static inline void __attribute__((__always_inline__))
twosum_f64(double *x, double *y, double a, double b)
{
    __twosum_base_f64((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f64_ext(double *x, double *y, double ah, double al, double bh, double bl)
{
    __twosum_base_f64((*x), (*y), ah, bh);
    *y += al + bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f64(double *x, double *y, double a, double b)
{
    __fastsum_base_f64((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f64_ext(double *x, double *y, double ah, double al, double bh, double bl)
{
    __fastsum_base_f64((*x), (*y), ah, bh);
    *y += al + bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f64(double *x, double *y, double a, double b)
{
    __twoprod_base_f64((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f64_ext(double *x, double *y, double ah, double al, double bh, double bl)
{
    register double yl = ah*bl + bh*al;
    __twoprod_base_f64((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f64(double *x, double *y, double a, double b)
{
    __approx_twodiv_base_f64((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f64_ext(double *x, double *y, double ah, double al, double b)
{
    register double yl = al/b;
    __approx_twodiv_base_f64((*x), (*y), ah, b);
    (*y) += yl;
}

static inline void __attribute__((__always_inline__))
split_f64(double *x, double *y, double a)
{
    __split_base_f64((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f64(float *x, float *y, float p, float r)
{
    __extract_scalar_base_f64((*x), (*y), p, r);
}
/* --------------------------------------------------------------------------------------
 * Vectorized versions, 128bit
 */

#if __SIMD_LENGTH >= 128

#ifdef __HAVE_SIMD32X4

static inline void __attribute__((__always_inline__))
twosum_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t a, float32x4_t b)
{
    __twosum_base_f32x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f32x4_ext(float32x4_t *x, float32x4_t *y,
                   float32x4_t ah, float32x4_t al, float32x4_t bh, float32x4_t bl)
{
    __twosum_base_f32x4((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t a, float32x4_t b)
{
    __fastsum_base_f32x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f32x4_ext(float32x4_t *x, float32x4_t *y,
                   float32x4_t ah, float32x4_t al, float32x4_t bh, float32x4_t bl)
{
    __fastsum_base_f32x4((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t a, float32x4_t b)
{
    __twoprod_base_f32x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f32x4_ext(float32x4_t *x, float32x4_t *y,
                   float32x4_t ah, float32x4_t al, float32x4_t bh, float32x4_t bl)
{
    register float32x4_t yl = ah*bl + bh*al;
    __twoprod_base_f32x4((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t a, float32x4_t b)
{
    __approx_twodiv_base_f32x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f32x4_ext(float32x4_t *x, float32x4_t *y,
                   float32x4_t ah, float32x4_t al, float32x4_t b)
{
    register float32x4_t yl = al/b;
    __approx_twodiv_base_f32x4((*x), (*y), ah, b);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
split_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t a)
{
    __split_base_f32x4((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f32x4(float32x4_t *x, float32x4_t *y, float32x4_t p, float32x4_t r)
{
    __extract_scalar_base_f32x4((*x), (*y), p, r);
}

#endif // __HAVE_SIMD32X4

#ifdef __HAVE_SIMD64X2

static inline void __attribute__((__always_inline__))
twosum_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t a, float64x2_t b)
{
    __twosum_base_f64x2((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f64x2_ext(float64x2_t *x, float64x2_t *y,
                 float64x2_t ah, float64x2_t al, float64x2_t bh, float64x2_t bl)
{
    __twosum_base_f64x2((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t a, float64x2_t b)
{
    __fastsum_base_f64x2((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f64x2_ext(float64x2_t *x, float64x2_t *y,
                  float64x2_t ah, float64x2_t al, float64x2_t bh, float64x2_t bl)
{
    __fastsum_base_f64x2((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t a, float64x2_t b)
{
    __twoprod_base_f64x2((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f64x2_ext(float64x2_t *x, float64x2_t *y,
                  float64x2_t ah, float64x2_t al, float64x2_t bh, float64x2_t bl)
{
    register float64x2_t yl = ah*bl + bh*al;
    __twoprod_base_f64x2((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t a, float64x2_t b)
{
    __approx_twodiv_base_f64x2((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f64x2_ext(float64x2_t *x, float64x2_t *y,
                 float64x2_t ah, float64x2_t al, float64x2_t b)
{
    register float64x2_t yl = al/b;
    __approx_twodiv_base_f64x2((*x), (*y), ah, b);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
split_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t a)
{
    __split_base_f64x2((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f64x2(float64x2_t *x, float64x2_t *y, float64x2_t p, float64x2_t r)
{
    __extract_scalar_base_f64x2((*x), (*y), p, r);
}

#endif // __HAVE_SIMD64X2

#endif // __SIMD_LENGTH >= 128

/* --------------------------------------------------------------------------------------
 * Vectorized versions, 256bit
 */

#if __SIMD_LENGTH >= 256
// for f32x8 and f64x4
#ifdef __HAVE_SIMD32X8

static inline void __attribute__((__always_inline__))
twosum_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t a, float32x8_t b)
{
    __twosum_base_f32x8((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f32x8_ext(float32x8_t *x, float32x8_t *y,
                 float32x8_t ah, float32x8_t al, float32x8_t bh, float32x8_t bl)
{
    __twosum_base_f32x8((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t a, float32x8_t b)
{
    __fastsum_base_f32x8((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f32x8_ext(float32x8_t *x, float32x8_t *y,
                  float32x8_t ah, float32x8_t al, float32x8_t bh, float32x8_t bl)
{
    __fastsum_base_f32x8((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t a, float32x8_t b)
{
    __twoprod_base_f32x8((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f32x8_ext(float32x8_t *x, float32x8_t *y,
                  float32x8_t ah, float32x8_t al, float32x8_t bh, float32x8_t bl)
{
    register float32x8_t yl = ah*bl + bh*al;
    __twoprod_base_f32x8((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t a, float32x8_t b)
{
    __approx_twodiv_base_f32x8((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f32x8_ext(float32x8_t *x, float32x8_t *y,
                 float32x8_t ah, float32x8_t al, float32x8_t b)
{
    register float32x8_t yl = al/b;
    __approx_twodiv_base_f32x8((*x), (*y), ah, b);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
split_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t a)
{
    __split_base_f32x8((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f32x8(float32x8_t *x, float32x8_t *y, float32x8_t p, float32x8_t r)
{
    __extract_scalar_base_f32x8((*x), (*y), p, r);
}

#endif  //__HAVE_SIMD32X8


#ifdef __HAVE_SIMD64X4

static inline void __attribute__((__always_inline__))
twosum_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t a, float64x4_t b)
{
    __twosum_base_f64x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twosum_f64x4_ext(float64x4_t *x, float64x4_t *y,
                 float64x4_t ah, float64x4_t al, float64x4_t bh, float64x4_t bl)
{
    __twosum_base_f64x4((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
fastsum_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t a, float64x4_t b)
{
    __fastsum_base_f64x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
fastsum_f64x4_ext(float64x4_t *x, float64x4_t *y,
                  float64x4_t ah, float64x4_t al, float64x4_t bh, float64x4_t bl)
{
    __fastsum_base_f64x4((*x), (*y), ah, bh);
    *y += al;
    *y += bl;
}

static inline void __attribute__((__always_inline__))
twoprod_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t a, float64x4_t b)
{
    __twoprod_base_f64x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twoprod_f64x4_ext(float64x4_t *x, float64x4_t *y,
                  float64x4_t ah, float64x4_t al, float64x4_t bh, float64x4_t bl)
{
    register float64x4_t yl = ah*bl + bh*al;
    __twoprod_base_f64x4((*x), (*y), ah, bh);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
twodiv_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t a, float64x4_t b)
{
    __approx_twodiv_base_f64x4((*x), (*y), a, b);
}

static inline void __attribute__((__always_inline__))
twodiv_f64x4_ext(float64x4_t *x, float64x4_t *y,
                 float64x4_t ah, float64x4_t al, float64x4_t b)
{
    register float64x4_t yl = al/b;
    __approx_twodiv_base_f64x4((*x), (*y), ah, b);
    *y += yl;
}

static inline void __attribute__((__always_inline__))
split_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t a)
{
    __split_base_f64x4((*x), (*y), a);
}

static inline void __attribute__((__always_inline__))
extract_f64x4(float64x4_t *x, float64x4_t *y, float64x4_t p, float64x4_t r)
{
    __extract_scalar_base_f64x4((*x), (*y), p, r);
}

#ifndef __nopragma
#pragma GCC diagnostic pop
#endif

#endif // __HAVE_SIMD64X4

#endif // __SIMD_LENGTH >= 256


#ifdef __cplusplus
}
#endif

#endif  // __ARMAS_EFTMACROS_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

