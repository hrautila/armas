
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_EFT_X86_64_H
#define __ARMAS_EFT_X86_64_H 1

// enhanced versions of basic error-free operations.
#include <emmintrin.h>
#include <immintrin.h>

#if ! defined(__COMPLEX128) && ! defined(__COMPLEX64)

// default is: double and single precision float here

#if defined(__FLOAT32)
#define _mm_set   _mm_set_ss
#define _mm_sub   _mm_sub_ss
#define _mm_add   _mm_add_ss
#define _mm_mul   _mm_mul_ss
#define _mm_div   _mm_div_ss
#define _mm_store _mm_store_ss
#define _mm_fmsub _mm_fmsub_ss
#define _mm_fmadd _mm_fmadd_ss

#else
// double precision instructions
#define _mm_set   _mm_set_sd
#define _mm_sub   _mm_sub_sd
#define _mm_add   _mm_add_sd
#define _mm_mul   _mm_mul_sd
#define _mm_div   _mm_div_sd
#define _mm_store _mm_store_sd
#define _mm_fmsub _mm_fmsub_sd
#define _mm_fmadd _mm_fmadd_sd
#endif

/*
 * Error free summation; computes x + y = a + b where x = fl(a+b) and |y| < |x|
 */
#if __SSE2__
// cost: 6 FLOPS
static inline
void twosum(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    register __m128d z, a0, b0, x0;
    a0 = _mm_set(a);
    b0 = _mm_set(b);
    x0 = _mm_add(a0, b0);
    _mm_store(x, x0);
    z  = _mm_sub(x0, a0);
    x0 = _mm_sub(x0, z);
    z  = _mm_sub(b0, z);
    x0 = _mm_sub(a0, x0);
    x0 = _mm_add(x0, z);
    _mm_store(y, x0);
}
#define twosum_enhanced 1
#endif // __SSE2__.twosum

/*
 * Error free summation; computes x + y = a + b where x = fl(a+b) and |y| < |x| iff |a| > |b|
 */
#if __SSE2__
// cost: 3 FLOPS
static inline
void fast_twosum(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    register __m128d z, a0, b0, x0;
    a0 = _mm_set(a);
    b0 = _mm_set(b);
    x0 = _mm_add(a0, b0);
    _mm_store(x, x0);
    z  = _mm_sub(x0, a0);
    x0 = _mm_sub(x0, z);
    _mm_store(y, x0);
}
#define fast_twosum_enhanced 1
#endif  // __SSE2__.fast_twosum

/*
 * Error free split; x + y = a
 */
#if __SSE2__
// cost: 4 FLOPS
static inline
void split(DTYPE *x, DTYPE *y, DTYPE a)
{
    register __m128d c0, a0, c1, f0;
  
    f0 = _mm_set_sd(__FACTOR);
    a0 = _mm_set_sd(a);
    c0 = _mm_mul_sd(f0, a0);
    c1 = _mm_sub_sd(c0, a0);
    c0 = _mm_sub_sd(c0, c1);
    _mm_store_sd(x, c0);
    c1 = _mm_sub_sd(a0, c0);
    _mm_store_sd(y, c1);
}

#define SPLT(x, y, a, f, type)			\
  do {						\
    register type c0;				\
    c0 = _mm_mul_sd(f, a);			\
    x  = _mm_sub_sd(c0, a);			\
    x  = _mm_sub_sd(c0, x);			\
    y  = _mm_sub_sd(a,  x);			\
  } while (0);

#define split_enhanced 1
#endif  // __SSE2__.split

/*
 * Error transformation extracting high order part
 */
#if __SSE2__
// cost: 3 FLOPS
static inline
void extract_scalar(DTYPE *x, DTYPE *y, DTYPE r, DTYPE p)
{
    register __m128d x0, r0, p0;

    r0 = _mm_set_sd(r);
    p0 = _mm_set_sd(p);
    x0 = _mm_add_sd(r0, p0);
    x0 = _mm_sub_sd(x0, r0);
    _mm_store_sd(x, x0);
    x0 = _mm_sub_sd(p0, x0);
    _mm_store_sd(y, x0);
}
#define extract_scalar_enhanced 1
#endif // __SSE2__.extract_scalar

/*
 * Error free product: x + y = a*b ; where x = fl(a*b) and |y| < |x|
 */
#if __FMA__
// cost: 2 FLOPS
static inline
void twoprod(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
  register __m128d x0, a0, b0;
  a0 = _mm_set_sd(a);
  b0 = _mm_set_sd(b);
  x0 = _mm_mul_sd(a0, b0);
  _mm_store_sd(x, x0);
  x0 = _mm_fmsub_sd(a0, b0, x0);
  _mm_store_sd(y, x0);
}

#define twoprod_enhanced 1

#elif __SSE2__
// cost: 17 FLOPS
static inline
void twoprod(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    register __m128d x0, y0, a0, a1, a2, b0, b1, b2, f0, z;
 
    f0 = _mm_set_sd(__FACTOR);
    a0 = _mm_set_sd(a);
    b0 = _mm_set_sd(b);

    z  = _mm_mul_sd(f0, a0);
    a1 = _mm_sub_sd(z, a0);
    a1 = _mm_sub_sd(z, a1);
    a2 = _mm_sub_sd(a0, a1);

    z  = _mm_mul_sd(f0, b0);
    b1 = _mm_sub_sd(z, b0);
    b1 = _mm_sub_sd(z, b1);
    b2 = _mm_sub_sd(b0, b1);

    x0 = _mm_mul_sd(a0, b0);
    y0 = _mm_mul_sd(a1, b1);
    _mm_store_sd(x, x0);
    x0 = _mm_sub_sd(x0, y0);
    y0 = _mm_mul_sd(a2, b1);
    x0 = _mm_sub_sd(x0, y0);
    y0 = _mm_mul_sd(a1, b2);
    x0 = _mm_sub_sd(x0, y0);
    y0 = _mm_mul_sd(a2, b2);
    x0 = _mm_sub_sd(y0, x0);
    _mm_store_sd(y, x0);
}

#define twoprod_enhanced 1
#endif // __FMA__.twoprod


#else
// Complex versions here

#endif

#endif  // __ARMAS_EFT_X86_64_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
