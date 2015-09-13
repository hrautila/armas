
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SIMD_H
#define __ARMAS_SIMD_H 1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__)
#if defined(__AVX__)

#include <immintrin.h>

#ifndef __HAVE_SIMD32X4 
#define __HAVE_SIMD32X4  1
#endif
#ifndef __HAVE_SIMD32X8 
#define __HAVE_SIMD32X8  1
#endif
#ifndef __HAVE_SIMD64X2
#define __HAVE_SIMD64X2  1
#endif
#ifndef __HAVE_SIMD64X4 
#define __HAVE_SIMD64X4  1
#endif


typedef __m128  float32x4_t;
typedef __m256  float32x8_t;
typedef __m128d float64x2_t;
typedef __m256d float64x4_t;

#if ! defined(__SIMD_LENGTH)
#define __SIMD_LENGTH 256
#endif

/* ------------------------------------------------------------------------
 * Single precision, 128bit, 4 elements
 */
#ifdef __HAVE_SIMD32X4

static inline float32x4_t __attribute__((__always_inline__))
set1_f32x4(float a)
{
    return (float32x4_t)_mm_set1_ps(a);
}

static inline float32x4_t __attribute__((__always_inline__))
set_f32x4(float a, float b, float c, float d)
{
    return (float32x4_t)_mm_set_ps(a, b, c, d);
}

static inline __attribute__((__always_inline__))
float hsum_f32x4(float32x4_t A)
{
    register float32x4_t B;
    B = _mm_hadd_ps(A, A);
    B = _mm_hadd_ps(B, B);
    return B[0];
}

static inline __attribute__((__always_inline__))
int eq_f32x4(float32x4_t A, float32x4_t B)
{
    float32x4_t C = A != B;
    return 0.0 == hsum_f32x4(C);
}

/*
 * {ABCD} : {CDAB} = {ABAB}
 * {ABAB} : {BABA} = {AAAA}
 */
static inline __attribute__((__always_inline__))
float max_f32x4(float32x4_t A)
{
    register float32x4_t B;
    B = _mm_max_ps(A, _mm_permute_ps(A, 0b00011011));
    B = _mm_max_ps(B, _mm_permute_ps(B, 0b01001110));
    return B[0];
}

static inline __attribute__((__always_inline__))
float min_f32x4(float32x4_t A)
{
    register float32x4_t B;
    B = _mm_min_ps(A, _mm_permute_ps(A, 0b00011011));
    B = _mm_min_ps(B, _mm_permute_ps(B, 0b01001110));
    return B[0];
}

// half flip
static inline __attribute__((__always_inline__))
float32x4_t hflip_f32x4(float32x4_t A)
{
  return _mm_permute_ps(A, 0b01001110);
}

// pairwise flip
static inline __attribute__((__always_inline__))
float32x4_t pflip_f32x4(float32x4_t A)
{
  return _mm_permute_ps(A, 0b10110001);
}

#endif // __HAVE_SIMD32X4

/* ------------------------------------------------------------------------
 * Double precision, 128bit, 2 elements
 */
#ifdef __HAVE_SIMD64X2

static inline float64x2_t __attribute__((__always_inline__))
set1_f64x2(double a)
{
    return (float64x2_t)_mm_set1_pd(a);
}

static inline float64x2_t __attribute__((__always_inline__))
set_f64x2(double a, double b)
{
    return (float64x2_t)_mm_set_pd(a, b);
}

static inline __attribute__((__always_inline__))
double hsum_f64x2(float64x2_t A)
{
    register float64x2_t B;
    B = _mm_hadd_pd(A, A);
    return B[0];
}

static inline __attribute__((__always_inline__))
double max_f64x2(float64x2_t A)
{
    register float64x2_t B;
    B = _mm_max_pd(B, _mm_permute_pd(B, 0b01));
    return B[0];
}

static inline __attribute__((__always_inline__))
double min_f64x2(float64x2_t A)
{
    register float64x2_t B;
    B = _mm_min_pd(B, _mm_permute_pd(B, 0b01));
    return B[0];
}

#endif // __HAVE_SIMD64X2

/* ------------------------------------------------------------------------
 * Double precision, 256bit, 4 elements
 */
#ifdef __HAVE_SIMD64X4

static inline float64x4_t __attribute__((__always_inline__))
set1_f64x4(double a)
{
    return (float64x4_t)_mm256_set1_pd(a);
}

static inline float64x4_t __attribute__((__always_inline__))
set_f64x4(double a, double b, double c, double d)
{
    return (float64x4_t)_mm256_set_pd(a, b, c, d);
}

/*
 * Horizontal sum if 4 double precision floats (in 4 instructions)
 */
static inline __attribute__((__always_inline__))
double hsum_f64x4(float64x4_t A)
{
    register float64x4_t B;
    B  = _mm256_hadd_pd(A, A);
    B += _mm256_permute2f128_pd(B, B, 1);
    return B[0];
}

static inline __attribute__((__always_inline__))
int eq_f64x4(float64x4_t A, float64x4_t B)
{
    register float64x4_t C = A != B;
    return 0.0 == hsum_f64x4(C);
}

/*
 * Assume A > B > C > D
 *
 * 1. swap low/high 128bit and compare
 *       {ABCD} : {CDAB} = {ABAB}
 * 2. swap 64 bit values pairwise and compare
 *       {ABAB} : {BABA} = {AAAA}
 */

static inline __attribute__((__always_inline__))
double max_f64x4(float64x4_t A)
{
    register float64x4_t B;
    B = _mm256_max_pd(A, _mm256_permute2f128_pd(A, A, 1));
    B = _mm256_max_pd(B, _mm256_permute_pd(B, 0b0101));
    return B[0];
}

static inline __attribute__((__always_inline__))
double min_f64x4(float64x4_t A)
{
    register float64x4_t B;
    B = _mm256_min_pd(A, _mm256_permute2f128_pd(A, A, 1));
    B = _mm256_min_pd(B, _mm256_permute_pd(B, 0b0101));
    return B[0];
}

// half flip; {1, 2, 3, 4} -> {3, 4, 1, 2}
static inline __attribute__((__always_inline__))
float64x4_t hflip_f64x4(float64x4_t A)
{
  return _mm256_permute2f128_pd(A, A, 1);
}

// pairwise flip; {1, 2, 3, 4} -> {2, 1, 4, 3}
static inline __attribute__((__always_inline__))
float64x4_t pflip_f64x4(float64x4_t A)
{
  return _mm256_permute_pd(A, 0b0101);
}

#endif // __HAVE_SIMD64X4

/* ------------------------------------------------------------------------
 * Float vector, 256bit, 8 elements
 */
#ifdef __HAVE_SIMD32X8

static inline float32x8_t __attribute__((__always_inline__))
set1_f32x8(float a)
{
    return (float32x8_t)_mm256_set1_ps(a);
}

static inline float32x8_t __attribute__((__always_inline__))
set_f32x8(float a, float b, float c, float d, float e, float f, float g, float h)
{
    return (float32x8_t)_mm256_set_ps(a, b, c, d, e, f, g, h);
}

/**
 * Horizontal sum of 256bit packed single precision floats (8)
 */
static inline __attribute__((__always_inline__))
float hsum_f32x8(float32x8_t A)
{
    register float32x8_t B;
    B  = _mm256_hadd_ps(A, A);
    B += _mm256_permute2f128_ps(B, B, 1);
    B  = _mm256_hadd_ps(B, B);
    return B[0];
}

/**
 * Test if single precision elements of 256bit vector are equal.
 */
static inline __attribute__((__always_inline__))
int eq_f32x8(float32x8_t A, float32x8_t B)
{
    register float32x8_t C = A != B;
    return 0.0 == hsum_f32x8(C);
}

/*
 * Assume A > B > ... > G > H
 *
 * 1. swap low/high 128bit and compare
 *       {ABCDEFGH} : {EFGHABCD} = {ABCDACBD}
 * 2. swap 32 values pairwise and compare
 *       {ABCDABCD} : {BADCBACD} = {AACCAACC}
 * 3. swap 64 bits pairwise and compare
 *       {AACCAACC} : {CCAACCAA} = {AAAAAAAA}
 */

/**
 * Max of 256bit packed single precision vector
 */
static inline __attribute__((__always_inline__))
float max_f32x8(float32x8_t A)
{
    register float32x8_t B;
    B = _mm256_max_ps(A, _mm256_permute2f128_ps(A, A, 1));
    B = _mm256_max_ps(B, _mm256_permute_ps(B, 0b10110001));
    B = _mm256_max_ps(B, _mm256_permute_ps(B, 0b00011011));
    return B[0];
}

/**
 * Min of 256bit packed single precision vector
 */
static inline __attribute__((__always_inline__))
float min_f32x8(float32x8_t A)
{
    register float32x8_t B;
    B = _mm256_min_ps(A, _mm256_permute2f128_ps(A, A, 1));
    B = _mm256_min_ps(B, _mm256_permute_ps(B, 0b10110001));
    B = _mm256_min_ps(B, _mm256_permute_ps(B, 0b00011011));
    return B[0];
}

/**
 * Flip 256bit vector halves of single precision elements, 32x4 parts
 */
static inline __attribute__((__always_inline__))
float32x8_t hflip_f32x8(float32x8_t A)
{
  return _mm256_permute2f128_ps(A, A, 1);
}

/**
 * Flip 256bit vector single precision elements pairwise, 
 */
static inline __attribute__((__always_inline__))
float32x8_t pflip_f32x8(float32x8_t A)
{
  return _mm256_permute_ps(A, 0b10110001);
}

/**
 * Flip 256bit vector  single precision elements in pairs, 32x2 parts
 */
static inline __attribute__((__always_inline__))
float32x8_t qflip_f32x8(float32x8_t A)
{
  return _mm256_permute_ps(A, 0b01001110);
}

#endif  // __HAVE_SIMD32X8

#endif  // __AVX__
#endif  // __x86_64__


#if defined(__arm__)
#if defined(__ARM_NEON)

#include <arm_neon.h>

#ifndef __HAVE_SIMD32X4
#define __HAVE_SIMD32X4 1
#endif

#ifdef __HAVE_SIMD32X4
static inline __attribute__((__always_inline__))
float hsum_f32x4(float32x4_t A)
{
  register float32x2_t al, ah;
  ah = vget_low_f32(A) + vget_high_f32(A);
  al = vpadd_f32(ah, ah);
  return al[0];
}

static inline __attribute__((__always_inline__))
int eq_f32x4(float32x4_t A, float32x4_t B)
{
  register float32x4_t C;
  C = (float32x4_t)(A != B); 
  return 0.0 == hsum_f32x4(C));
}

static inline __attribute__((__always_inline__))
float max_f32x4(float32x4_t A)
{
  register float32x4_t B;
  register float32x2_t C;
  B = vmaxq_f32(A, vrev64q_f32(A));
  C = vmax_f32(vget_low_f32(B), vget_high_f32(B));
  return C[0];
}

static inline __attribute__((__always_inline__))
float min_f32x4(float32x4_t A)
{
  register float32x4_t B;
  register float32x2_t C;
  B = vminq_f32(A, vrev64q_f32(A));
  C = vmin_f32(vget_low_f32(B), vget_high_f32(B));
  return C[0];
}


#endif // __HAVE_SIMD32X4

#endif // __ARM_NEON
#endif // __arm__

#ifdef __cplusplus
}
#endif

#endif // __SIMD_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
