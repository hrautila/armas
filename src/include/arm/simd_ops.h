
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SIMD_H
#define __ARMAS_SIMD_H 1

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Macros in architecture spesific headers generate multiple "break strict-aliasing rules" warnings.
 * Ignore these warnings.
 */
#ifndef __nopragma
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#if defined(__arm__)
#if defined(__ARM_NEON)

#include <arm_neon.h>

#ifndef __HAVE_SIMD32X4
#define __HAVE_SIMD32X4 1
#endif

#ifdef __HAVE_SIMD32X4

static inline __attribute__((__always_inline__))
float32x4_t set1_f32x4(float v)
{
    return (float32x4_t){v, v, v, v};
}

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
  return 0.0 == hsum_f32x4(C);
}

#if 0
static inline __attribute__((__always_inline__))
int le_f32x4(float32x4_t A, float32x4_t B)
{
  register float32x4_t C;
  C = (float32x4_t)(A > B); 
  return 0.0 == hsum_f32x4(C);
}

static inline __attribute__((__always_inline__))
int lt_f32x4(float32x4_t A, float32x4_t B)
{
  register float32x4_t C;
  C = (float32x4_t)(A >= B); 
  return 0.0 == hsum_f32x4(C);
}
#endif

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

static inline __attribute__((__always_inline__))
float32x4_t abs_f32x4(float32x4_t A)
{
    return vabsq_f32(A);
}

static inline __attribute__((__always_inline__))
float32x4_t neg_f32x4(float32x4_t A)
{
    return vnegq_f32(A);
}

static inline __attribute__((__always_inline__))
float32x4_t zero_f32x4()
{
    uint32x4_t z;
    return (float32x4_t)veorq_u32(z, z);
}

static inline __attribute__((__always_inline__))
float32x4_t load_f32x4(const float *a)
{
    return vld1q_f32(a);
}

static inline __attribute__((__always_inline__))
float32x4_t loadu_f32x4(const float *a)
{
    return vld1q_f32(a);
}

#endif // __HAVE_SIMD32X4

#endif // __ARM_NEON
#endif // __arm__

#ifndef __nopragma
#pragma GCC diagnostic pop
#endif

#ifdef __cplusplus
}
#endif

#endif // __SIMD_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
