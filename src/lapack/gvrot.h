
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_LAPACK_GVROT_H
#define ARMAS_LAPACK_GVROT_H 1

#if defined(FLOAT64) && defined(__AVX__)
#include <immintrin.h>
/*
 * This optimized for AVX instruction set. Approx. 5 times faster than the basic
 * version.
 */
static inline
void gvrotg(DTYPE * c, DTYPE * s, DTYPE * r, DTYPE a, DTYPE b)
{
    register __m256d x0, x1, t0, t2, u0, u1, one, b0, b1;
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
        return;
    }
    if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
        return;
    }
    // Compute for both cases: |a| > |b| and |b| > |a|
    // - set_pd() order: [3, 2, 1, 0]
    // - access:
    //     |a| > |b|  x[0], x[1]
    //     |b| > |a|  x[2], x[3]

    x0 = _mm256_set_pd(1.0, a, b, 1.0); // x0 = {1, a,   b,   1}
    x1 = _mm256_set_pd(1.0, b, a, 1.0); // x0 = {1, b,   a,   1}
    t0 = _mm256_div_pd(x0, x1);         // t0 = {1, a/b, b/a, 1}
    x0 = _mm256_mul_pd(t0, t0);         // x3 = {1, (a/b)^2, (b/a)^2, 1}
    t2 = _mm256_hadd_pd(x0, x0);        // x3 = {1+(a/b)^2, ., (b/a)^2+1, ..}
    u0 = _mm256_sqrt_pd(t2);            // u0 = {sqrt(1+(a/b)^2), .., sqrt((b/2)^2+1)}
    one = _mm256_set1_pd(1.0);
    u1 = _mm256_div_pd(one, u0);
    b0 = _mm256_blend_pd(u0, u1, 0x9);  // b0 = {1/u(a),   u(a),   u(b), 1/u(b)}
    b0 = _mm256_mul_pd(b0, x1);         // b0 = {1/u(a), b*u(a), a*u(b), 1/u(b)}
    b1 = _mm256_mul_pd(t0, u1);         // b1 = {1/u(a), t*u(a), t*u(b), 1/u(b)}

    if (ABS(b) > ABS(a)) {
        *s = b0[3];
        *r = b0[2];
        *c = b1[2];
        if (signbit(b)) {
            *s = -(*s);
            *c = -(*c);
            *r = -(*r);
        }
    } else {
        *c = b0[0];
        *r = b0[1];
        *s = b1[1];
    }
}
#else
/*
 * \brief Compute Givens rotation
 */
static inline
void gvrotg(DTYPE * c, DTYPE * s, DTYPE * r, DTYPE a, DTYPE b)
{
    DTYPE t, u;

    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
    } else if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
    } else if (ABS(b) > ABS(a)) {
        t = a / b;
        u = SQRT(1.0 + t * t);
        if (SIGN(b))
            u = -u;
        *s = 1.0 / u;
        *c = (*s) * t;
        *r = b * u;
    } else {
        t = b / a;
        u = SQRT(1.0 + t * t);
        *r = a * u;
        *c = 1.0 / u;
        *s = (*c) * t;
    }
}
#endif // FLOAT64 && __AVX__

#endif /* ARMAS_LAPACK_GVROT_H */
