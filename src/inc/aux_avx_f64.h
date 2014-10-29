
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_AUX_AVX_F64_H
#define __ARMAS_AUX_AVX_F64_H 1

#include <immintrin.h>

#if ! defined(__gvrotg_enhanced)
#define __gvrotg_enhanced 1

/*
 * This optimized for AVX instruction set. Approx. 5 times faster than the basic
 * version.
 */
static inline
void __gvrotg(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b)
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

    x0  = _mm256_set_pd(1.0, a, b, 1.0);   // x0 = {1, a,   b,   1}
    x1  = _mm256_set_pd(1.0, b, a, 1.0);   // x0 = {1, b,   a,   1}
    t0  = _mm256_div_pd(x0, x1);           // t0 = {1, a/b, b/a, 1}
    x0  = _mm256_mul_pd(t0, t0);           // x3 = {1, (a/b)^2, (b/a)^2, 1}
    t2  = _mm256_hadd_pd(x0, x0);          // x3 = {1+(a/b)^2, ., (b/a)^2+1, ..}
    u0  = _mm256_sqrt_pd(t2);              // u0 = {sqrt(1+(a/b)^2), .., sqrt((b/2)^2+1)}
    one = _mm256_set1_pd(1.0);
    u1  = _mm256_div_pd(one, u0);
    b0  = _mm256_blend_pd(u0, u1, 0x9);    // b0 = {1/u(a),   u(a),   u(b), 1/u(b)} 
    b0  = _mm256_mul_pd(b0, x1);           // b0 = {1/u(a), b*u(a), a*u(b), 1/u(b)} 
    b1  = _mm256_mul_pd(t0, u1);           // b1 = {1/u(a), t*u(a), t*u(b), 1/u(b)} 

    if (__ABS(b) > __ABS(a)) {
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
#endif // defined(__gvrotg_enhanced)

#if defined(__enable_gvright_simd)
#define __gvright_enhanced 1
// this is not enable by default as it seems that AVX/FMA instruction set enabled
// version is not really faster than the simple basic version. This is here
// just for reference. (See perfgvr.c in tests/perf)
static inline
void __gvright(armas_d_dense_t *A, double c, double s, int c1, int c2, int row, int nrow)
{
    register __m256d V0, V1, V2, V3, T0, T1, X0, X1, COS, SIN;
    double *y0, *y1, t0;
    int k;
    
    y0 = &A->elems[c1*A->step+row];
    y1 = &A->elems[c2*A->step+row];
    COS = _mm256_set1_pd(c);
    SIN = _mm256_set1_pd(s);
    
    V0 = _mm256_loadu_pd(&y0[0]);
    V1 = _mm256_loadu_pd(&y1[0]);
    // t0 = c * v0 + s * v1;
    // t1 = c * v1 - s * v0;
    for (k = 0; k < nrow-3; k += 4) {
        V2 = _mm256_loadu_pd(&y0[k+4]);
        V3 = _mm256_loadu_pd(&y1[k+4]);
	T0 = COS*V0;
	T1 = COS*V1;
	X0 = T0 + V1*SIN;
	X1 = T1 - V0*SIN;
        _mm256_storeu_pd(&y0[k], X0);
        _mm256_storeu_pd(&y1[k], X1);
        V0 = V2; V1 = V3;
    }
    if (k == nrow)
        return;

    for (; k < nrow; k++) {
        t0    = c * y0[k] + s * y1[k];
        y1[k] = c * y1[k] - s * y0[k];
        y0[k] = t0;
    }
}

#endif

#if defined(__enable_gvleft_simd)
#define __gvleft_enhanced 1
// this is not enable by default as it seems that AVX/FMA instruction set enabled
// version is not really faster than the simple basic version. This is here
// just for reference. 
static inline
void __gvleft(armas_d_dense_t *A, double c, double s, int r1, int r2, int col, int ncol)
{
    register __m256d V0, V1, V2, V3, T0, T1, X0, X1, COS, SIN;
    double *y0, *y1, t0;
    int k, n0, n1, n2, n3, n4, n5, n6, n7, nshift;
    
    y0 = &A->elems[col*A->step+r1];
    y1 = &A->elems[col*A->step+r2];
    COS = _mm256_set1_pd(c);
    SIN = _mm256_set1_pd(s);
    
    // t0 = c * v0 + s * v1;
    // t1 = c * v1 - s * v0;
    n0 = 0;            n1 = n0 + A->step; n2 = n1 + A->step; n3 = n2 + A->step;
    n4 = n3 + A->step; n5 = n4 + A->step; n6 = n5 + A->step; n7 = n6 + A->step;
    V0 = _mm256_set_pd(y0[n3], y0[n2], y0[n1], y0[n0]);
    V1 = _mm256_set_pd(y1[n3], y1[n2], y1[n1], y1[n0]);

    nshift = A->step << 2;
    for (k = 0; k < ncol-3; k += 4) {
        V2 = _mm256_set_pd(y0[n7], y0[n6], y0[n5], y0[n4]);
        V3 = _mm256_set_pd(y1[n7], y1[n6], y1[n5], y1[n4]);
	T0 = COS*V0;
	T1 = COS*V1;
	X0 = T0 + V1*SIN;
	X1 = T1 - V0*SIN;
        y0[n3] = X0[3]; y0[n2] = X0[2]; y0[n1] = X0[1]; y0[n0] = X0[0];
        y1[n3] = X1[3]; y1[n2] = X1[2]; y1[n1] = X1[1]; y1[n0] = X1[0];

        n3 += nshift; n2 += nshift; n1 += nshift; n0 += nshift;
        n4 += nshift; n5 += nshift; n6 += nshift; n7 += nshift;
        V0 = V2; V1 = V3;
    }
    if (k == ncol)
        return;

    for (; k < ncol; k++, n0 += A->step) {
        t0    = c * y0[n0] + s * y1[n0];
        y1[n0] = c * y1[n0] - s * y0[n0];
        y0[n0] = t0;
    }
}

#endif // __enable_gvleft_simd


#endif  // __ARMAS_AUX_AVX_F64_H

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
