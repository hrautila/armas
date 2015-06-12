
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_EFT_H
#define __ARMAS_EFT_H 1

/*
 * Error-free transformation of floating point operations
 *
 * References:
 * (1) T. Ogita, S. Rump, S. Oishi
 *     Accurate Sum and Dot Product,
 *     2005, SIAM Journal of Scientific Computing
 * (3) Ph. Langlois, N. Louvet
 *     Solving Triangular Systems More Accurately and Efficiently
 * (4) T.J. Dekker
 *     A Floating-Point Technique for Extending the Available Precission,
 *     1972, Numeriche Mathematik 18
 * (5) D.E. Knuth
 *     The Art of Computer Programming: Seminumerical Algorithms, Vol 2
 *     1969
 */


#if defined(__FLOAT32) || defined(__COMPLEX64)
#define __NBITS 14
#define __MBITS 27
#else
// default is double precision float
#define __NBITS 27
#define __MBITS 53
#endif

#if ! defined(__COMPLEX128) && ! defined(__COMPLEX64)
// default is: double and single precision float here
#define __FACTOR ((DTYPE)((1 << __NBITS)+1))
#endif

#if defined(__x86_64__)
#include "eft_x86_64.h"
#endif

#if ! defined(__COMPLEX128) && ! defined(__COMPLEX64)

// default is: double and single precision float here

/*
 * Error free summation; computes x + y = a + b where x = fl(a+b) and |y| < |x|
 * (from (1), originally from (5))
 */
#if ! defined(twosum_enhanced)
static inline
void twosum(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    // Define these volatile otherwise compiler may optimize them away;
    // (This produces correct result with gcc -O3 level.)
    volatile DTYPE z, y0;
    *x = a + b;
    z  = *x - a;
    y0 = *x - z;
    y0 = a - y0;
    *y = y0 + (b - z);
    return;
}
#endif

/*
 * Error free summation; computes x + y = a + b where x = fl(a+b) and |y| < |x| iff |a| > |b|
 * (from (1), originally from (4))
 */
#if ! defined(fast_twosum_enhanced)
static inline
void fast_twosum(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    volatile DTYPE q;
    *x = a + b;
    q  = *x - a;
    *y = b - q;
}
#endif

/*
 * Error free split; x + y = a
 * From (2)
 */
#if ! defined(split_enhanced)
static inline
DTYPE split(DTYPE *x, DTYPE *y, DTYPE a)
{
    volatile register DTYPE c;
    c = __FACTOR*a;
    *x = (c - (c - a));
    *y = a - *x;
    return c; // don't allow optimizing away
}
#endif

/*
 * Error transformation extracting high order part
 * From (2)
 */
#if ! defined(extract_scalar_enhanced)
static inline
DTYPE extract_scalar(DTYPE *x, DTYPE *y, DTYPE r, DTYPE p)
{
    volatile register DTYPE q;
    q  = r + p;
    *x = q - r;
    *y = p - *x;
}
#endif

/*
 * Error free product: x + y = a*b ; where x = fl(a*b) and |y| < |x|
 * From (1) 
 */
#if ! defined(twoprod_enhanced)
static inline
void twoprod(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    volatile DTYPE x0, y0, z1, z2;
    register DTYPE a1, a2, b1, b2; //, z1, z2;

    // [a1, a2] = split(a)
    z1 = __FACTOR*a;
    a1 = z1 - a;
    a1 = z1 - a1;
    a2 = a  - a1;
    // [b1, b2] = split(b)
    z2 = __FACTOR*b;
    b1 = z2 - b;
    b1 = z2 - b1;
    b2 = b  - b1;
    // 
    x0 = a*b;
    y0 = (((x0 - a1*b1) - a2*b1) - a1*b2);
    *y = a2*b2 - y0;
    *x = x0;
    return;
}
#endif

#if ! defined(approx_twodiv_enhanced)
static inline
void approx_twodiv(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    DTYPE v, w;
    *x = a/b;
    twoprod(&v, &w, *x, b);
    *y = (a - v - w)/b;
}
#endif

// Algorithm 4.1 int (1); cascaded summation 
static inline
DTYPE __eft_sum2s(DTYPE *s, DTYPE *c, DTYPE *a, int N, DTYPE s0, DTYPE c0)
{
    DTYPE st, ct, p;
    int k;

    st = s0; ct = c0;
    for (k = 0; k < N; k++) {
        // q + p = s + a[k]
        twosum(&st, &p, st, a[k]);
        // summation of correction terms
        ct += p;
    }
    *s = st;
    *c = ct;
}

static
DTYPE sum2s(DTYPE *a, int N)
{
    DTYPE s, c;
    __eft_sum2s(&s, &c, a, N, 0.0, 0.0);
    return s + c;
}


// Algorithm 4.3 in (1); error-free vector transformation for summation
// Also called "distillation algorithm".
static 
DTYPE vecsum(DTYPE *p, int N)
{
    DTYPE c;
    int k;
    c = 0.0;
    for (k = 1; k < N; k++) {
        // break the connection p0 -- p[k], p1 -- p[k-1] with argument ordering
        twosum(&p[k], &p[k-1], p[k-1], p[k]);
        c += p[k-1];
    }
    return p[N-1] + c;
}

// Algorithm 4.8 in (1); Summation in K-fold precission by (K-1)-fold error-free
// vector transformation
static 
DTYPE sumK(DTYPE *p, int N, int K)
{
    DTYPE s;
    int k, j;

    for (k = 0; k < K-1; k++) {
        // vecsum here
        for (j = 1; j < N; j++) {
            twosum(&p[j], &p[j-1], p[j-1], p[j]);
        }
    }
    s = 0.0;
    for (k = 0; k < N; k++) {
        s += p[k];
    }
    return s;
}

// Algorithm 5.3 in (1); Dot product in twice the working precission; 
static inline
void __eft_dot2s(DTYPE *p, DTYPE *c, DTYPE *a, DTYPE *b, int N, DTYPE p0, DTYPE c0)
{
    DTYPE p1, s1, h1, q1, r1;
    int k;

    p1 = p0; s1 = c0;
    //twoprod(&p1, &s1, a[0], b[0]);
    for (k = 0; k < N; k++) {
        // h + r = a*b
        twoprod(&h1, &r1, a[k], b[k]);
        // p + q = p + h;
        twosum(&p1, &q1, p1, h1);
        // sum error terms
        s1 += q1 + r1;
    }
    *p = p1;
    *c = s1;
}

static
DTYPE dot2s(DTYPE *a, DTYPE *b, int N)
{
    DTYPE p, s;
    __eft_dot2s(&p, &s, a, b, N, 0.0, 0.0);
    return p + s;
}

#if 0
static
DTYPE dot2s(DTYPE *a, DTYPE *b, int N)
{
    DTYPE p, s, h, q, r;
    int k;

    twoprod(&p, &s, a[0], b[0]);
    for (k = 1; k < N; k++) {
        // h + r = a*b
        twoprod(&h, &r, a[k], b[k]);
        // p + q = p + h;
        twosum(&p, &q, p, h);
        // sum error terms
        s += q + r;
    }
    return p + s;
}
#endif

// Algorithm 5.10 in (1); Dot product in K-fold precission
static
DTYPE dotK(DTYPE *x, DTYPE *y, int N, int K)
{
    DTYPE p, h, s, *r;
    int k;
    if (N == 1) {
        twoprod(&p, &h, x[0], y[0]);
        return p + h;
    }    

    r = calloc(2*N, sizeof(DTYPE));
    twoprod(&p, &r[0], x[0], y[0]);
    for (k = 1; k < N; k++) {
        twoprod(&h, &r[k], x[k], y[k]);
        twosum(&p, &r[N+k-1], p, h);
    }
    r[2*N-1] = p;
    s = sumK(r, 2*N, K-1);
    free(r);
    return s;
}

#else
// Complex versions here

#endif

#endif  // __ARMAS_EFT_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
