
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Roots of quadratic function

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qdroots) && defined(armas_x_discriminant)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

/*
 * This only works for double precision reals. (At the moment).
 */

#if defined(__x86_64__) && defined(__FMA__) && defined(FLOAT64)

#include <immintrin.h>

/*
 * Compute the discriminant of quadratic equation extra-precisely 
 * using the Fused-Multiply-Add instructions on x86_64 architecture
 *
 * Compute P = b*b; dP = b*b - P;  Q = a*c; dQ = a*c - Q
 * and discriminant D = b*b - a*c = (P - Q) + (dP - dQ)
 *
 * Test for discriminant size is not included as it creates on optimization level -O3
 * more instructions for the test than the additional FP instructions not needed
 * when round off does not cancel significant bits.
 */
static inline
DTYPE discriminant(DTYPE a, DTYPE b, DTYPE c)
{
    register __m256d y0, y1, y2, y3;
    y0 = _mm256_set_pd(a, b, a, b);     // y0 = {b, a}
    y1 = _mm256_set_pd(c, b, c, b);     // y1 = {b, c}
    y2 = _mm256_set1_pd(0.0);           // y2 = {0, 0}
    y2 = _mm256_fmadd_pd(y0, y1, y2);   // y2 = {b*b, a*c} + {0, 0} = {P, Q}
    y3 = _mm256_fmsub_pd(y0, y1, y2);   // y3 = {b*b, a*c} - {P, Q} = {dP, dQ}
    y2 = _mm256_hsub_pd(y2, y2);        // y2 = {P-Q, ...}
    y3 = _mm256_hsub_pd(y3, y3);        // y3 = {dP-dQ, ...}
    y0 = _mm256_add_pd(y2, y3);         // y0 = {(P-Q)+(dP-dQ), ...}
    return y0[0];
}

#else /* NOT (defined(__x86_64__) && defined(__FMA__)) */

// The PI security parameter in (1) round-off cancelation 
#define ROUNDOFF_SEC_CONST 3
#if defined(FLOAT32)
#define BIG_CONST ((1 << 14) + 1)
#else
#define BIG_CONST ((1 << 27) + 1)
#endif

/*
 * Break 53 sig. bit DTYPE to two 26 sig. bit parts.
 */
static inline
DTYPE break2(DTYPE * xh, DTYPE * xt, DTYPE x)
{
    register DTYPE bigx, y;
    bigx = x * BIG_CONST;
    y = x - bigx;
    *xh = y + bigx;
    *xt = x - (*xh);
    return y;                   // don't allow optimizing away
}

/*
 * Compute the discriminant of quadratic equation extra-precisely if
 * necessary to ensure accuracy to the last sig. bit or two.
 */
static
DTYPE discriminant(DTYPE a, DTYPE b, DTYPE c)
{
    DTYPE d, e, ah, at, bh, bt, ch, ct, p, q, dp, dq;
    d = b * b - a * c;
    e = b * b + a * c;
    // good enough ?
    if (ROUNDOFF_SEC_CONST * ABS(d) > e)
        return d;

    p = b * b;
    q = a * c;
    break2(&ah, &at, a);
    break2(&bh, &bt, b);
    break2(&ch, &ct, c);

    dp = ((bh * bh - p) + 2 * bh * bt) + bt * bt;
    dq = ((ah * ch - q) + (ah * ct + at * ch)) + at * ct;
    d = (p - q) + (dp - dq);
    return d;
}
#endif /* defined(__x86_64__) && defined(__FMA__) */


/**
 * @brief Compute roots of quadratic equation.
 *
 * Computes roots of quadratic equation \f$ A*x^2 - 2B*x + C = 0 \f$ with
 * precission as described in
 *   W. Kahan,
 *   On the Cost of Floating-Point Computation Without Extra-Precise Arithmetic
 *   2004"
 *
 * @param x1, x2 [out]
 *	Computed roots, |x1| >= |x2|
 * @param a, b, c [in]
 *	Coefficient of quadratic equation.
 * @return
 *	zero if roots are real and non-zero if roots are complex or coincident real.
 *
 * For details see
 *   W. Kahan, 2004
 */
int armas_x_qdroots(DTYPE * x1, DTYPE * x2, DTYPE a, DTYPE b, DTYPE c)
{
    DTYPE d, r, s, signb;
    d = discriminant(a, b, c);
    if (d < 0.0) {
        r = b / a;
        s = SQRT(ABS(d)) / a;
        *x1 = -r / 2.0;
        *x2 = s / 2.0;
        return 1;
    }
    signb = SIGN(b) ? -1.0 : 1.0;
    s = SQRT(d) * (signb + (DTYPE) (b == ZERO)) + b;
    *x1 = s / a;
    *x2 = c / s;
    return 0;
}

/**
 * \brief Compute, with precission, value of discriminant in \f$ A x^2 + 2B x + C \f$
 *
 * \param[out] dval
 *      Value of discriminant
 * \param[in] a, b, c
 *      Coefficients of quadratic function.
 *
 * For details see
 *   W. Kahan, On the Cost of Floating-Point Computation Without Extra-Precise Arithmetic, 2004
 */
void armas_x_discriminant(DTYPE * dval, DTYPE a, DTYPE b, DTYPE c)
{
    *dval = discriminant(a, b, c);
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
