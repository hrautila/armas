
// Copyright (c) Harri Rautila, 2016-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_hhouse)  && defined(armas_x_hhouse_apply)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1) && defined(armas_x_blas2)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------


//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond


/*
 * References:
 *   Geijn, Van Zee:
 *
 * Notes:
 *      beta = +/- sqrt(alpha^2 - normx^2)
 *
 *      tau  = 2.0/(1 - v^T*v)
 *           = 2.0/(1 - x^T*x/(alpha-beta)^2)
 *           = 2.0/(1 - (normx/(alpha-beta))*(normx/(alpha-beta)))
 *   set
 *      beta = - sign(alpha)*sqrt(alpha^2 - normx^2)
 *
 */

// undefined if x < y;
static inline double sqrt_x2my2(double x, double y)
{
    // sqrt(x^2 - y^2) == sqrt(abs(x) - abs(y))*sqrt(abs(x) + abs(y))
    return ABS(x) * SQRT(1.0 -
                             (ABS(y) / ABS(x)) * (ABS(y) / ABS(x)));
}


/*
 * Generates a hyperbolic real elementary reflector H of order n, such that
 *
 *       H x = +/- beta e_0,  H*J*H^T = J
 *
 * where J is (1,n-1) signature matrix and x is an n-element real vector.
 * H is represented in the form
 *
 *       H = I - tau*J*v*v^T
 *
 * where tau is a real scalar and v is a real n-element vector such that v_0 = 1.0
 *
 * Depending on flag bits generates H such that,
 *   flags == 0
 *     Hx = beta e_0, beta in R
 *   flags == ARMAS_NONNEG
 *     Hx = beta e_0, beta >= 0.0
 *   flags == ARMAS_HHNEGATIVE
 *     Hx == -beta e_0, beta in R
 *   flags == ARMAS_HHNEGATIVE|ARMAS_NONNEG
 *     Hx == -beta e_0, beta >= 0.0
 *
 *  \param [in,out] a11
 *     On entry first element of x-vector. On exit value of beta.
 *  \param [in,out] x
 *     On entry elements 1:n-1 of x. On exit elements 1:n-1 of vector v.
 *  \param [out] tau
 *     On exit value scalar tau in singleton matrix.
 *  \param [in] flags
 *     Flag bits ARMAS_NONNEG, ARMAS_HHNEGATIVE
 *  \param [in] conf
 *     Configuration block.
 *
 *  \retval 0 
 */
int armas_x_hhouse(armas_x_dense_t * a11, armas_x_dense_t * x,
                   armas_x_dense_t * tau, int flags, armas_conf_t * cf)
{
    DTYPE rsafmin, safmin, normx, alpha, beta, sign, delta, scale, t;
    int nscale = 0;

    if (!cf)
        cf = armas_conf_default();

    safmin = SAFEMIN / EPS;

    normx = armas_x_nrm2(x, cf);
    alpha = armas_x_get(a11, 0, 0);
    if (ABS(alpha) < normx) {
        // alpha^2 - normx^2 < 0; sqrt not defined
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (normx == 0.0) {
        armas_x_set(tau, 0, 0, 0.0);
        return 0;
    }
    sign = SIGN(alpha) ? -1.0 : 1.0;
    beta = sqrt_x2my2(alpha, normx);
    // guard agains underflow; see (2)
    if (ABS(beta) < safmin) {
        printf("beta : %e\n", beta);
        // normx, beta may be inaccurate, scale and recompute
        rsafmin = 1.0 / safmin;
        do {
            nscale++;
            armas_x_scale(x, rsafmin, cf);
            beta *= rsafmin;
            alpha *= rsafmin;
        } while (ABS(beta) < safmin);
        // now beta in [safmin ... 1.0]
        normx = armas_x_nrm2(x, cf);
        beta = sqrt_x2my2(alpha, normx);
    }

    switch ((flags & (ARMAS_HHNEGATIVE | ARMAS_NONNEG))) {
    case ARMAS_HHNEGATIVE | ARMAS_NONNEG:
        // solve for H*[alpha; x]^T = [-beta; 0]^T && beta >= 0
        if (alpha >= 0) {
            delta = alpha + beta;
            scale = -1.0 / delta;
        } else {
            // alpha+beta == (alpha^2-beta^2)/(alpha-beta) == normx^2/(alpha-beta)
            delta = (normx / (alpha - beta)) * normx;
            scale = -1.0 / delta;
        }
        break;
    case ARMAS_HHNEGATIVE:
        // solve for H*[alpha; x]^T = [-beta; 0]^T
        beta = sign * beta;
        delta = alpha + beta;
        scale = -1.0 / delta;
        break;
    case ARMAS_NONNEG:
        // solve for H*[alpha; x]^T = [beta; 0]^T && beta >= 0
        if (alpha <= 0) {
            delta = alpha - beta;
            scale = -1.0 / delta;
        } else {
            // alpha-beta == (alpha^2-beta^2)/(alpha+beta) == normx^2/(alpha+beta)
            delta = (normx / (alpha + beta)) * normx;
            scale = -1.0 / delta;
        }
        break;
    default:
        // solve for H*[alpha; x]^T = [beta; 0]^T
        beta = -sign * beta;
        delta = alpha - beta;
        scale = -1.0 / delta;
        break;
    }
    // v = - x/(alpha - beta)
    armas_x_scale(x, scale, cf);
    // tau = 2.0/(1 - v^T v) = 2.0/(1 - (normx/delta)*(normx/delta)
    t = 2.0 / (1.0 - (normx / delta) * (normx / delta));
    armas_x_set(tau, 0, 0, t);

    while (nscale-- > 0) {
        beta *= safmin;
    }
    armas_x_set(a11, 0, 0, beta);
    return 0;
}

/*
 * Applies a real elementary reflector H to a real m by n matrix A,
 * from either the left or the right. 
 *
 *    A = H*A or A = A*H
 *
 *  \param [in] tau
 *     Householder scalar
 *  \param [in] v
 *     Reflector vector 
 *  \param [in,out] a1
 *     On entry top row of m by n matrix A. On exit transformed values.
 *  \param [in,out] A2
 *     On entry rows 2:m-1 of m by n matrix A. On exit transformed values
 *  \param [out] w
 *     Workspace of size vector a1.
 *  \param [in] flags
 *     Flag bits ARMAS_LEFT or ARMAS_RIGHT
 *  \param [in] conf
 *     Configration block
 *
 * If tau = 0, then H is taken to be the I identity matrix.
 *
 * Notes:
 *    A is ( a1 )   a1 := a1 - w1
 *         ( A2 )   A2 := A2 + v*w1
 *                  w1 := tau*(a1 + A2.T*v) if side == LEFT
 *                     := tau*(a1 + A2*v)   if side == RIGHT
 */
int armas_x_hhouse_apply(armas_x_dense_t * tau, armas_x_dense_t * v,
                         armas_x_dense_t * a1, armas_x_dense_t * A2,
                         armas_x_dense_t * w, int flags, armas_conf_t * cf)
{
    DTYPE tval;
    armas_x_dense_t w1;

    if (!cf)
        cf = armas_conf_default();
    if (armas_x_size(w) < armas_x_size(a1)) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
    armas_x_make(&w1, armas_x_size(a1), 1, armas_x_size(a1), armas_x_data(w));

    tval = armas_x_get(tau, 0, 0);
    if (tval == 0.0) {
        return 0;
    }
    // w1 = a1
    armas_x_copy(&w1, a1, cf);
    if (flags & ARMAS_LEFT) {
        // w1 = a1 + A2.T*v
        armas_x_mvmult(ONE, &w1, ONE, A2, v, ARMAS_TRANSA, cf);
    } else {
        // w1 = a1 + A2*v
        armas_x_mvmult(ONE, &w1, ONE, A2, v, ARMAS_NONE, cf);
    }
    // w1 = tau*w1
    armas_x_scale(&w1, tval, cf);

    // a1 = a1 - w1
    armas_x_axpy(a1, -ONE, &w1, cf);

    // A2 = A2 + v*w1
    if (flags & ARMAS_LEFT) {
        armas_x_mvupdate(ONE, A2, ONE, v, &w1, cf);
    } else {
        armas_x_mvupdate(ONE, A2, ONE, &w1, v, cf);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
