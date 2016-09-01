
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_hhouse)  && defined(__armas_hhouse_apply)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_blas1) && defined(__armas_blas2)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


//! \cond
#include "internal.h"
#include "matrix.h"
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
static inline
double sqrt_x2my2(double x, double y)
{
    // sqrt(x^2 - y^2) == sqrt(abs(x) - abs(y))*sqrt(abs(x) + abs(y))
    return __ABS(x)*__SQRT(1.0 - (__ABS(y)/__ABS(x))*(__ABS(y)/__ABS(x)));
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
int __armas_hhouse(__armas_dense_t *a11, __armas_dense_t *x,
                   __armas_dense_t *tau, int flags, armas_conf_t *conf)
{
    DTYPE rsafmin, safmin, normx, alpha, beta, sign, delta, scale, t;
    int nscale = 0;
    
    if (!conf)
        conf = armas_conf_default();

    safmin = __SAFEMIN/__EPS;
    
    normx = __armas_nrm2(x, conf);  
    alpha   = __armas_get(a11, 0, 0);
    if (__ABS(alpha) < normx) {
        // alpha^2 - normx^2 < 0; sqrt not defined
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    if (normx == 0.0) {
        __armas_set(tau, 0, 0, 0.0);
        return 0;
    }
    sign = __SIGN(alpha) ? -1.0 : 1.0;
    beta = sqrt_x2my2(alpha, normx); 
    // guard agains underflow; see (2)
    if (__ABS(beta) < safmin) {
        printf("beta : %e\n", beta);
        // normx, beta may be inaccurate, scale and recompute
        rsafmin = 1.0/safmin;
        do {
            nscale++;
            __armas_scale(x, rsafmin, conf);
            beta *= rsafmin;
            alpha *= rsafmin;
        } while (__ABS(beta) < safmin);
        // now beta in [safmin ... 1.0]
        normx = __armas_nrm2(x, conf);
        beta  = sqrt_x2my2(alpha, normx);
    }

    switch ((flags & (ARMAS_HHNEGATIVE|ARMAS_NONNEG))) {
    case ARMAS_HHNEGATIVE|ARMAS_NONNEG:
        // solve for H*[alpha; x]^T = [-beta; 0]^T && beta >= 0
        if (alpha >= 0) {
            delta = alpha + beta;
            scale = - 1.0/delta;
        } else {
            // alpha + beta == (alpha^2-beta^2)/(alpha-beta) == normx^2/(alpha-beta)
            delta = (normx/(alpha-beta))*normx;
            scale = - 1.0/delta;
        }
        break;
    case ARMAS_HHNEGATIVE:
        // solve for H*[alpha; x]^T = [-beta; 0]^T
        beta = sign * beta;
        delta = alpha + beta;
        scale = - 1.0/delta;
        break;
    case ARMAS_NONNEG:
        // solve for H*[alpha; x]^T = [beta; 0]^T && beta >= 0
        if (alpha <= 0) {
            delta = alpha - beta;
            scale = - 1.0/delta;
        } else {
            // alpha - beta == (alpha^2-beta^2)/(alpha+beta) == normx^2/(alpha+beta)
            delta = (normx/(alpha+beta))*normx;
            scale = - 1.0/delta;
        }
        break;
    default:
        // solve for H*[alpha; x]^T = [beta; 0]^T
        beta = - sign * beta;
        delta = alpha - beta;
        scale = - 1.0/delta;
        break;
    }
    // v = - x/(alpha - beta)
    __armas_scale(x, scale, conf);
    // tau = 2.0/(1 - v^T v) = 2.0/(1 - (normx/delta)*(normx/delta)
    t = 2.0/(1.0 - (normx/delta) * (normx/delta));
    __armas_set(tau, 0, 0, t);

    while (nscale-- > 0) {
        beta *= safmin;
    }
    __armas_set(a11, 0, 0, beta);
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
int __armas_hhouse_apply(__armas_dense_t *tau, __armas_dense_t *v,
                         __armas_dense_t *a1,  __armas_dense_t *A2,
                         __armas_dense_t *w,  int flags, armas_conf_t *conf)
{
    DTYPE tval;
    __armas_dense_t w1;
    
    if (!conf)
        conf = armas_conf_default();
    if (__armas_size(w) < __armas_size(a1)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    __armas_make(&w1, __armas_size(a1), 1, __armas_size(a1), __armas_data(w));

    tval = __armas_get(tau, 0, 0);
    if (tval == 0.0) {
        return 0;
    }

    // w1 = a1
    __armas_copy(&w1, a1, conf);
    if (flags & ARMAS_LEFT) {
        // w1 = a1 + A2.T*v
        __armas_mvmult(&w1, A2, v, 1.0, 1.0, ARMAS_TRANSA, conf);
    } else {
        // w1 = a1 + A2*v
        __armas_mvmult(&w1, A2, v, 1.0, 1.0, ARMAS_NONE, conf);
    }
    // w1 = tau*w1
    __armas_scale(&w1, tval, conf);
  
    // a1 = a1 - w1
    __armas_axpy(a1, &w1, -1.0, conf);

    // A2 = A2 + v*w1
    if (flags & ARMAS_LEFT) {
        __armas_mvupdate(A2, v, &w1, 1.0, conf);
    } else {
        __armas_mvupdate(A2, &w1, v, 1.0, conf);
    }
    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

