
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__householder) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1) && defined(armas_x_blas2)
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
 * Generates a real elementary reflector H of order n, such that
 *
 *       H * ( alpha ) = ( +/-beta ),   H^T*H = H*H = I.
 *           (   x   )   (      0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element real
 * vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v^T ) ,
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element vector.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit matrix.
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
 * Notes:
 *  Setting
 *    ( v1 ) = (alpha) - (beta)  => v1 = 1
 *    ( v2 )   (  x  )   ( 0  )     v2 = x / (alpha - beta)
 * 
 *  and 
 *    tau = (beta - alpha)/beta
 *    normx = ||x||_2
 *
 *  Selecting beta in classic version (always adding factors with same sign, no cancelation):
 *
 *    beta = - sign(alpha)*sqrt(alpha^2 + normx^2)
 *
 *  If always positive beta needed then:
 *
 *    beta = sqrt(alpha^2 + normx^2)
 *
 *    (alpha - beta) = (alpha - beta)(alpha + beta)/(alpha + beta)
 *                   = (alpha^2 - beta^2)/(alpha + beta)
 *                   = - normx^2/(alpha + beta)
 *                   = - normx*(normx/(alpha + beta))
 *   
 *     tau = (beta - alpha)/beta 
 *         = -(alpha - beta)/beta
 *         = normx^2/(alpha + beta)/beta
 *         = (normx/beta)*(normx/(alpha + beta))
 * 
 *    here: normx <= beta && alpha <= beta
 *
 * References: 
 * (1) Golub, Van Load: Matrix Computation, 4th edition, Section 5.1.2 and 5.1.3
 * (2) Demmel, Hoemmen, Hida & Riedy: Lapack Working Note #203, section 3
 *
 */

static 
int __internal_householder(armas_x_dense_t *a11, armas_x_dense_t *x,
                           armas_x_dense_t *tau, int flags, armas_conf_t *conf)
{
    DTYPE normx, alpha, beta, delta, sign, safmin, rsafmin, t;
    int nscale = 0;

    normx = armas_x_nrm2(x, conf);
    if (normx == 0.0) {
        armas_x_set(tau, 0, 0, 0.0);
        return 0;
    }

    alpha = armas_x_get_unsafe(a11, 0, 0);
    sign = __SIGN(alpha) ? -1.0 : 1.0;
    
    beta = __HYPOT(alpha, normx);
    safmin = __SAFEMIN/__EPS;
    
    if (beta < safmin) {
        // normx, beta may be inaccurate, scale and recompute
        rsafmin = 1.0/safmin;
        do {
            nscale++;
            armas_x_scale(x, rsafmin, conf);
            beta *= rsafmin;
            alpha *= rsafmin;
        } while (beta < safmin);

        // now beta in [safmin ... 1.0]
        normx = armas_x_nrm2(x, conf);
        beta = __HYPOT(alpha, normx);
    }

    switch ((flags & (ARMAS_HHNEGATIVE|ARMAS_NONNEG))) {
    case ARMAS_HHNEGATIVE|ARMAS_NONNEG:
        // compute for Hx = [-beta; 0]^T && beta >= 0 (ARMAS_COMPLEMENT)
        if (alpha >= 0.0) {
            delta = alpha + beta;
        } else {
            // alpha < 0 && beta > 0  (delta = alpha + beta)
            delta = - normx * (normx/(alpha - beta));
        }
        t = delta/beta;
        break;
    case ARMAS_HHNEGATIVE:
        // compute for Hx = [-beta; 0]^T 
        beta = sign * beta;
        delta = alpha + beta;
        t     = delta/beta;
        break;
    case ARMAS_NONNEG:
        // compute for Hx = [beta; 0]^T && beta >= 0; see (2)
        if (alpha <= 0.0) {
            delta = alpha - beta;
        } else {
            // alpha > 0 && beta > 0  (delta = alpha - beta)
            delta = - normx * (normx/(alpha + beta));
        }
        t = -delta/beta;
        break;
    default:
        // compute for Hx = [beta; 0]^T ; classic LAPACK
        beta = -sign * beta;
        delta = alpha - beta;
        t = -delta/beta;
        break;
    }
    armas_x_scale(x, 1.0/delta, conf);
    armas_x_set_unsafe(tau, 0, 0, t);
    
    while (nscale-- > 0) {
        beta *= safmin;
    }
    armas_x_set_unsafe(a11, 0, 0, beta);
    
    return 0;
}

/*
 * Compute the unscaled Householder reflector H = I - 2*v*v^T such that Hx = beta*e_0
 * 
 */
static 
int __hhcompute_unscaled(armas_x_dense_t *x0, armas_x_dense_t *x1, 
                         armas_x_dense_t *tau, int flags, armas_conf_t *conf)
{
    DTYPE normx, x0val, alpha, beta, delta, sign;

    normx = armas_x_nrm2(x1, conf);
    if (normx == 0.0) {
        armas_x_set_unsafe(tau, 0, 0, armas_x_get_unsafe(x0, 0, 0));
        return 0;
    }

    x0val = armas_x_get_unsafe(x0, 0, 0);
    sign  = __SIGN(x0val) ? -1.0 : 1.0;
    
    beta = __HYPOT(x0val, normx);
    
    switch (flags & (ARMAS_NONNEG)) {
    case ARMAS_NONNEG:
        if (x0val <= 0.0) {
            alpha = x0val - beta;
            delta = __HYPOT(normx, alpha);
        } else {
            alpha = - normx * (normx/(x0val + beta));
            delta = __HYPOT(normx, alpha);
        }
        break;
    default:
        beta = -sign*beta;
        delta = __HYPOT(normx, x0val - beta);
        alpha = x0val - beta;
        break;
    }

    armas_x_set_unsafe(x0, 0, 0, alpha/delta);
    armas_x_scale(x1, 1.0/delta, conf);
    armas_x_set_unsafe(tau, 0, 0, beta);
    return 0;
}


/**
 * Generates a real elementary reflector H of order n, such that
 *
 *       H x = (+/-beta e_0)^T,  H H = I 
 *
 * where x is an n-element real vector. H is represented in the form
 *
 *       H = I - tau*v*v^T  or H = I - 2*v*v^T
 *
 * where tau is a real scalar and v is a real n-element vector.
 * If flag ARMAS_UNIT is used to generated scaled reflector vectors then `tau`
 * holds the scaling factor. 
 *
 * Depending on flag bits generates H such that,
 *   flags == 0
 *     Hx = (beta e_0)^T, beta in R  
 *   flags == ARMAS_NONNEG
 *     Hx = (beta e_0)^T, beta >= 0.0
 *
 *  \param [in,out] a11
 *     On entry first element of x-vector. On exit the first element if reflector vector v.
 *     If ARAMS_UNIT is set then a11 holds on exit the value of beta.
 *  \param [in,out] x
 *     On entry elements 1:n-1 of x. On exit elements 1:n-1 of vector `v`.
 *  \param [out] tau
 *     If ARMAS_UNIT is set then on exit value scalar `tau` in singleton matrix. Otherwise the
 *     the scalar value of `beta`.
 *  \param [in] flags
 *     Use ARMAS_NONNEG to generate non-negative beta values. 
 *     Use ARMAS_UNIT to generate scaled reflector vectors (stardard LAPACK).
 *  \param [in] conf
 *     Configuration block.
 *
 *  \retval 0 
 */
int armas_x_house(armas_x_dense_t *a11, armas_x_dense_t *x,
                  armas_x_dense_t *tau, int flags, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();
    // scaled householder reflector (standard lapack version)
    if ((flags & ARMAS_UNIT) != 0)
        return __internal_householder(a11, x, tau, flags, conf);
    // unscaled householder reflector
    return __hhcompute_unscaled(a11, x, tau, flags, conf);
}

int armas_x_house_vec(armas_x_dense_t *x, armas_x_dense_t *tau, int flags, armas_conf_t *conf)
{
    armas_x_dense_t alpha, x2;
    if (!conf)
        conf = armas_conf_default();

    armas_x_submatrix(&alpha, x, 0, 0, 1, 1);
    if (x->rows == 1) {
        armas_x_submatrix(&x2, x, 0, 1, 1, armas_x_size(x)-1);
    } else {
        armas_x_submatrix(&x2, x, 1, 0, armas_x_size(x)-1, 1);
    }
    return armas_x_house(&alpha, &x2, tau, flags, conf);
}

// library internal 
void __compute_householder(armas_x_dense_t *a11, armas_x_dense_t *x,
                             armas_x_dense_t *tau, armas_conf_t *conf)
{
    int flags = (conf->optflags & ARMAS_NONNEG) != 0 ? ARMAS_NONNEG : 0;
    __internal_householder(a11, x, tau, flags, conf);
}

// library internal 
void __compute_householder_vec(armas_x_dense_t *x, armas_x_dense_t *tau, armas_conf_t *conf)
{
    armas_x_dense_t alpha, x2;

    if (armas_x_size(x) == 0) {
        armas_x_set(tau, 0, 0, 0.0);
        return;
    }
                              
    int flags = (conf->optflags & ARMAS_NONNEG) != 0 ? ARMAS_NONNEG : 0;

    armas_x_submatrix(&alpha, x, 0, 0, 1, 1);
    if (x->rows == 1) {
        armas_x_submatrix(&x2, x, 0, 1, 1, armas_x_size(x)-1);
    } else {
        armas_x_submatrix(&x2, x, 1, 0, armas_x_size(x)-1, 1);
    }
    //__compute_householder(&alpha, &x2, tau, conf);
    __internal_householder(&alpha, &x2, tau, flags, conf);
}


// library internal 
void __compute_householder_rev(armas_x_dense_t *x, armas_x_dense_t *tau, armas_conf_t *conf)
{
    armas_x_dense_t alpha, x2;

    if (armas_x_size(x) == 0) {
        armas_x_set(tau, 0, 0, 0.0);
        return;
    }

    int flags = (conf->optflags & ARMAS_NONNEG) != 0 ? ARMAS_NONNEG : 0;

    if (x->rows == 1) {
        armas_x_submatrix(&alpha, x, 0, -1, 1, 1);
        armas_x_submatrix(&x2, x, 0, 0, 1, armas_x_size(x)-1);
    } else {
        armas_x_submatrix(&alpha, x, -1, 0, 1, 1);
        armas_x_submatrix(&x2, x, 0, 0, armas_x_size(x)-1, 1);
    }
    //__compute_householder(&alpha, &x2, tau, conf);
    __internal_householder(&alpha, &x2, tau, flags, conf);
}

/* 
 * Aplies a real elementary reflector H to a real m by n matrix A,
 * from either the left or the right. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v.T )
 *                     ( v )
 *
 * where tau is a real scalar and v is a real vector.
 *
 * If tau = 0, then H is taken to be the unit cmat.
 *
 * A is ( a1 )   a1 := a1 - w1
 *      ( A2 )   A2 := A2 - v*w1
 *               w1 := tau*(a1 + A2.T*v) if side == LEFT
 *                  := tau*(a1 + A2*v)   if side == RIGHT
 *
 * Intermediate work space w1 required as parameter, no allocation.
 */
int __apply_householder2x1(armas_x_dense_t *tau, armas_x_dense_t *v,
                           armas_x_dense_t *a1,  armas_x_dense_t *A2,
                           armas_x_dense_t *w1,  int flags, armas_conf_t *conf)
{
    DTYPE tval;
    tval = tau ? armas_x_get(tau, 0, 0) : __TWO;
    if (tval == 0.0) {
        return 0;
    }

    // w1 = a1
    armas_x_axpby(__ZERO, w1, __ONE, a1, conf);
    if (flags & ARMAS_LEFT) {
        // w1 = a1 + A2.T*v
        armas_x_mvmult(__ONE, w1, __ONE, A2, v, ARMAS_TRANSA, conf);
    } else {
        // w1 = a1 + A2*v
        armas_x_mvmult(__ONE, w1, __ONE, A2, v, ARMAS_NONE, conf);
    }
    // w1 = tau*w1
    armas_x_scale(w1, tval, conf);
  
    // a1 = a1 - w1
    armas_x_axpy(a1, -__ONE, w1, conf);

    // A2 = A2 - v*w1
    if (flags & ARMAS_LEFT) {
        armas_x_mvupdate(A2, -__ONE, v, w1, conf);
    } else {
        armas_x_mvupdate(A2, -__ONE, w1, v, conf);
    }
    return 0;
}


/*
 *  Apply elementary Householder reflector v to matrix A2.
 *
 *    H = I - tau*v*v.t;
 *
 *  RIGHT:  A = A*H = A - tau*A*v*v.T = A - tau*w1*v.T
 *  LEFT:   A = H*A = A - tau*v*v.T*A = A - tau*v*A.T*v = A - tau*v*w1
 */
int __apply_householder1x1(armas_x_dense_t *tau, armas_x_dense_t *v,
                           armas_x_dense_t *A, armas_x_dense_t *w,  int flags, armas_conf_t *conf) 
{
    DTYPE tval;

    tval = tau ? armas_x_get(tau, 0, 0) : __TWO;
    if (tval == __ZERO) {
        return 0;
    }
    if (flags & ARMAS_LEFT) {
        // w = A.T*v
        armas_x_mvmult(__ZERO, w, __ONE, A, v, ARMAS_TRANS, conf);
        // A = A - tau*v*w
        armas_x_mvupdate(A, -tval, v, w, conf);
    } else {
        // w = A*v
        armas_x_mvmult(__ZERO, w, __ONE, A, v, ARMAS_NONE, conf);
        // A = A - tau*w*v
        armas_x_mvupdate(A, -tval, w, v, conf);
    }
    return 0;
}

/**
 * Applies a real elementary reflector H to a real m by n matrix A,
 * from either the left or the right. 
 *
 *  ( a1^t ) =  H * ( a1^T ) 
 *  ( A2   )        ( A2   )
 *
 *  \param [in,out] a1
 *     On entry top row/leftmost column of matrix A. On exit transformed values.
 *  \param [in,out] A2
 *     On entry rows/columns 2:m of matrix A. On exit transformed values
 *  \param [in] v
 *     Reflector vector; of length len(a1)-1 if unit scaled reflector. Otherwise of lenght len(a1).
 *  \param [in] tau
 *     Householder scalar or null if reflector vector is unscaled
 *  \param [in] w
 *     Workspace, at least of size len(a1) elements
 *  \param [in] flags
 *     Flag bits ARMAS_LEFT or ARMAS_RIGHT
 *  \param [in] conf
 *     Configuration block
 *
 */
int armas_x_houseapply2x1(armas_x_dense_t *a1,  armas_x_dense_t *A2,
                          armas_x_dense_t *tau, armas_x_dense_t *v,
                          armas_x_dense_t *w,  int flags, armas_conf_t *conf)
{
    armas_x_dense_t w1;
    if (!conf)
        conf = armas_conf_default();
    if (armas_x_size(w) < armas_x_size(a1)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    armas_x_make(&w1, armas_x_size(a1), 1, armas_x_size(a1), armas_x_data(w));
    return __apply_householder2x1(tau, v, a1, A2, &w1, flags, conf);
}

/**
 * Applies a real elementary reflector H to matrix A either from left or right. 
 *
 *    A = H * A  or A = A * H
 *
 *  \param [in,out] A
 *     On entry matrix A. On exit transformed values.
 *  \param [in] tau
 *     Householder scalar for unit scaled reflector or null for unscaled reflector.
 *  \param [in] v
 *     Reflector vector; for unscaled reflector then length is rows(A) if applying from right
 *     and if applying from LEFT then length is cols(A). If unit scaled reflector then length one 
 *     element shorter.
 *  \param [in] w
 *     Workspace, at least of size rows(A) for ARMAS_RIGHT or cols(A) for ARMAS_LEFT. If parameter
 *     matrix A is vector then workspace is not needed.
 *  \param [in] flags
 *     Flag bits ARMAS_LEFT or ARMAS_RIGHT
 *  \param [in] conf
 *     Configuration block
 *
 */
int armas_x_houseapply(armas_x_dense_t *A,
                       armas_x_dense_t *tau, armas_x_dense_t *v,
                       armas_x_dense_t *w, int flags, armas_conf_t *conf)
{
    armas_x_dense_t w1, a1, A2;
    if (!conf)
        conf = armas_conf_default();
    
    int scaled = !tau || (flags & ARMAS_UNIT) != 0 ? 1 : 0;
    
    if (armas_x_isvector(A)) {
        DTYPE alpha;
        if (scaled) {
            if (armas_x_size(A)-1 != armas_x_size(v)) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
            armas_x_subvector(&A2, A, 1, armas_x_size(A)-1);
            alpha = armas_x_dot(&A2, v, conf);
            alpha += armas_x_get_unsafe(A, 0, 0);
            alpha *= armas_x_get_unsafe(tau, 0, 0);
            armas_x_set_unsafe(A, 0, 0, armas_x_get_unsafe(A, 0, 0) - alpha);
            armas_x_axpy(&A2, -alpha, v, conf);
        } else {
            if (armas_x_size(A) != armas_x_size(v)) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
            alpha = __TWO * armas_x_dot(A, v, conf);
            armas_x_axpy(A, -alpha, v, conf);
        }
        return 0;
    }
    // if tau != __nil then unit scaled reflector otherwise unscaled reflector
    int ok = (flags & ARMAS_RIGHT) != 0
        ? (scaled && A->rows-1 == armas_x_size(v)) || (!scaled && A->rows  == armas_x_size(v))
        : (scaled && A->cols-1 == armas_x_size(v)) || (!scaled && A->cols  == armas_x_size(v));

    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (!w || armas_x_size(w) < armas_x_size(v) + scaled) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    
    if (scaled) {
        switch (flags & (ARMAS_RIGHT|ARMAS_LEFT)) {
        case ARMAS_RIGHT:
            armas_x_submatrix(&a1, A, 0, 0, A->rows, 1);
            armas_x_submatrix(&A2, A, 0, 1, A->rows, A->cols-1);
            break;
        case ARMAS_LEFT:
        default:
            armas_x_submatrix(&a1, A, 0, 0, 1, A->cols);
            armas_x_submatrix(&A2, A, 1, 0, A->rows-1, A->cols);
            break;
        }
        armas_x_make(&w1, armas_x_size(&a1), 1, armas_x_size(&a1), armas_x_data(w));
        return __apply_householder2x1(tau, v, &a1, &A2, &w1, flags, conf);
    }
    // unscaled householder reflector here
    int   n = (flags & ARMAS_RIGHT) != 0 ? A->rows : A->cols;   
    armas_x_make(&w1, n, 1, n, armas_x_data(w));
    // tau is null pointer here
    return __apply_householder1x1(tau, v, A, &w1, flags, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

