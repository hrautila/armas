
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
#if defined(__armas_blas1) && defined(__armas_blas2)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


#include "internal.h"
#include "matrix.h"


static inline
DTYPE sqrt_x2y2(DTYPE x, DTYPE y)
{
  DTYPE xabs = __ABS(x);
  DTYPE yabs = __ABS(y);
  DTYPE w = xabs;
  DTYPE z = xabs;
  if (yabs > w) {
    w = yabs;
  }
  if (yabs < z) {
    z = yabs;
  }
  if (z == 0.0) {
    return w;
  }
  return w * __SQRT(1.0 + (z/w)*(z/w));
}

/* 
 * Generates a real elementary reflector H of order n, such
 * that
 *
 *       H * ( alpha ) = ( beta ),   H**T * H = I.
 *           (   x   )   (   0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element real
 * vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v**T ) ,
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element
 * vector.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit cmat.
 *
 * Otherwise  1 <= tau <= 2.
 */
void __compute_householder(__armas_dense_t *a11, __armas_dense_t *x,
                           __armas_dense_t *tau, armas_conf_t *conf)
{
  DTYPE norm_x2, alpha, beta, sign, safmin, rsafmin;
  int nscale = 0;

  norm_x2 = __armas_nrm2(x, conf);
  if (norm_x2 == 0.0) {
    __armas_set(tau, 0, 0, 0.0);
    return;
  }

  alpha = __armas_get(a11, 0, 0);
  sign = __SIGN(alpha) ? -1.0 : 1.0;

  // beta = -(alpha / |alpha|) * ||alpha x||
  //      = -sign(alpha) * sqrt(alpha^2, norm_x2^2)
  //      = -sign * hypot(alpha, norm_x2)
  beta = -sign * sqrt_x2y2(alpha, norm_x2);
  safmin = __SAFEMIN/__EPS;

  if (__ABS(beta) < safmin) {
    //printf("beta [%e] < safemin [%e]\n", beta, safmin);
    // norm_x2, beta may be inaccurate, scale and recompute
    rsafmin = 1.0/safmin;
    do {
      nscale++;
      __armas_scale(x, rsafmin, conf);
      beta *= rsafmin;
      alpha *= rsafmin;
    } while (__ABS(beta) < safmin);

    // now beta in [safmin ... 1.0]
    norm_x2 = __armas_nrm2(x, conf);
    beta = -sign * sqrt_x2y2(alpha, norm_x2);
  }

  // x = x / (alpha-beta)
  __armas_invscale(x, alpha-beta, conf);
  __armas_set(tau, 0, 0, (beta-alpha)/beta);

  while (nscale-- > 0) {
    beta *= safmin;
  }
  __armas_set(a11, 0, 0, beta);
}

void __compute_householder_vec(__armas_dense_t *x, __armas_dense_t *tau, armas_conf_t *conf) {
  __armas_dense_t alpha, x2;

  if (__armas_size(x) == 0) {
    __armas_set(tau, 0, 0, 0.0);
    return;
  }
                              
  __armas_submatrix(&alpha, x, 0, 0, 1, 1);
  if (x->rows == 1) {
    __armas_submatrix(&x2, x, 0, 1, 1, __armas_size(x)-1);
  } else {
    __armas_submatrix(&x2, x, 1, 0, __armas_size(x)-1, 1);
  }
  __compute_householder(&alpha, &x2, tau, conf);
}


void __compute_householder_rev(__armas_dense_t *x, __armas_dense_t *tau, armas_conf_t *conf) {
  __armas_dense_t alpha, x2;

  if (__armas_size(x) == 0) {
    __armas_set(tau, 0, 0, 0.0);
    return;
  }

  if (x->rows == 1) {
    __armas_submatrix(&alpha, x, 0, -1, 1, 1);
    __armas_submatrix(&x2, x, 0, 0, 1, __armas_size(x)-1);
  } else {
    __armas_submatrix(&alpha, x, -1, 0, 1, 1);
    __armas_submatrix(&x2, x, 0, 0, __armas_size(x)-1, 1);
  }
  __compute_householder(&alpha, &x2, tau, conf);
}

/* 
 * Applies a real elementary reflector H to a real m by n matrix A,
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
int __apply_householder2x1(__armas_dense_t *tau, __armas_dense_t *v,
                           __armas_dense_t *a1,  __armas_dense_t *A2,
                           __armas_dense_t *w1,  int flags, armas_conf_t *conf)
{
  DTYPE tval;
  tval = __armas_get(tau, 0, 0);
  if (tval == 0.0) {
    return 0;
  }

  // w1 = a1
  __armas_axpby(w1, a1, 1.0, 0.0, conf);
  if (flags & ARMAS_LEFT) {
    // w1 = a1 + A2.T*v
    __armas_mvmult(w1, A2, v, 1.0, 1.0, ARMAS_TRANSA, conf);
  } else {
    // w1 = a1 + A2*v
    __armas_mvmult(w1, A2, v, 1.0, 1.0, ARMAS_NONE, conf);
  }
  // w1 = tau*w1
  __armas_scale(w1, tval, conf);
  
  // a1 = a1 - w1
  __armas_axpy(a1, w1, -1.0, conf);

  // A2 = A2 - v*w1
  if (flags & ARMAS_LEFT) {
    __armas_mvupdate(A2, v, w1, -1.0, conf);
  } else {
    __armas_mvupdate(A2, w1, v, -1.0, conf);
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
int __apply_householder1x1(__armas_dense_t *tau, __armas_dense_t *v,
                           __armas_dense_t *a1,  __armas_dense_t *A2,
                           __armas_dense_t *w1,  int flags, armas_conf_t *conf) 
{
  DTYPE tval;

  tval = __armas_get(tau, 0, 0);
  if (tval == 0.0) {
    return 0;
  }
  if (flags & ARMAS_LEFT) {
    // w1 = A2.T*v
    __armas_mvmult(w1, A2, v, 1.0, 0.0, ARMAS_TRANS, conf);
    // A2 = A2 - tau*v*w1
    __armas_mvupdate(A2, v, w1, -tval, conf);
  } else {
    // w1 = A2*v
    __armas_mvmult(w1, A2, v, 1.0, 0.0, ARMAS_NONE, conf);
    // A2 = A2 - tau*w1*v
    __armas_mvupdate(A2, w1, v, -tval, conf);
  }

  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

