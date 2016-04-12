
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! two norm of vector

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_nrm2) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
//! \endcond

// return vector norm; naive version
static inline
ABSTYPE __vec_nrm2(const mvec_t *X,  int N)
{
  register int i, k;
  register ABSTYPE c0, c1, c2, c3, a0, a1, a2, a3;
  register DTYPE z0, z1, z2, z3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    z0 = X->md[(i+0)*X->inc];
    z1 = X->md[(i+1)*X->inc];
    z2 = X->md[(i+2)*X->inc];
    z3 = X->md[(i+3)*X->inc];
    a0 = __ABS(z0);
    a1 = __ABS(z1);
    a2 = __ABS(z2);
    a3 = __ABS(z3);
    c0 += a0*a0;
    c1 += a1*a1;
    c2 += a2*a2;
    c3 += a3*a3;
  }    
  if (i == N)
    goto update;

  k = i*X->inc;
  switch (N-i) {
  case 3:
    a0 = __ABS(X->md[k]);
    c0 += a0*a0;
    k += X->inc;
  case 2:
    a1 = __ABS(X->md[k]);
    c1 += a1*a1;
    k += X->inc;
  case 1:
    a2 = __ABS(X->md[k]);
    c2 += a2*a2;
  }
 update:
  return __SQRT(c0 + c1 + c2 + c3);
}

/*
 * Nick Higham in Accurrancy and Precision:
 *   For about half of all machine numbers x, value of x^2 either
 *   underflows or overflows
 *
 * Overflow is avoided by summing squares of scaled numbers and
 * then multiplying then with the scaling factor. Following is
 * is by Hammarling and included in BLAS reference libary.
 */

/*
 * Scaled summation of squares
 * ---------------------------
 *
 *  X = {x0, x1, x2, x3} and |x0| > |x_i|, i!=0
 * 
 *  then w_0 = amax(X)
 *       S_0 = 1.0 + sum (|x_k|/|w|)^2, where k != iamax(X)
 *
 *  X_1  = X_0 + {x4} and |x_4| > amax(X_0)
 *
 *   then w_1 = |x_4|
 *        S_1 = 1.0 + (w_0/|x_4|)^2 * S_0
 *
 *  [nrm2] := nrm2(X, N)
 *     w = |x_0|
 *     S = 1.0
 *     for k = 1 : N-1
 *        |x_k| <= |w|:
 *           S = S + (|x_k|/|w|)^2
 *        |x_k| >  |w|:
 *           S = 1.0 + (|w|/|x_k|)^2 * S
 *           w = |x_k|
 *      return w * sqrt(S)
 *
 *   updated sum-of-squares
 *   [w, S] := ssumsq(X, N, w_0, S_0)
 *      w = w_0
 *      S = S_0
 *      for k = 0 : N-1:
 *        |x_k| <= |w|:
 *           S = S + (|x_k|/|w|)^2
 *        |x_k| >  |w|:
 *           S = 1.0 + (|w|/|x_k|)^2 * S
 *           w = |x_k|
 *      return [w, S]
 */
static inline
ABSTYPE __vec_nrm2_scaled(const mvec_t *X,  int N)
{
  register int i;
  register ABSTYPE a0, sum, scale;

  sum = __ABSONE;
  scale = __ABSZERO;
  for (i = 0; i < N; i += 1) {
    if (X->md[(i+0)*X->inc] != __ZERO) {
      a0 = __ABS(X->md[(i+0)*X->inc]);
      if (a0 > scale) {
        sum = __ONE + sum * ((scale/a0)*(scale/a0));
        scale = a0;
      } else {
        sum = sum + (a0/scale)*(a0/scale);
      }
    }
  }    
  return scale*__SQRT(sum);
}

/**
 * @brief Norm2 of vector, \f$ ||x||_2 \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @ingroup blas1
 */
ABSTYPE __armas_nrm2(const __armas_dense_t *x, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return __ABSZERO;
  }

  if (__armas_size(x) == 0) {
    return __ABSZERO;
  }
  if (__armas_size(x) == 1) {
    return __ABS(x->elems[0]);
  }
  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  if (conf && (conf->optflags & ARMAS_ONAIVE)) {
    return __vec_nrm2(&X, __armas_size(x));
  }
  return __vec_nrm2_scaled(&X, __armas_size(x));
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:

