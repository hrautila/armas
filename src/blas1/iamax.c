
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Absolute maximum

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_iamax) && defined(__armas_amax)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"

// return index of max absolute value
static inline
int __vec_iamax(const mvec_t *X,  int N)
{
  register int i, ix, n;
  register ABSTYPE max, c0, c1;

  if (N <= 1)
    return 0;

  max = 0.0;
  ix = 0;
  for (i = 0; i < N-1; i += 2) {
    c0 = __ABS(X->md[(i+0)*X->inc]);
    c1 = __ABS(X->md[(i+1)*X->inc]);
    if (c1 > c0) {
      n = 1;
      c0 = c1;
    }
    if (c0 > max) {
      ix = i+n;
      max = c0;
    }
    n = 0;
  }    
  if (i < N) {
    c0 = __ABS(X->md[i*X->inc]);
    ix = c0 > max ? N-1 : ix;
  }
  return ix;
}

static inline
int __vec_iamin(const mvec_t *X,  int N)
{
  register int i, ix, n;
  register ABSTYPE min, c0, c1;

  if (N <= 1)
    return 0;

  min = __ABS(X->md[0]);
  ix = 0;
  for (i = 0; i < N-1; i += 2) {
    c0 = __ABS(X->md[(i+0)*X->inc]);
    c1 = __ABS(X->md[(i+1)*X->inc]);
    if (c1 < c0) {
      n = 1;
      c0 = c1;
    }
    if (c0 < min) {
      ix = i+n;
      min = c0;
    }
    n = 0;
  }    
  if (i < N) {
    c0 = __ABS(X->md[i*X->inc]);
    ix = c0 < min ? N-1 : ix;
  }
  return ix;
}

/**
 * @brief Index of \f$ \max_{k} |x| \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @retval >= 0 index of maximum element
 * @retval -1  error, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_iamax(const __armas_dense_t *x, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  
  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  // assume column vector
  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  return __vec_iamax(&X, __armas_size(x));
}

/**
 * @brief Maximum absolute value of vector.
 */
ABSTYPE __armas_amax(const __armas_dense_t *x, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  int imax = __armas_iamax(x, conf);
  if (imax != -1) {
    int r = x->rows == 0 ? 0    : imax;
    int c = x->rows == 0 ? imax : 0;
    return __ABS(__armas_get(x, r, c));
  }
  return __ZERO;
}


/**
 * @brief Index of \f$ \min_{k} |x| \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @retval >= 0 index of minimum element
 * @retval -1  error, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_iamin(const __armas_dense_t *x, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  
  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  // assume column vector
  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  return __vec_iamin(&X, __armas_size(x));
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:

