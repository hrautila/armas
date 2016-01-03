
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Swap vectors

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_swap) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"


static
void __vec_swap(mvec_t *X,  mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double y0, y1, y2, y3, x0, x1, x2, x3;

  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    X->md[(i+0)*X->inc] = y0;
    X->md[(i+1)*X->inc] = y1;
    X->md[(i+2)*X->inc] = y2;
    X->md[(i+3)*X->inc] = y3;
    Y->md[(i+0)*Y->inc] = x0;
    Y->md[(i+1)*Y->inc] = x1;
    Y->md[(i+2)*Y->inc] = x2;
    Y->md[(i+3)*Y->inc] = x3;
  }    
  if (i == N)
    return;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
  }
}

/**
 * @brief Swap vectors X and Y.
 *
 * @param[in,out] X, Y vectors
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_swap(__armas_dense_t *Y, __armas_dense_t *X, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (X->cols != 1 && X->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (Y->cols != 1 && Y->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (__armas_size(X) != __armas_size(Y)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (__armas_size(X) == 0 || __armas_size(Y) == 0) {
    return 0;
  }
  mvec_t x = {X->elems, (X->rows == 1 ? X->step : 1)};
  mvec_t y = {Y->elems, (Y->rows == 1 ? Y->step : 1)};

  __vec_swap(&y, &x, __armas_size(Y));
  return 0;
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:

