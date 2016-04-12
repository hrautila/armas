
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Copy vector

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_copy) 
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

static
void __vec_copy(mvec_t *X,  const mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double f0, f1, f2, f3;

  // gcc compiles loop body to use different target XMM registers
  for (i = 0; i < N-3; i += 4) {
    f0 = Y->md[(i+0)*Y->inc];
    f1 = Y->md[(i+1)*Y->inc];
    f2 = Y->md[(i+2)*Y->inc];
    f3 = Y->md[(i+3)*Y->inc];
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // calculate indexes only once
  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    X->md[kx] = Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 2:
    X->md[kx] = Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 1:
    X->md[kx] = Y->md[ky];
  }
}

/**
 * @brief Copy vector, \f$ Y := X \f$
 *
 * @param[out] Y target vector
 * @param[in]  X source vector
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_copy(__armas_dense_t *Y, const __armas_dense_t *X, armas_conf_t *conf)
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
  if (__armas_size(X) < __armas_size(Y)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (__armas_size(X) == 0 || __armas_size(Y) == 0) {
    return 0;
  }
  const mvec_t x = {X->elems, (X->rows == 1 ? X->step : 1)};
  mvec_t y = {Y->elems, (Y->rows == 1 ? Y->step : 1)};

  __vec_copy(&y, &x, __armas_size(Y));
  return 0;
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:

