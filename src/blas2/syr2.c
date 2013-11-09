
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mv2update_sym)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(__update_trmv_unb) && defined(__update_ger_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"


static
void __update_syr2_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N)
{
  mvec_t x0, y0;
  mdata_t A0;

  if (N < MIN_MVEC_SIZE) {
    __update_trmv_unb(A, X, Y, alpha, flags, N, N);
    __update_trmv_unb(A, Y, X, alpha, flags, N, N);
    return;
  }

  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (N/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N/2, N/2);
    __update_trmv_unb(&A0, &y0, &x0, alpha, flags, N/2, N/2);
  } else {
    __update_syr2_recursive(&A0, &x0, &y0, alpha, flags, N/2);
  }

  if (flags & ARMAS_UPPER) {
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, N/2);
    __subblock(&A0, A, 0, N/2);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, N-N/2, N/2);
    __subvector(&x0, X, N/2);
    __subvector(&y0, Y, 0);
    __update_ger_recursive(&A0, &y0, &x0, alpha, flags, N-N/2, N/2);
  } else {
    __subvector(&y0, Y, 0);
    __subvector(&x0, X, N/2);
    __subblock(&A0, A, N/2, 0);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, N/2, N-N/2);
    __subvector(&y0, Y, N/2);
    __subvector(&x0, X, 0);
    __update_ger_recursive(&A0, &y0, &x0, alpha, flags, N/2, N-N/2);
  }

  __subvector(&y0, Y, N/2);
  __subvector(&x0, X, N/2);
  __subblock(&A0, A, N/2, N/2);
  if (N-N/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N-N/2, N-N/2);
    __update_trmv_unb(&A0, &y0, &x0, alpha, flags, N-N/2, N-N/2);
  } else {
    __update_syr2_recursive(&A0, &x0, &y0, alpha, flags, N-N/2);
  }
}


int __armas_mv2update_sym(__armas_dense_t *A,
                          const __armas_dense_t *X,  const __armas_dense_t *Y,  
                          DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ok;
  mvec_t x, y;
  mdata_t A0;
  int nx = __armas_size(X);
  int ny = __armas_size(Y);

  if (!conf)
    conf = armas_conf_default();

  if (A->cols == 0 || A->rows == 0)
    return 0;
  
  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (Y->rows != 1 && Y->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (A->cols != A->rows || ny != nx) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  switch (conf->optflags) {
  case ARMAS_SNAIVE:
    __update_trmv_unb(&A0, &x, &y, alpha, flags, nx, nx);
    __update_trmv_unb(&A0, &y, &x, alpha, flags, nx, nx);
    break;

  case ARMAS_RECURSIVE:
  default:
    __update_syr2_recursive(&A0, &x, &y, alpha, flags, nx);
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRED */

// Local Variables:
// indent-tabs-mode: nil
// End:
