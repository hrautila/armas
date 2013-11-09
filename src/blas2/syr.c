
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvupdate_sym)
#define __ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(__update_trmv_unb) && defined(__update_trmv_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"


int __armas_mvupdate_sym(__armas_dense_t *A,
                         const __armas_dense_t *X,
                         DTYPE alpha, int flags, armas_conf_t *conf)
{
  mvec_t x;
  mdata_t A0;
  int nx = __armas_size(X);

  if (!conf)
    conf = armas_conf_default();

  if (A->cols == 0 || A->rows == 0)
    return 0;
  
  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (A->cols != nx || A->rows != nx) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  switch (conf->optflags) {
  case ARMAS_SNAIVE:
    __update_trmv_unb(&A0, &x, &x, alpha, flags, nx, nx);
    break;

  case ARMAS_RECURSIVE:
  default:
    __update_trmv_recursive(&A0, &x, &x, alpha, flags, nx, nx);
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
