
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mnorm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_amax) && defined(__armas_asum) && defined(__armas_nrm2)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


#include "internal.h"
#include "matrix.h"


static
ABSTYPE __armas_matrix_norm_one(const __armas_dense_t *x, armas_conf_t *conf)
{
  int k;
  __armas_dense_t v;
  ABSTYPE cmax, amax = __ABSZERO;

  for (k = 0; k < x->cols; k++) {
    __armas_submatrix(&v, x, 0, k, x->rows, 1);
    cmax = __armas_amax(&v, conf);
    if (cmax > amax) {
      amax = cmax;
    }
  }
  return amax;
}

static
ABSTYPE __armas_matrix_norm_inf(const __armas_dense_t *x, armas_conf_t *conf)
{
  int k;
  __armas_dense_t v;
  ABSTYPE cmax, amax = __ABSZERO;

  for (k = 0; k < x->rows; k++) {
    __armas_submatrix(&v, x, k, 0, 1, x->cols);
    cmax = __armas_amax(&v, conf);
    if (cmax > amax) {
      amax = cmax;
    }
  }
  return amax;
}


ABSTYPE __armas_mnorm(const __armas_dense_t *x, int which, armas_conf_t *conf)
{
  ABSTYPE normval;

  if (!conf)
    conf = armas_conf_default();

  if (! x || __armas_size(x) == 0)
    return __ABSZERO;

  int is_vector = x->rows == 1 || x->cols == 1;
  switch (which) {
  case ARMAS_NORM_ONE:
    if (is_vector) {
      normval = __armas_asum(x, conf);
    } else {
      normval = __armas_matrix_norm_one(x, conf);
    }
    break;
  case ARMAS_NORM_TWO:
    if (is_vector) {
      normval = __armas_nrm2(x, conf);
    } else {
      conf->error = ARMAS_EIMP;
      normval = __ABSZERO;
    }
    break;
  case ARMAS_NORM_INF:
    if (is_vector) {
      normval = __armas_amax(x, conf);
    } else {
      normval = __armas_matrix_norm_inf(x, conf);
    }
    break;
  default:
    conf->error = ARMAS_EINVAL;
    break;
  }
  return normval;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

