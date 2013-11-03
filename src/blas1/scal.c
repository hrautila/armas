
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_scale) && defined(__armas_invscale) && defined(__armas_add)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"

static inline
void __vec_scal(mvec_t *X,  const DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] * alpha;
    f1 =  X->md[(i+1)*X->inc] * alpha;
    f2 =  X->md[(i+2)*X->inc] * alpha;
    f3 =  X->md[(i+3)*X->inc] * alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] *= alpha;
    k += X->inc;
  case 2:
    x0[k] *= alpha;
    k += X->inc;
  case 1:
    x0[k] *= alpha;
  }
}

static inline
void __vec_invscal(mvec_t *X,  const DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] / alpha;
    f1 =  X->md[(i+1)*X->inc] / alpha;
    f2 =  X->md[(i+2)*X->inc] / alpha;
    f3 =  X->md[(i+3)*X->inc] / alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] /= alpha;
    k += X->inc;
  case 2:
    x0[k] /= alpha;
    k += X->inc;
  case 1:
    x0[k] /= alpha;
  }
}

static inline
void __vec_add(mvec_t *X,  const DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] + alpha;
    f1 =  X->md[(i+1)*X->inc] + alpha;
    f2 =  X->md[(i+2)*X->inc] + alpha;
    f3 =  X->md[(i+3)*X->inc] + alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] += alpha;
    k += X->inc;
  case 2:
    x0[k] += alpha;
    k += X->inc;
  case 1:
    x0[k] += alpha;
  }
}


int __armas_scale(const __armas_dense_t *x, const DTYPE alpha, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};

  __vec_scal(&X, alpha, __armas_size(x));
  return 0;
}

int __armas_invscale(const __armas_dense_t *x, const DTYPE alpha, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};

  __vec_invscal(&X, alpha, __armas_size(x));
  return 0;
}

int __armas_add(const __armas_dense_t *x, const DTYPE alpha, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};

  __vec_add(&X, alpha, __armas_size(x));
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

