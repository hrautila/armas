
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! dot product

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_dot) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"

#if EXT_PRECISION && defined(__vec_dot_ext)
#define HAVE_EXT_PRECISION 1
extern DTYPE __vec_dot_ext(const mvec_t *X,  const mvec_t *Y, int N);
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"


static inline
DTYPE __vec_dot(const mvec_t *X,  const mvec_t *Y, int N)
{
  register int i, kx, ky;
  register DTYPE c0, c1, c2, c3, x0, x1, x2, x3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    c0 += x0*Y->md[(i+0)*Y->inc];
    c1 += x1*Y->md[(i+1)*Y->inc];
    c2 += x2*Y->md[(i+2)*Y->inc];
    c3 += x3*Y->md[(i+3)*Y->inc];
  }    
  if (i == N)
    goto update;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    c0 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 2:
    c1 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 1:
    c2 += X->md[kx] * Y->md[ky];
  }
 update:
  return c0 + c1 + c2 + c3;
}

static
DTYPE __vec_dot_recursive(const mvec_t *X, const mvec_t *Y, int n)
{
  register DTYPE c0, c1, c2, c3;
  mvec_t x0, y0;

  if (n < MIN_MVEC_SIZE)
    return __vec_dot(X, Y, n);

  if (n/2 < MIN_MVEC_SIZE) {
    c0 = __vec_dot(__subvector(&x0, X, 0),   __subvector(&y0, Y, 0),   n/2);
    c1 = __vec_dot(__subvector(&x0, X, n/2), __subvector(&y0, Y, n/2), n-n/2);
    return c0+c1;
  }

  if (n/4 < MIN_MVEC_SIZE) {
    c0 = __vec_dot(__subvector(&x0, X, 0),       __subvector(&y0, Y, 0),       n/4);
    c1 = __vec_dot(__subvector(&x0, X, n/4),     __subvector(&y0, Y, n/4),     n/2-n/4);
    c2 = __vec_dot(__subvector(&x0, X, n/2),     __subvector(&y0, Y, n/2),     n/4);
    c3 = __vec_dot(__subvector(&x0, X, n/2+n/4), __subvector(&y0, Y, n/2+n/4), n-n/2-n/4);
    return c0 + c1 + c2 + c3;
  }

  c0 = __vec_dot_recursive(__subvector(&x0, X, 0),       __subvector(&y0, Y, 0),       n/4);
  c1 = __vec_dot_recursive(__subvector(&x0, X, n/4),     __subvector(&y0, Y, n/4),     n/2-n/4);
  c2 = __vec_dot_recursive(__subvector(&x0, X, n/2),     __subvector(&y0, Y, n/2),     n/4);
  c3 = __vec_dot_recursive(__subvector(&x0, X, n/2+n/4), __subvector(&y0, Y, n/2+n/4), n-n/2-n/4);
  return c0 + c1 + c2 + c3;
}

/**
 * @brief Computes inner product of two vectors, \f$ v = x^T*y \f$
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in] x vector
 * @param[in] y vector
 * @param[in] conf configuration block
 *
 * @ingroup blas1
 */
DTYPE __armas_dot(const __armas_dense_t *x, const __armas_dense_t *y, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  // only for column or row vectors
  if (x->cols != 1 && x->rows != 1) {
    if (conf) conf->error = ARMAS_ENEED_VECTOR;
	return __ZERO;
  }
  if (y->cols != 1 && y->rows != 1) {
    if (conf) conf->error = ARMAS_ENEED_VECTOR;
	return __ZERO;
  }
  if (__armas_size(x) != __armas_size(y)) {
    if (conf) conf->error = ARMAS_ESIZE;
    return __ZERO;
  }

  const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  const mvec_t Y = {y->elems, (y->rows == 1 ? y->step : 1)};

  // extended precision 
  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    return __vec_dot_ext(&Y, &X, __armas_size(y));
  }
  
  // single precision
  if (conf->optflags & ARMAS_ONAIVE)
    return __vec_dot(&Y, &X, __armas_size(y));

  return __vec_dot_recursive(&Y, &X, __armas_size(y));
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */


// Local Variables:
// indent-tabs-mode: nil
// End:

