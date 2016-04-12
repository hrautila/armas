
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! vector-vector summation

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_axpy) 
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

#if EXT_PRECISION && defined(__vec_axpy_ext) && defined(__vec_axpby_ext)
#define HAVE_EXT_PRECISION 1
int __vec_axpy_ext(mvec_t *Y,  const mvec_t *X, DTYPE alpha, int N);
int __vec_axpby_ext(mvec_t *Y,  const mvec_t *X, DTYPE alpha, DTYPE beta, int N);
#else
#define HAVE_EXT_PRECISION 0
#endif

//#include "cond.h"

static inline
void __vec_axpy(mvec_t *Y,  const mvec_t *X, DTYPE alpha, int N)
{
  register int i, kx, ky;
  register DTYPE y0, y1, y2, y3, x0, x1, x2, x3;

  // gcc uses different XMM target registers for yN, xN; 
  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    y0 += alpha*x0;
    y1 += alpha*x1;
    y2 += alpha*x2;
    y3 += alpha*x3;
    Y->md[(i+0)*Y->inc] = y0;
    Y->md[(i+1)*Y->inc] = y1;
    Y->md[(i+2)*Y->inc] = y2;
    Y->md[(i+3)*Y->inc] = y3;
  }    
  if (i == N)
	return;

  kx = i*X->inc; ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
  }
}

static inline
void __vec_axpby(mvec_t *Y,  const mvec_t *X, DTYPE alpha, DTYPE beta, int N)
{
  register int i, kx, ky;
  register DTYPE y0, y1, y2, y3, x0, x1, x2, x3;

  // gcc uses different XMM target registers for yN, xN; 
  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc] * beta;
    y1 = Y->md[(i+1)*Y->inc] * beta;
    y2 = Y->md[(i+2)*Y->inc] * beta;
    y3 = Y->md[(i+3)*Y->inc] * beta;
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    y0 += alpha*x0;
    y1 += alpha*x1;
    y2 += alpha*x2;
    y3 += alpha*x3;
    Y->md[(i+0)*Y->inc] = y0;
    Y->md[(i+1)*Y->inc] = y1;
    Y->md[(i+2)*Y->inc] = y2;
    Y->md[(i+3)*Y->inc] = y3;
  }    
  if (i == N)
	return;

  kx = i*X->inc; ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky] * beta;
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky] * beta;
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky] * beta;
    Y->md[ky] = y0 + alpha*X->md[kx];
  }
}

/**
 * @brief Compute \f$ y = y + alpha*x \f$
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out] y target and source vector
 * @param[in]     x source vector
 * @param[in]     alpha scalar multiplier
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_axpy(__armas_dense_t *y, const __armas_dense_t *x, DTYPE alpha, armas_conf_t *conf)
{
  // only for column or row vectors
  if (!conf)
    conf = armas_conf_default();

  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (y->cols != 1 && y->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (__armas_size(x) != __armas_size(y)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  mvec_t Y       = {y->elems, (y->rows == 1 ? y->step : 1)};

  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    __vec_axpy_ext(&Y, &X, alpha, __armas_size(y));
    return 0;
  }
  
  __vec_axpy(&Y, &X, alpha, __armas_size(y));
  return 0;
}


/**
 * @brief Compute \f$ y = beta*y + alpha*x \f$
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out] y target and source vector
 * @param[in]     x source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     beta scalar multiplier
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup blas1
 */
int __armas_axpby(__armas_dense_t *y, const __armas_dense_t *x, DTYPE alpha, DTYPE beta, armas_conf_t *conf)
{
  // only for column or row vectors
  if (!conf)
    conf = armas_conf_default();

  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (y->cols != 1 && y->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (__armas_size(x) != __armas_size(y)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  mvec_t Y       = {y->elems, (y->rows == 1 ? y->step : 1)};


  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    __vec_axpby_ext(&Y, &X, alpha, beta, __armas_size(y));
    return 0;
  }
  
  __vec_axpby(&Y, &X, alpha, beta, __armas_size(y));
  return 0;
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:

