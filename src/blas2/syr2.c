
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvupdate2_sym)
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

#if EXT_PRECISION && defined(__update2_symv_ext_unb)
#define HAVE_EXT_PRECISION 1
extern int __update2_symv_ext_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                                  DTYPE alpha, int flags, int N);
#endif

#include "cond.h"

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


/**
 * @brief Symmetric matrix rank-2 update.
 *
 * Computes 
 *
 * > A := A + alpha*X*Y.T + alpha*Y*X.T
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag ARMAS_LOWER (ARMAR_UPPER) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * @param[in,out]  A target matrix
 * @param[in]      X, Y source vector
 * @param[in]      alpha scalar multiplier
 * @param[in]      flags flag bits 
 * @param[in]      conf configuration block
 *
 * @ingroup blas2
 */
int __armas_mvupdate2_sym(__armas_dense_t *A,
                          const __armas_dense_t *X,  const __armas_dense_t *Y,  
                          DTYPE alpha, int flags, armas_conf_t *conf)
{
  mvec_t x, y;
  mdata_t A0;
  int nx = __armas_size(X);
  int ny = __armas_size(Y);

  if (__armas_size(A) == 0 || __armas_size(X) == 0 || __armas_size(Y) == 0)
    return 0;
  
  if (!conf)
    conf = armas_conf_default();

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

  // if extended precision enable and requested
  IF_EXTPREC_RVAL(conf->optflags&ARMAS_OEXTPREC, 0, 
                  __update2_symv_ext_unb(&A0, &x, &y, alpha, flags, nx));

  // default precision here
  switch (conf->optflags) {
  case ARMAS_RECURSIVE:
    __update_syr2_recursive(&A0, &x, &y, alpha, flags, nx);
    break;

  case ARMAS_SNAIVE:
  default:
    __update_trmv_unb(&A0, &x, &y, alpha, flags, nx, nx);
    __update_trmv_unb(&A0, &y, &x, alpha, flags, nx, nx);
    break;

  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRED */

// Local Variables:
// indent-tabs-mode: nil
// End:
