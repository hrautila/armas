
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvupdate_trm) && defined(__update_trmv_recursive) && \
  defined(__update_trmv_unb)
#define __ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(__update_ger_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

// update one column of A
static inline
void __mult_mv1axpy(DTYPE *Ac, DTYPE *Xc, int incX, DTYPE alpha, int N)
{
  register int i;
  for (i = 0; i < N-3; i += 4) {
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    Ac[(i+1)] += Xc[(i+1)*incX]*alpha;
    Ac[(i+2)] += Xc[(i+2)*incX]*alpha;
    Ac[(i+3)] += Xc[(i+3)*incX]*alpha;
  }
  if (i == N)
    return;
  switch (N-i) {
  case 3:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    i++;
  case 2:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    i++;
  case 1:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
  }
}

/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 * (M is rows, N is columns.)
 */
void __update_trmv_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int flags, int N, int M)
{
  register int j;
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
  case ARMAS_UPPER:
    for (j = 0; j < N; j++) {
      __mult_mv1axpy(&A->md[j*A->step], &X->md[0], X->inc, alpha*Y->md[j*Y->inc], min(M, j+1));
    }
    break;
  case ARMAS_LOWER:
  default:
    for (j = 0; j < N; j++) {
      __mult_mv1axpy(&A->md[j+j*A->step], &X->md[j*X->inc], X->inc, alpha*Y->md[j*Y->inc], M-j);
    }
  }
}

void __update_trmv_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N, int M)
{
  mvec_t x0, y0;
  mdata_t A0;
  int nd = min(M, N);

  if (M < MIN_MVEC_SIZE || N < MIN_MVEC_SIZE) {
    __update_trmv_unb(A, X, Y, alpha, flags, N, M);
    return;
  }

  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (nd/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, nd/2, nd/2);
  } else {
    __update_trmv_recursive(&A0, &x0, &y0, alpha, flags, nd/2, nd/2);
  }

  if (flags & ARMAS_UPPER) {
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, nd/2);
    __subblock(&A0, A, 0, nd/2);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, N-nd/2, nd/2);
  } else {
    __subvector(&x0, X, nd/2);
    __subblock(&A0, A, nd/2, 0);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, nd/2, M-nd/2);
  }

  __subvector(&y0, Y, nd/2);
  __subvector(&x0, X, nd/2);
  __subblock(&A0, A, nd/2, nd/2);
  if (N-nd/2 < MIN_MVEC_SIZE || M-nd/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N-nd/2, M-nd/2);
  } else {
  __update_trmv_recursive(&A0, &x0, &y0, alpha, flags, N-nd/2, M-nd/2);
  }
}


/**
 * @brief General triangular/trapezoidial matrix rank update.
 *
 * Computes
 *
 * > A := A + alpha*X*Y.T.
 *
 * where A is upper (lower) triangular or trapezoidial matrix as defined with
 * flag bits ARMAS_UPPER (ARMAS_LOWER).
 *
 * @param[in,out]  A target matrix
 * @param[in]      X source vector
 * @param[in]      Y source vector
 * @param[in]      alpha scalar multiplier
 * @param[in]      flags flag bits
 * @param[in]      conf  configuration block
 *
 * @ingroup blas2
 */
int __armas_mvupdate_trm(__armas_dense_t *A,
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
  
  if (A->cols != ny || A->rows != nx) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  switch (conf->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
  case ARMAS_RECURSIVE:
    __update_trmv_recursive(&A0, &x, &y, alpha, flags, ny, nx);
    break;

  case ARMAS_SNAIVE:
  default:
    __update_trmv_unb(&A0, &x, &y, alpha, flags, ny, nx);
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
