
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/** @defgroup blas2 BLAS level 2 functions.
 *
 */
#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvmult_sym) //&& defined(__symv_recursive)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(__gemv_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

#if EXT_PRECISION && defined(__symv_ext_unb)
#define HAVE_EXT_PRECISION 1
extern int __symv_ext_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                          DTYPE alpha, DTYPE beta, int flags, int N);
#endif

#include "cond.h"


/*
 * Objective: read matrix A in memory order, along columns.
 *
 *  y0    a00 |  0   0     x0     y0 = a00*x0 + a10*x1 + a20*x2
 *  --    --------------   --
 *  y1    a10 | a11  0     x1     y1 = a10*x0 + a11*x1 + a21*x2
 *  y2    a20 | a21  a22   x2     y2 = a20*x0 + a21*x1 + a22*x2
 *
 *  y1 += (a11) * x1  
 *  y2    (a21)
 *
 *  y1 += a21.T*x2
 *
 * UPPER:
 *  y0    a00 | a01 a02   x0     y0 = a00*x0 + a01*x1 + a02*x2
 *  --    --------------   --
 *  y1     0  | a11 a12   x1     y1 = a01*x0 + a11*x1 + a12*x2
 *  y2     0  |  0  a22   x2     y2 = a02*x0 + a12*x1 + a22*x2
 *
 *  (y0) += (a01) * x1
 *  (y1)    (a11)
 */
static
void __symv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                DTYPE alpha, int flags, int N)
{
  int j;
  mvec_t yy, xx, aa;
  
  if ( N <= 0 )
    return;

  if (flags & ARMAS_LOWER) {
    for (j = 0; j < N; j++) {
      __subvector(&yy, Y, j);
      __subvector(&xx, X, j);
      __colvec(&aa, A, j, j);
      __vmult1axpy(yy.md, yy.inc, aa.md, xx.md, xx.inc, alpha, N-j);
      __vmult1dot(yy.md, yy.inc, &aa.md[1], &xx.md[xx.inc], xx.inc, alpha, N-j-1);
    }
    return;
  }

  // Upper here;
  //  1. update elements 0:j with current column and x[j]
  //  2. update current element y[j] with product of a[0:j-1]*x[0:j-1]
  for (j = 0; j < N; j++) {
    __subvector(&xx, X, j);
    __colvec(&aa, A, 0, j);
    __vmult1axpy(Y->md, Y->inc, aa.md, xx.md, xx.inc, alpha, j+1);

    __subvector(&yy, Y, j);
    __vmult1dot(yy.md, yy.inc, aa.md, X->md, X->inc, alpha, j);
  }
}

/*
 * LOWER:
 *  ( y0 ) = ( A00  A10.T) * ( x0 )
 *  ( y1 )   ( A10  A11  )   ( x1 )
 *
 *  y0 = A00*x0 + A10.T*x1  = symv(A00, x0) + gemv(A10, x1, T)
 *  y1 = A10*x0 + A11*x1    = symv(A11, x1) + gemv(A10, x0, N)
 *
 * UPPER:
 *  ( y0 ) = ( A00  A01 ) * ( x0 )
 *  ( y1 )   (  0   A11 )   ( x1 )
 *
 *  y0 = A00*x0   + A01*x1  = symv(A00, x0) + gemv(A01, x1, N)
 *  y1 = A01.T*x0 + A11*x1  = symv(A11, x1) + gemv(A01, x0, T)
 *
 */
#if 0
void __symv_recursive(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                      DTYPE alpha, int flags, int N)
{
  mvec_t x0, y0;
  mdata_t A0;

  //printf("__symv_recursive: N=%d\", N);

  if (N < MIN_MVEC_SIZE) {
    __symv_unb(Y, A, X, alpha, flags, N);
    return;
  }

  // 1st part  ; diagonal [0:nx/2, 0:ny/2]
  __subvector(&y0, Y, 0);
  __subvector(&x0, X, 0);
  __subblock(&A0, A, 0, 0);
  if (N/2 < MIN_MVEC_SIZE) {
    __symv_unb(&y0, &A0, &x0, alpha, flags, N/2);
  } else {
    __symv_recursive(&y0, &A0, &x0, alpha, flags, N/2);
  }

  if (flags & ARMAS_LOWER) {
    // update y[0:N/2] with rectangular part
    __subblock(&A0, A, N/2, 0);
    __subvector(&x0, X, N/2);
    __gemv_recursive(&y0, &A0, &x0, alpha, 1.0, ARMAS_TRANS, 0, N-N/2, 0, N/2);

    // update y[N/2:N] with rectangular part
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, N/2);
    __gemv_recursive(&y0, &A0, &x0, alpha, 1.0, ARMAS_NONE, 0, N/2, 0, N-N/2);
  } else {
    // update y[0:N/2] with rectangular part
    __subblock(&A0, A, 0, N/2);
    __subvector(&x0, X, N/2);
    __gemv_recursive(&y0, &A0, &x0, alpha, 1.0, ARMAS_NONE, 0, N-N/2, 0, N/2);

    // update y[N/2:N] with rectangular part
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, N/2);
    __gemv_recursive(&y0, &A0, &x0, alpha, 1.0, ARMAS_TRANS, 0, N/2, 0, N-N/2);
  }

  // 2nd part ; diagonal [N/2:N, N/2:N]
  __subvector(&x0, X, N/2);
  __subblock(&A0, A, N/2, N/2);
  if (N/2 < MIN_MVEC_SIZE) {
    __symv_unb(&y0, &A0, &x0, alpha, flags, N-N/2);
  } else {
    __symv_recursive(&y0, &A0, &x0, alpha, flags, N-N/2);
  }

}
#endif


/**
 * @brief Symmetic matrix-vector multiply.
 *
 * Computes
 *
 * > Y := alpha*A*X + beta*Y
 *
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      A   symmetrix lower (upper) matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      alpha, beta scalars
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 * @ingroup blas2
 */
int __armas_mvmult_sym(__armas_dense_t *Y, const __armas_dense_t *A, const __armas_dense_t *X,
                       DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int ok;
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

  ok = A->cols == A->rows && nx == ny && nx == A->cols;
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  // if extended precision enabled and requested
  IF_EXTPREC_RVAL(conf->optflags&ARMAS_OEXTPREC, 0, 
                  __symv_ext_unb(&y, &A0, &x, alpha, beta, flags, nx));          

  // normal precision here
  if (beta != 1.0) {
    __armas_scale(Y, beta, conf);
  }
  __symv_unb(&y, &A0, &x, alpha, flags, nx);
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
