
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! matrix-vector multiplication

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvmult) && defined(__gemv_recursive)
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
#include "linalg.h"
#include "mvec_nosimd.h"

#ifdef MIN_MVEC_SIZE
#undef MIN_MVEC_SIZE
#define  MIN_MVEC_SIZE 2048
#endif

#if EXT_PRECISION && defined(__gemv_ext_unb)
#define HAVE_EXT_PRECISION 1
extern int __gemv_ext_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                          DTYPE alpha, DTYPE beta, int flags, int nX, int nY);
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"
//! \endcond

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
static
void __gemv_unb_abs(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                    DTYPE alpha, int flags, int S, int L, int R, int E)
{
  int i, j;
  register DTYPE *y;
  register const DTYPE *x;
  register const DTYPE *a0, *a1, *a2, *a3;

  // L - S is columns in A, elements in X
  // E - R is rows in A, elements in Y 
  if (L - S <= 0 || E - R <= 0) {
    return;
  }

  if ((flags & ARMAS_TRANS) != 0) {

    x = &X->md[S*X->inc];
    for (i = R; i < E-3; i += 4) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[S+(i+0)*A->step];
      a1 = &A->md[S+(i+1)*A->step];
      a2 = &A->md[S+(i+2)*A->step];
      a3 = &A->md[S+(i+3)*A->step];
      __vmult4dot_abs(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, L-S);
    }
    if (i == E)
      return;

    switch (E-i) {
    case 3:
    case 2:
      y = &Y->md[i*Y->inc];
      a0 = &A->md[S+(i+0)*A->step];
      a1 = &A->md[S+(i+1)*A->step];
      __vmult2dot_abs(y, Y->inc, a0, a1, x, X->inc, alpha, L-S);
      i += 2;
    }
    if (i < E) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[S+(i+0)*A->step];
      __vmult1dot_abs(y, Y->inc, a0, x, X->inc, alpha, L-S);
    }
    return;
  }

  // Non-Transposed A here

  y = &Y->md[R*Y->inc];
  for (j = S; j < L-3; j += 4) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    a2 = &A->md[R+(j+2)*A->step];
    a3 = &A->md[R+(j+3)*A->step];
    __vmult4axpy_abs(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, E-R);
  }

  if (j == L)
    return;

  switch (L-j) {
  case 3:
  case 2:
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    __vmult2axpy_abs(y, Y->inc, a0, a1, x, X->inc, alpha, E-R);
    j += 2;
  }
  if (j < L) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    __vmult1axpy_abs(y, Y->inc, a0, x, X->inc, alpha, E-R);
  }
}

static
void __gemv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                DTYPE alpha, int flags, int S, int L, int R, int E)
{
  int i, j;
  register DTYPE *y;
  register const DTYPE *x;
  register const DTYPE *a0, *a1, *a2, *a3;

  // L - S is columns in A, elements in X
  // E - R is rows in A, elements in Y 
  if (L - S <= 0 || E - R <= 0) {
    return;
  }

  if (flags & ARMAS_ABS) {
    __gemv_unb_abs(Y, A, X, alpha, flags, S, L, R, E);
    return;
  }
  
  if ((flags & ARMAS_TRANSA) || (flags & ARMAS_TRANS)) {

    x = &X->md[R*X->inc];
    for (i = S; i < L-3; i += 4) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[R+(i+0)*A->step];
      a1 = &A->md[R+(i+1)*A->step];
      a2 = &A->md[R+(i+2)*A->step];
      a3 = &A->md[R+(i+3)*A->step];
      __vmult4dot(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, E-R);
    }
    if (i == L)
      return;

    switch (L-i) {
    case 3:
    case 2:
      y = &Y->md[i*Y->inc];
      a0 = &A->md[R+(i+0)*A->step];
      a1 = &A->md[R+(i+1)*A->step];
      __vmult2dot(y, Y->inc, a0, a1, x, X->inc, alpha, E-R);
      i += 2;
    }
    if (i < L) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[R+(i+0)*A->step];
      __vmult1dot(y, Y->inc, a0, x, X->inc, alpha, E-R);
    }
    return;
  }

  // Non-Transposed A here

  y = &Y->md[R*Y->inc];
  for (j = S; j < L-3; j += 4) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    a2 = &A->md[R+(j+2)*A->step];
    a3 = &A->md[R+(j+3)*A->step];
    __vmult4axpy(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, E-R);
  }

  if (j == L)
    return;

  switch (L-j) {
  case 3:
  case 2:
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    __vmult2axpy(y, Y->inc, a0, a1, x, X->inc, alpha, E-R);
    j += 2;
  }
  if (j < L) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    __vmult1axpy(y, Y->inc, a0, x, X->inc, alpha, E-R);
  }
}


void __gemv(mvec_t *Y, const mdata_t *A, const mvec_t *X,
            DTYPE alpha, int flags, int M, int N)
{
  if (M <= 0 || N <= 0)
    return;
  
  __gemv_unb(Y, A, X, alpha, flags, 0, N, 0, M);
}



void __gemv_recursive(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                      DTYPE alpha, DTYPE beta, int flags,
                      int S, int L, int R, int E)
{
  mvec_t x0, y0;
  mdata_t A0;
  int ny = E - R;
  int nx = L - S;

  //printf("__gemv_recursive: S=%d, L=%d, R=%d, E=%d, nx=%d, ny=%d\n", S, L, R, E, nx, ny);

  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, S, L, R, E);
    return;
  }

  // 1st block
  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx/2, 0, ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx/2, 0, ny/2);
  }

  // 2nd block
  __subvector(&y0, Y, ny/2);
  if (flags & ARMAS_TRANS) {
    __subblock(&A0, A, 0, ny/2);
  } else {
    __subblock(&A0, A, ny/2, 0);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx/2, 0, ny-ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx/2, 0, ny-ny/2);
  }

  // 3rd block
  __subvector(&x0, X, nx/2);
  __subvector(&y0, Y, 0);
  if (flags & ARMAS_TRANS) {
    __subblock(&A0, A, nx/2, 0);
  } else {
    __subblock(&A0, A, 0, nx/2);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx-nx/2, 0, ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx-nx/2, 0, ny/2);
  }

  // 4th block
  __subvector(&y0, Y, ny/2);
  if (flags & ARMAS_TRANS) {
    __subblock(&A0, A, nx/2, ny/2);
  } else {
    __subblock(&A0, A, ny/2, nx/2);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx-nx/2, 0, ny-ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx-nx/2, 0, ny-ny/2);
  }
}


/**
 * @brief General matrix-vector multiply.
 *
 * Computes
 *   - \f$ Y = alpha \times A X + beta \times Y \f$
 *   - \f$ Y = alpha \times A^T X + beta \times Y  \f$   if *ARMAS_TRANS* set
 *   - \f$ Y = alpha \times |A| |X|  + beta \times Y \f$ if *ARMAS_ABS* set
 *   - \f$ Y = alpha \times |A^T| |X| + beta \times Y \f$ if *ARMAS_ABS* and *ARMAS_TRANS* set
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      A   source operand matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      alpha, beta scalars
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvmult(armas_x_dense_t *Y, const armas_x_dense_t *A, const armas_x_dense_t *X,
                   DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int ok;
  mvec_t x, y;
  mdata_t A0;
  int nx = armas_x_size(X);
  int ny = armas_x_size(Y);
  
  if (!conf)
    conf = armas_conf_default();

  if (armas_x_size(A) == 0 || armas_x_size(X) == 0 || armas_x_size(Y) == 0)
    return 0;
  
  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (Y->rows != 1 && Y->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  // check consistency
  switch (flags & ARMAS_TRANS) {
  case ARMAS_TRANS:
    ok = A->cols == ny && A->rows == nx;
    break;
  default:
    ok = A->rows == ny && A->cols == nx;
    break;
  }
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    __gemv_ext_unb(&y, &A0, &x, alpha, beta, flags, nx, ny);
    return 0;
  }

  // single precision here
  if (beta != 1.0) {
    armas_x_scale(Y, beta, conf);
  }
  if (conf->optflags & ARMAS_ORECURSIVE) {
    __gemv_recursive(&y, &A0, &x, alpha, beta, flags, 0, nx, 0, ny);
  } else {
    __gemv_unb(&y, &A0, &x, alpha, flags, 0, A->cols, 0, A->rows);
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
