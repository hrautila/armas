
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! symmetric matrix - vector multiplication

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvmult_sym) //&& defined(__symv_recursive)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(armas_x_adot_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//#include "nosimd/mvec.h"
//#include "cond.h"
//! \endcond


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
void symv_unb(
    DTYPE beta,
    armas_x_dense_t *Y,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    int flags)
{
    armas_x_dense_t xx, aa;
    DTYPE yk;
    int N = armas_x_size(Y);

    if (flags & ARMAS_LOWER) {
        for (int j = 0; j < N; j++) {
            armas_x_subvector_unsafe(&xx, X, 0, j+1);
            armas_x_submatrix_unsafe(&aa, A, j, 0, 1, j+1);
            yk = ZERO;
            armas_x_adot_unsafe(&yk, alpha, &xx, &aa);

            armas_x_subvector_unsafe(&xx, X, j+1, N-j-1);
            armas_x_submatrix_unsafe(&aa, A, j+1, j, N-j-1, 1);
            armas_x_adot_unsafe(&yk, alpha, &xx, &aa);
            armas_x_set_at_unsafe(Y, j, beta*yk);
        }
        return;
    }

    for (int j = 0; j < N; j++) {
        armas_x_subvector_unsafe(&xx, X, 0, j+1);
        armas_x_submatrix_unsafe(&aa, A, 0, j, j+1, 1);
        yk = ZERO;
        armas_x_adot_unsafe(&yk, alpha, &xx, &aa);

        armas_x_subvector_unsafe(&xx, X, j+1, N-j-1);
        armas_x_submatrix_unsafe(&aa, A, j, j+1, 1, N-j-1);
        armas_x_adot_unsafe(&yk, alpha, &xx, &aa);
        armas_x_set_at_unsafe(Y, j, beta*yk);
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
void symv_recursive(
    DTYPE beta,
    armas_x_dense_t *Y,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    DTYPE alpha,
    int flags,
    int blas2min)
{
    armas_x_dense_t x0, y0, xT, xB, yT, yB;
    armas_x_dense_t ATL, ATR, ABL, ABR;
    int N = armas_x_size(Y);

    if (N < blas2min) {
        symv_unb(beta, Y, alpha, A, X, flags);
        return;
    }

    mat_partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, N/2, N/2, ARMAS_PTOPLEFT);
    mat_partitiion2x1(
        &xT,
        &xB, /**/ X, N/2, ARMAS_PTOP);
    mat_partitiion2x1(
        &yT,
        &yB, /**/ Y, N/2, ARMAS_PTOP);

    symv_recursive(beta, &yT, alpha, &ATL, &xT, flags, blas2min);
    armas_x_mvmult_recursive(ONE, &yT, alpha, &ATR, &xB, 0, blas2min);
}
#endif


/**
 * @brief Symmetric matrix-vector multiply.
 *
 * Computes
 *    - \f$ Y = alpha \times A X + beta \times Y \f$
 *
 * Matrix A elements are stored on lower (upper) triangular part of the matrix
 * if flag bit *ARMAS_LOWER* (*ARMAS_UPPER*) is set.
 *
 *  @param[in]      beta scalar
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      alpha scalar
 *  @param[in]      A   symmetrix lower (upper) matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvmult_sym(
    DTYPE beta,
    armas_x_dense_t *y,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *x,
    int flags,
    armas_conf_t *conf)
{
    int ok;
    int nx = armas_x_size(x);
    int ny = armas_x_size(y);

    if (armas_x_size(A) == 0 || nx == 0 || ny == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    if (!armas_x_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (!armas_x_isvector(y)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }

    ok = A->cols == A->rows && nx == ny && nx == A->cols;
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    symv_unb(beta, y, alpha, A, x, flags, nx);
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
