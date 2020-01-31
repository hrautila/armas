
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
#if defined(__gemv_recursive)
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
    int flags,
    int N)
{
    armas_x_dense_t yy, xx, aa;
    DTYPE yk;

    if (N <= 0)
        return;

    if (flags & ARMAS_LOWER) {
        for (int j = 0; j < N; j++) {
            armas_x_subvector_unsafe(&xx, X, 0, j);
            armas_x_submatrix_unsafe(&aa, A, j, 0, 1, j);
            yk = __ZERO;
            armas_x_adot_unsafe(&yk, alpha, &xx, &aa);

            armas_x_subvector_unsafe(&xx, X, j+1, N-j-1);
            armas_x_submatrix_unsafe(&aa, A, j+1, j, N-j-1, 1);
            armas_x_adot_unsafe(&yk, alpha, &xx, &aa);
            armas_x_set_at_unsafe(Y, j, beta*yk);
        }
        return;
    }

    for (j = 0; j < N; j++) {
        armas_x_subvector_unsafe(&xx, X, 0, j);
        armas_x_submatrix_unsafe(&aa, A, 0, j, j, 1);
        yk = __ZERO;
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
    int N)
{
    armas_x_dense_t x0, y0;
    armas_x_dense_t A0;

    //printf("symv_recursive: N=%d\", N);

    if (N < MIN_MVEC_SIZE) {
        symv_unb(Y, A, X, alpha, flags, N);
        return;
    }

    // 1st part  ; diagonal [0:nx/2, 0:ny/2]
    __subvector(&y0, Y, 0);
    __subvector(&x0, X, 0);
    __subblock(&A0, A, 0, 0);
    if (N/2 < MIN_MVEC_SIZE) {
        symv_unb(&y0, &A0, &x0, alpha, flags, N/2);
    } else {
        symv_recursive(&y0, &A0, &x0, alpha, flags, N/2);
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
        symv_unb(&y0, &A0, &x0, alpha, flags, N-N/2);
    } else {
        symv_recursive(&y0, &A0, &x0, alpha, flags, N-N/2);
    }

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

    symv_unb(beta, y, A, x, alpha, flags, nx);
    return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
