
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! symmetric matrix - vector multiplication

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mvmult_sym) //&& defined(__symv_recursive)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(armas_adot_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "partition.h"

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
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
    int flags)
{
    armas_dense_t x1, y1, a01;
    DTYPE x0, y0, a00;
    /*
     *    y0     a00  a01^T  x0
     *    y1     a01  A11    x1
     *
     *    y0 = y0 + alpha*a00*x0 + alpha*dot(a01, x1)
     *    y1 = y1 + alpha*x0*a01
     */
    for (int j = 0; j < A->cols; j++) {
        if (flags & ARMAS_LOWER) {
            armas_submatrix_unsafe(&a01, A, j + 1, j, A->rows - j - 1, 1);
        } else {
            armas_submatrix_unsafe(&a01, A, j, j + 1, 1, A->cols - j - 1);
        }
        armas_subvector_unsafe(&x1, X, j + 1, A->rows - j - 1);
        armas_subvector_unsafe(&y1, Y, j + 1, A->rows - j - 1);
        y0 = armas_get_at_unsafe(Y, j);
        x0 = armas_get_at_unsafe(X, j);
        a00 = armas_get_unsafe(A, j, j);
        // y0 = y0 + alpha*x0*a00, + alpha*dot(x1, x01)
        y0 += alpha * x0 * a00;
        armas_adot_unsafe(&y0, alpha, &x1, &a01);
        armas_set_at_unsafe(Y, j, y0);
        // y1 = y1 + alpha*a00*x1
        armas_axpby_unsafe(ONE, &y1, alpha * x0, &a01);
    }
    return;
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
static
void symv_recursive(
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
    int flags,
    int blas2min)
{
    armas_dense_t xT, xB, yT, yB;
    armas_dense_t ATL, ATR, ABL, ABR;
    int N = armas_size(Y);

    if (N < blas2min) {
        symv_unb(Y, alpha, A, X, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, N/2, N/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, N/2, ARMAS_PTOP);
    vec_partition_2x1(
        &yT,
        &yB, /**/ Y, N/2, ARMAS_PTOP);

    if (flags & ARMAS_UPPER) {
        symv_recursive(&yT, alpha, &ATL, &xT, flags, blas2min);
        armas_mvmult_unsafe(ONE, &yT, alpha, &ATR, &xB, 0);

        symv_recursive(&yB, alpha, &ABR, &xB, flags, blas2min);
        armas_mvmult_unsafe(ONE, &yB, alpha, &ATR, &xT, ARMAS_TRANS);
    } else {
        symv_recursive(&yT, alpha, &ATL, &xT, flags, blas2min);
        armas_mvmult_unsafe(ONE, &yT, alpha, &ABL, &xB, ARMAS_TRANS);

        symv_recursive(&yB, alpha, &ABR, &xB, flags, blas2min);
        armas_mvmult_unsafe(ONE, &yB, alpha, &ABL, &xT, 0);

    }
}

void armas_mvmult_sym_unsafe(
    DTYPE beta, armas_dense_t *y, DTYPE alpha,
    const armas_dense_t *A, const armas_dense_t *x, int flags)
{
    armas_env_t *env = armas_getenv();
    if (beta != ONE) {
        armas_scale_unsafe(y, beta);
    }
    symv_recursive(y, alpha, A, x, flags, env->blas2min);
}
/**
 * @brief Symmetric matrix-vector multiply.
 *
 * Computes
 *    \f$ y = beta \times y + alpha \times A x \f$
 *
 * Matrix A elements are stored on lower (upper) triangular part of the matrix
 * if flag bit *ARMAS_LOWER* (*ARMAS_UPPER*) is set.
 *
 *  @param[in]      beta scalar
 *  @param[in,out]  y   target and source vector
 *  @param[in]      alpha scalar
 *  @param[in]      A   symmetrix lower (upper) matrix
 *  @param[in]      x   source operand vector
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blas
 */
int armas_mvmult_sym(
    DTYPE beta,
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *x,
    int flags,
    armas_conf_t *conf)
{
    int ok;
    int nx = armas_size(x);
    int ny = armas_size(y);

    if (armas_size(A) == 0 || nx == 0 || ny == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (!armas_isvector(y)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }

    ok = A->cols == A->rows && nx == ny && nx == A->cols;
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    armas_env_t *env = armas_getenv();
    if (beta != ONE) {
        armas_scale_unsafe(y, beta);
    }
    symv_recursive(y, alpha, A, x, flags, env->blas2min);
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
