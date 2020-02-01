
// Copyright (c) Harri Rautila, 2012-2020

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
#if defined(armas_x_mvmult) && defined(armas_x_mvmult_unsafe)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "nosimd/mvec.h"
//! \endcond

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
static
void gemv_unb_abs(
    armas_x_dense_t *Y,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    DTYPE alpha,
    int flags)
{
    int i, j, xinc, yinc;
    register DTYPE *y;
    register const DTYPE *x;
    register const DTYPE *a0, *a1, *a2, *a3;

    xinc = X->rows == 1 ? X->step : 1;
    yinc = Y->rows == 1 ? Y->step : 1;

    if ((flags & ARMAS_TRANS) != 0) {

        x = &X->elems[0*xinc];
        for (i = 0; i < A->cols-3; i += 4) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            a2 = &A->elems[(i+2)*A->step];
            a3 = &A->elems[(i+3)*A->step];
            __vmult4dot_abs(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
        }
        if (i == A->cols)
            return;

        switch (A->cols-i) {
        case 3:
        case 2:
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            __vmult2dot_abs(y, yinc, a0, a1, x, xinc, alpha, A->rows);
            i += 2;
        }
        if (i < A->cols) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            __vmult1dot_abs(y, yinc, a0, x, xinc, alpha, A->rows);
        }
        return;
    }

    // Non-Transposed A here

    y = &Y->elems[yinc];
    for (j = 0; j < A->cols-3; j += 4) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        a2 = &A->elems[(j+2)*A->step];
        a3 = &A->elems[(j+3)*A->step];
        __vmult4axpy_abs(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
    }

    if (j == A->cols)
        return;

    switch (A->cols-j) {
    case 3:
    case 2:
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        __vmult2axpy_abs(y, yinc, a0, a1, x, xinc, alpha, A->rows);
        j += 2;
    }
    if (j < A->cols) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        __vmult1axpy_abs(y, yinc, a0, x, xinc, alpha, A->cols);
    }
}

static
void gemv_unb(
    armas_x_dense_t *Y,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    DTYPE alpha,
    int flags)
{
    int i, j, yinc, xinc;
    register DTYPE *y;
    register const DTYPE *x;
    register const DTYPE *a0, *a1, *a2, *a3;

    if (flags & ARMAS_ABS) {
        gemv_unb_abs(Y, A, X, alpha, flags);
        return;
    }

    xinc = X->rows == 1 ? X->step : 1;
    yinc = Y->rows == 1 ? Y->step : 1;
    if ((flags & ARMAS_TRANSA) || (flags & ARMAS_TRANS)) {

        x = &X->elems[0*xinc];
        for (i = 0; i < A->cols-3; i += 4) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            a2 = &A->elems[(i+2)*A->step];
            a3 = &A->elems[(i+3)*A->step];
            __vmult4dot(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
        }
        if (i == A->cols)
            return;

        switch (A->cols-i) {
        case 3:
        case 2:
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            __vmult2dot(y, yinc, a0, a1, x, xinc, alpha, A->rows);
            i += 2;
        }
        if (i < A->cols) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            __vmult1dot(y, yinc, a0, x, xinc, alpha, A->rows);
        }
        return;
    }

    // Non-Transposed A here

    y = &Y->elems[0*yinc];
    for (j = 0; j < A->cols-3; j += 4) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        a2 = &A->elems[(j+2)*A->step];
        a3 = &A->elems[(j+3)*A->step];
        __vmult4axpy(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
    }

    if (j == A->cols)
        return;

    switch (A->cols-j) {
    case 3:
    case 2:
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        __vmult2axpy(y, yinc, a0, a1, x, xinc, alpha, A->rows);
        j += 2;
    }
    if (j < A->cols) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        __vmult1axpy(y, yinc, a0, x, xinc, alpha, A->rows);
    }
}

static
void gemv_recursive(
    armas_x_dense_t *Y,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    DTYPE alpha,
    DTYPE beta,
    int flags,
    int min_mvec_size)
{
    armas_x_dense_t x0, y0;
    armas_x_dense_t A0;
    int ny = armas_x_size(Y);
    int nx = armas_x_size(X);

    // ny is rows in A, Y, nx is cols in A, rows in X
    if (ny < min_mvec_size || nx < min_mvec_size) {
        gemv_unb(Y, A, X, alpha, flags);
        return;
    }

    // 1st block [0, 0] -> [ny/2, nx/2]
    armas_x_subvector_unsafe(&x0, X, 0, nx/2);
    armas_x_subvector_unsafe(&y0, Y, 0, ny/2);
    if (flags & ARMAS_TRANS) {
        armas_x_submatrix_unsafe(&A0, A, 0, 0, nx/2, ny/2);
    } else {
        armas_x_submatrix_unsafe(&A0, A, 0, 0, ny/2, nx/2);
    }
    if (ny/2 < min_mvec_size || nx/2 < min_mvec_size) {
        gemv_unb(&y0, &A0, &x0, alpha, flags);
    } else {
        gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, min_mvec_size);
    }

    // 2nd block; lower part of y, A = [ny/2, 0] -> [ny-ny/2, nx/2]
    armas_x_subvector_unsafe(&y0, Y, ny/2, ny-ny/2);
    if (flags & ARMAS_TRANS) {
        armas_x_submatrix_unsafe(&A0, A, 0, ny/2, nx/2, ny-ny/2);
    } else {
        armas_x_submatrix_unsafe(&A0, A, ny/2, 0, ny-ny/2, nx/2);
    }
    if (ny/2 < min_mvec_size || nx/2 < min_mvec_size) {
        gemv_unb(&y0, &A0, &x0, alpha, flags);
    } else {
        gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, min_mvec_size);
    }

    // 3rd block; uppert of Y, lower part of X
    armas_x_subvector_unsafe(&x0, X, nx/2, nx-nx/2);
    armas_x_subvector_unsafe(&y0, Y, 0, ny/2);
    if (flags & ARMAS_TRANS) {
        armas_x_submatrix_unsafe(&A0, A, nx/2, 0, nx-nx/2, ny/2);
    } else {
        armas_x_submatrix_unsafe(&A0, A, 0, nx/2, ny/2, nx-nx/2);
    }
    if (ny/2 < min_mvec_size || nx/2 < min_mvec_size) {
        gemv_unb(&y0, &A0, &x0, alpha, flags);
    } else {
        gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, min_mvec_size);
    }

    // 4th block
    armas_x_subvector_unsafe(&y0, Y, ny/2, ny-ny/2);
    if (flags & ARMAS_TRANS) {
        armas_x_submatrix_unsafe(&A0, A, nx/2, ny/2, nx-nx/2, ny-ny/2);
    } else {
        armas_x_submatrix_unsafe(&A0, A, ny/2, nx/2, ny-ny/2, nx-nx/2);
    }
    if (ny/2 < min_mvec_size || nx/2 < min_mvec_size) {
        gemv_unb(&y0, &A0, &x0, alpha, flags);
    } else {
        gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, min_mvec_size);
    }
}

/**
 * Matrix vector multiply with no bounds check.
 */
int armas_x_mvmult_unsafe(
    DTYPE beta,
    armas_x_dense_t *y,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *x,
    int flags)
{
    if (armas_x_size(A) == 0)
        return 0;

    if (beta != ONE)
        armas_x_scale_unsafe(y, beta);
    gemv_unb(y, A, x, alpha, flags);
    return 0;
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
 *  @param[in]      beta scalar
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      alpha scalar
 *  @param[in]      A   source operand matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvmult(
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

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(A) == 0 || armas_x_size(x) == 0 || armas_x_size(y) == 0)
        return 0;

    if (!(armas_x_isvector(x) && armas_x_isvector(y))) {
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

    if (beta != ONE) {
        armas_x_scale_unsafe(y, beta);
    }
    if (conf->optflags & ARMAS_ORECURSIVE) {
        armas_env_t *env = armas_getenv();
        gemv_recursive(y, A, x, alpha, ONE, flags, env->blas2min);
    } else {
        gemv_unb(y, A, x, alpha, flags);
    }
    return 0;
}

#else
#warning "Missing defines; no code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
