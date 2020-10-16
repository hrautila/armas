
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! matrix-vector multiplication

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mvmult) && defined(armas_mvmult_unsafe)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "nosimd/mvec.h"
#include "partition.h"

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
static
void gemv_unb_abs(
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
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
            vmult4dot_abs(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
        }
        if (i == A->cols)
            return;

        switch (A->cols-i) {
        case 3:
        case 2:
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            vmult2dot_abs(y, yinc, a0, a1, x, xinc, alpha, A->rows);
            i += 2;
        }
        if (i < A->cols) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            vmult1dot_abs(y, yinc, a0, x, xinc, alpha, A->rows);
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
        vmult4axpy_abs(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
    }

    if (j == A->cols)
        return;

    switch (A->cols-j) {
    case 3:
    case 2:
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        vmult2axpy_abs(y, yinc, a0, a1, x, xinc, alpha, A->rows);
        j += 2;
    }
    if (j < A->cols) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        vmult1axpy_abs(y, yinc, a0, x, xinc, alpha, A->cols);
    }
}

static
void gemv_unb(
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
    int flags)
{
    int i, j, yinc, xinc;
    register DTYPE *y;
    register const DTYPE *x;
    register const DTYPE *a0, *a1, *a2, *a3;

    if (flags & ARMAS_ABS) {
        gemv_unb_abs(Y, alpha, A, X, flags);
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
            vmult4dot(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
        }
        if (i == A->cols)
            return;

        switch (A->cols-i) {
        case 3:
        case 2:
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            a1 = &A->elems[(i+1)*A->step];
            vmult2dot(y, yinc, a0, a1, x, xinc, alpha, A->rows);
            i += 2;
        }
        if (i < A->cols) {
            y = &Y->elems[i*yinc];
            a0 = &A->elems[(i+0)*A->step];
            vmult1dot(y, yinc, a0, x, xinc, alpha, A->rows);
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
        vmult4axpy(y, yinc, a0, a1, a2, a3, x, xinc, alpha, A->rows);
    }

    if (j == A->cols)
        return;

    switch (A->cols-j) {
    case 3:
    case 2:
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        a1 = &A->elems[(j+1)*A->step];
        vmult2axpy(y, yinc, a0, a1, x, xinc, alpha, A->rows);
        j += 2;
    }
    if (j < A->cols) {
        x = &X->elems[j*xinc];
        a0 = &A->elems[(j+0)*A->step];
        vmult1axpy(y, yinc, a0, x, xinc, alpha, A->rows);
    }
}

static
void gemv_recursive(
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
    int flags,
    int min_mvec_size)
{
    armas_dense_t xT, xB, yT, yB;
    armas_dense_t ATL, ATR, ABL, ABR;
    int ny = armas_size(Y);
    int nx = armas_size(X);

    // ny is rows in A, Y, nx is cols in A, rows in X
    if (min_mvec_size == 0 || ny < min_mvec_size || nx < min_mvec_size) {
        gemv_unb(Y, alpha, A, X, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, nx/2, ARMAS_PTOP);
    vec_partition_2x1(
        &yT,
        &yB, /**/ Y, ny/2, ARMAS_PTOP);

    gemv_recursive(&yT, alpha, &ATL, &xT, flags, min_mvec_size);
    if (flags & ARMAS_TRANS) {
        gemv_recursive(&yT, alpha, &ABL, &xB, flags, min_mvec_size);
        gemv_recursive(&yB, alpha, &ATR, &xT, flags, min_mvec_size);
    } else {
        gemv_recursive(&yT, alpha, &ATR, &xB, flags, min_mvec_size);
        gemv_recursive(&yB, alpha, &ABL, &xT, flags, min_mvec_size);
    }
    gemv_recursive(&yB, alpha, &ABR, &xB, flags, min_mvec_size);
}

/*
 * Matrix vector multiply with no bounds check.
 */
void armas_mvmult_unsafe(
    DTYPE beta,
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *x,
    int flags)
{
    armas_env_t *env = armas_getenv();
    if (armas_size(A) == 0)
        return;

    if (beta != ONE)
        armas_scale_unsafe(y, beta);
    gemv_recursive(y, alpha, A, x, flags, env->blas2min);
}


/**
 * @brief General matrix-vector multiply.
 *
 * Computes
 *   - \f$ Y = alpha \times A X + beta \times Y \f$
 *   - \f$ Y = alpha \times A^T X + beta \times Y  \f$   if *ARMAS_TRANS* set
 *
 *  @param[in]      beta scalar
 *  @param[in,out]  y   target and source vector
 *  @param[in]      alpha scalar
 *  @param[in]      A   source operand matrix
 *  @param[in]      x   source operand vector
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blas
 */
int armas_mvmult(
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

    if (!conf)
        conf = armas_conf_default();

    if (armas_size(A) == 0 || armas_size(x) == 0 || armas_size(y) == 0)
        return 0;

    if (!(armas_isvector(x) && armas_isvector(y))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
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
        return -ARMAS_ESIZE;
    }

    if (beta != ONE) {
        armas_scale_unsafe(y, beta);
    }
    if (conf->optflags & ARMAS_ONAIVE) {
        gemv_unb(y, alpha, A, x, flags);
    } else {
        armas_env_t *env = armas_getenv();
        gemv_recursive(y, alpha, A, x, flags, env->blas2min);
    }
    return 0;
}

#else
#warning "Missing defines; no code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
