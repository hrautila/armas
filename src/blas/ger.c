
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix rank update

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvupdate) && \
  defined(armas_x_mvupdate_unb) && \
  defined(armas_x_mvupdate_rec)

#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond


static
void update4axpy(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int M)
{
    register int i;
    register DTYPE *a0, *a1, *a2, *a3;
    register DTYPE y0, y1, y2, y3;
    int xinc = X->rows == 1 ? X->step : 1;
    int yinc = Y->rows == 1 ? Y->step : 1;

    y0 = alpha*Y->elems[0];
    y1 = alpha*Y->elems[yinc];
    y2 = alpha*Y->elems[2*yinc];
    y3 = alpha*Y->elems[3*yinc];
    a0 = &A->elems[0];
    a1 = &A->elems[A->step];
    a2 = &A->elems[2*A->step];
    a3 = &A->elems[3*A->step];
    for (i = 0; i < M-3; i += 4) {
        a0[(i+0)] = beta*a0[(i+0)] + X->elems[(i+0)*xinc]*y0;
        a1[(i+0)] = beta*a1[(i+0)] + X->elems[(i+0)*xinc]*y1;
        a2[(i+0)] = beta*a2[(i+0)] + X->elems[(i+0)*xinc]*y2;
        a3[(i+0)] = beta*a3[(i+0)] + X->elems[(i+0)*xinc]*y3;

        a0[(i+1)] = beta*a0[(i+1)] + X->elems[(i+1)*xinc]*y0;
        a1[(i+1)] = beta*a1[(i+1)] + X->elems[(i+1)*xinc]*y1;
        a2[(i+1)] = beta*a2[(i+1)] + X->elems[(i+1)*xinc]*y2;
        a3[(i+1)] = beta*a3[(i+1)] + X->elems[(i+1)*xinc]*y3;

        a0[(i+2)] = beta*a0[(i+2)] + X->elems[(i+2)*xinc]*y0;
        a1[(i+2)] = beta*a1[(i+2)] + X->elems[(i+2)*xinc]*y1;
        a2[(i+2)] = beta*a2[(i+2)] + X->elems[(i+2)*xinc]*y2;
        a3[(i+2)] = beta*a3[(i+2)] + X->elems[(i+2)*xinc]*y3;

        a0[(i+3)] = beta*a0[(i+3)] + X->elems[(i+3)*xinc]*y0;
        a1[(i+3)] = beta*a1[(i+3)] + X->elems[(i+3)*xinc]*y1;
        a2[(i+3)] = beta*a2[(i+3)] + X->elems[(i+3)*xinc]*y2;
        a3[(i+3)] = beta*a3[(i+3)] + X->elems[(i+3)*xinc]*y3;
    }
    if (i == M)
        return;
    switch (M-i) {
    case 3:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
        a1[i] = beta*a1[i] + X->elems[i*xinc]*y1;
        a2[i] = beta*a2[i] + X->elems[i*xinc]*y2;
        a3[i] = beta*a3[i] + X->elems[i*xinc]*y3;
        i++;
    case 2:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
        a1[i] = beta*a1[i] + X->elems[i*xinc]*y1;
        a2[i] = beta*a2[i] + X->elems[i*xinc]*y2;
        a3[i] = beta*a3[i] + X->elems[i*xinc]*y3;
        i++;
    case 1:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
        a1[i] = beta*a1[i] + X->elems[i*xinc]*y1;
        a2[i] = beta*a2[i] + X->elems[i*xinc]*y2;
        a3[i] = beta*a3[i] + X->elems[i*xinc]*y3;
    }
}


static
void update1axpy(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int M)
{
    register int i;
    register DTYPE *a0;
    register DTYPE y0;
    int xinc = X->rows == 1 ? X->step : 1;

    y0 = alpha*Y->elems[0];
    a0 = &A->elems[0];
    for (i = 0; i < M-3; i += 4) {
        a0[(i+0)] = beta*a0[(i+0)] + X->elems[(i+0)*xinc]*y0;
        a0[(i+1)] = beta*a0[(i+1)] + X->elems[(i+1)*xinc]*y0;
        a0[(i+2)] = beta*a0[(i+2)] + X->elems[(i+2)*xinc]*y0;
        a0[(i+3)] = beta*a0[(i+3)] + X->elems[(i+3)*xinc]*y0;
    }
    if (i == M)
        return;
    switch (M-i) {
    case 3:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
        i++;
    case 2:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
        i++;
    case 1:
        a0[i] = beta*a0[i] + X->elems[i*xinc]*y0;
    }
}

/*
 * Unblocked update of general M-by-N matrix.
 */
static
void update_ger_unb(
    armas_x_dense_t *A,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    DTYPE beta,
    DTYPE alpha,
    int flags,
    int N, int M)
{
    armas_x_dense_t y0;
    armas_x_dense_t A0;
    register int j;
    for (j = 0; j < N-3; j += 4) {
        armas_x_submatrix_unsafe(&A0, A, 0, j, A->rows, 4);
        armas_x_subvector_unsafe(&y0, Y, j, 4);
        update4axpy(beta, &A0, alpha, X, &y0, M);
    }
    if (j == N)
        return;

    for (; j < N; j++) {
        armas_x_submatrix_unsafe(&A0, A, 0, j, A->rows, 1);
        armas_x_subvector_unsafe(&y0, Y, j, 1);
        update1axpy(beta, &A0, alpha, X, &y0, M);
    }
}

static
void update_ger_recursive(
    armas_x_dense_t *A,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    DTYPE beta,
    DTYPE alpha,
    int flags,
    int N, int M,
    int min_mvec_size)
{
    armas_x_dense_t x0, y0;
    armas_x_dense_t A0;

    if (M < min_mvec_size || N < min_mvec_size) {
        update_ger_unb(A, X, Y, beta, alpha, flags, N, M);
        return;
    }

    //printf("update 1. block.. [0,0] - [%d,%d]\n", M/2, N/2);
    armas_x_subvector_unsafe(&x0, X, 0, M/2);
    armas_x_subvector_unsafe(&y0, Y, 0, N/2);
    armas_x_submatrix_unsafe(&A0, A, 0, 0, M/2, N/2);
    update_ger_recursive(&A0, &x0, &y0, beta, alpha, flags, N/2, M/2, min_mvec_size);

    //printf("update 2. block... [%d,0] - [%d,%d]\n", M/2, M-M/2, N/2);
    armas_x_subvector_unsafe(&x0, X, M/2, M-M/2);
    armas_x_submatrix_unsafe(&A0, A, M/2, 0, M-M/2, N/2);
    update_ger_recursive(&A0, &x0, &y0, beta, alpha, flags, N/2, M-M/2, min_mvec_size);

    //printf("update 3. block... [0,%d] - [%d,%d]\n", N/2, M/2, N-N/2);
    armas_x_subvector_unsafe(&x0, X, 0, M/2);
    armas_x_subvector_unsafe(&y0, Y, N/2, N-N/2);
    armas_x_submatrix_unsafe(&A0, A, 0, N/2, M/2, N-N/2);
    update_ger_recursive(&A0, &x0, &y0, beta, alpha, flags, N-N/2, M/2, min_mvec_size);

    //printf("update 4. block... [%d,%d] - [%d,%d]\n", M/2, N/2, M-M/2, N-N/2);
    armas_x_subvector_unsafe(&x0, X, M/2, M-M/2);
    armas_x_submatrix_unsafe(&A0, A, M/2, N/2, M-M/2, N-N/2);
    update_ger_recursive(&A0, &x0, &y0, beta, alpha, flags, N-N/2, M-M/2, min_mvec_size);
}

void armas_x_mvupdate_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    update_ger_unb(A, X, Y, beta, alpha, flags, A->cols, A->rows);
}

void armas_x_mvupdate_rec(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    armas_env_t *env = armas_getenv();
    update_ger_recursive(A, X, Y, beta, alpha, flags, A->cols, A->rows, env->blas2min);
}

/**
 * @brief General matrix rank update.
 *
 * Computes
 *   - \f$ A = A + alpha \times X Y^T \f$
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out]  A target matrix
 * @param[in]      X source vector
 * @param[in]      Y source vector
 * @param[in]      alpha scalar multiplier
 * @param[in]      conf  configuration block
 *
 * @ingroup blas2
 */
int armas_x_mvupdate(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    armas_conf_t *conf)
{
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
    if (A->cols != ny || A->rows != nx) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_env_t *env = armas_getenv();
    // normal precision here
    switch (conf->optflags) {
    case ARMAS_ONAIVE:
        update_ger_unb(A, x, y, beta, alpha, 0, ny, nx);
        break;

    case ARMAS_ORECURSIVE:
    default:
        update_ger_recursive(A, x, y, beta, alpha, 0, ny, nx, env->blas2min);
        break;
    }
    return 0;
}

#else
#warning "Missing defines; no code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
