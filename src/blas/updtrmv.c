
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Triangular matrix update

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvupdate_trm) && \
  defined(armas_x_mvupdate_trm_rec) && \
  defined(armas_x_mvupdate_trm_unb)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(armas_x_mvupdate_rec)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 * (M is rows, N is columns.)
 */
static
void update_trmv_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    armas_x_dense_t a0, x0, y0;
    DTYPE dk;

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        for (int j = 0; j < A->rows; j++) {
            armas_x_submatrix_unsafe(&a0, A, j, j, 1, A->cols-j);
            armas_x_subvector_unsafe(&y0, Y, j, A->cols-j);
            dk = armas_x_get_at_unsafe(X, j);
            armas_x_axpby_unsafe(beta, &a0, alpha*dk, &y0);
        }
        break;
    case ARMAS_LOWER:
    default:
        for (int j = 0; j < A->cols; j++) {
            armas_x_submatrix_unsafe(&a0, A, j, j, A->rows-j, 1);
            armas_x_subvector_unsafe(&x0, X, j, A->rows-j);
            dk = armas_x_get_at_unsafe(Y, j);
            armas_x_axpby_unsafe(beta, &a0, alpha*dk, &x0);
        }
    }
}

static
void update_trmv_recursive(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags,
    int N, int M,
    int min_mvec_size)
{
    armas_x_dense_t x0, y0;
    armas_x_dense_t A0;
    int nd = min(M, N);

    if (M < min_mvec_size || N < min_mvec_size) {
        update_trmv_unb(beta, A, alpha, X, Y, flags);
        return;
    }

    armas_x_subvector_unsafe(&x0, X, 0, nd/2);
    armas_x_subvector_unsafe(&y0, Y, 0, nd/2);
    armas_x_submatrix_unsafe(&A0, A, 0, 0, nd/2, nd/2);
    if (nd/2 < min_mvec_size) {
        update_trmv_unb(beta, &A0, alpha, &x0, &y0, flags);
    } else {
        update_trmv_recursive(beta, &A0, alpha, &x0, &y0, flags, nd/2, nd/2, min_mvec_size);
    }

    if (flags & ARMAS_UPPER) {
        armas_x_subvector_unsafe(&y0, Y, nd/2, N-nd/2);
        armas_x_submatrix_unsafe(&A0, A, 0, nd/2, nd/2, N-nd/2);
        armas_x_mvupdate_rec(beta, &A0, alpha, &x0, &y0, flags);
    } else {
        armas_x_subvector_unsafe(&x0, X, nd/2, M-nd/2);
        armas_x_submatrix_unsafe(&A0, A, nd/2, 0, M-nd/2, nd/2);
        armas_x_mvupdate_rec(beta, &A0, alpha, &x0, &y0, flags);
    }

    armas_x_subvector_unsafe(&y0, Y, nd/2, N-nd/2);
    armas_x_subvector_unsafe(&x0, X, nd/2, M-nd/2);
    armas_x_submatrix_unsafe(&A0, A, nd/2, nd/2, M-nd/2, N-nd/2);
    if (N-nd/2 < min_mvec_size || M-nd/2 < min_mvec_size) {
        update_trmv_unb(beta, &A0, alpha, &x0, &y0, flags);
    } else {
        update_trmv_recursive(beta, &A0, alpha, &x0, &y0, flags, N-nd/2, M-nd/2, min_mvec_size);
    }
}

void armas_x_mvupdate_trm_rec(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    int flags)
{
    armas_env_t *env = armas_getenv();
    update_trmv_recursive(beta, A, alpha, x, y, flags, A->cols, A->rows, env->blas2min);
}

void armas_x_mvupdate_trm_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    int flags)
{
    update_trmv_unb(beta, A, alpha, x, y, flags);
}

/**
 * @brief General triangular/trapezoidial matrix rank update.
 *
 * Computes 
 *    - \f$ A = A + alpha \times X Y^T \f$
 *
 * where A is upper (lower) triangular or trapezoidial matrix as defined with
 * flag bits *ARMAS_UPPER* (*ARMAS_LOWER*).
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      X source vector
 * @param[in]      Y source vector
 * @param[in]      flags flag bits
 * @param[in]      conf  configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvupdate_trm(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    int flags,
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
    switch (conf->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
    case ARMAS_ORECURSIVE:
        update_trmv_recursive(beta, A, alpha, x, y, flags, ny, nx, env->blas2min);
        break;

    case ARMAS_ONAIVE:
    default:
        update_trmv_unb(beta, A, alpha, x, y, flags);
        break;
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
