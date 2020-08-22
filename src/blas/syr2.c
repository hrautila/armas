
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Symmetric matrix rank-2 update

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvupdate2_sym)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(armas_x_mvupdate_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "partition.h"
//! \endcond

/*
 *    a00 a10^T   x0 y0 y1^T   y0 x0 x1^T
 *    a10 A22     x1           y1
 * 
 *    a00 = a00 + alpha*x0*y0 + alpha*y0*x0
 *    a10 = a10 + alpya*y0*x1 + alpha*x0*y1
 */
static
void update_syr2_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    armas_x_dense_t a10, x1, y1;
    DTYPE x0, y0;

    for (int j = 0; j < A->cols; j++) {
        if (flags & ARMAS_UPPER) {
            // row of A
            armas_x_submatrix_unsafe(&a10, A, j, j, 1, A->cols - j);
        } else {
            // column of A
            armas_x_submatrix_unsafe(&a10, A, j, j, A->cols - j, 1);
        }
        armas_x_subvector_unsafe(&y1, Y, j, A->cols - j);
        armas_x_subvector_unsafe(&x1, X, j, A->cols - j);
        x0 = armas_x_get_at_unsafe(&x1, 0);
        y0 = armas_x_get_at_unsafe(&y1, 0);
        armas_x_axpby_unsafe(beta, &a10, alpha*y0, &x1);
        armas_x_axpby_unsafe(ONE,  &a10, alpha*x0, &y1);
    }
}

static
void update_syr2_recursive(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags,
    int min_mvec_size)
{
    armas_x_dense_t xT, xB, yT, yB;
    armas_x_dense_t ATL, ATR, ABL, ABR;

    if (A->rows < min_mvec_size) {
        update_syr2_unb(beta, A, alpha, X, Y, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, A->rows/2, ARMAS_PTOP);
    vec_partition_2x1(
        &yT,
        &yB, /**/ Y, A->rows/2, ARMAS_PTOP);

    update_syr2_recursive(beta, &ATL, alpha, &xT, &yT, flags, min_mvec_size);
    if (flags & ARMAS_UPPER) {
        armas_x_mvupdate_unsafe(beta, &ATR, alpha, &xT, &yB);
        armas_x_mvupdate_unsafe(ONE, &ATR, alpha, &yT, &xB);
    } else {
        armas_x_mvupdate_unsafe(beta, &ABL, alpha, &xB, &yT);
        armas_x_mvupdate_unsafe(ONE, &ABL, alpha, &yB, &xT);
    }
    update_syr2_recursive(beta, &ABR, alpha, &xB, &yB, flags, min_mvec_size);
}

/**
 * @brief Symmetric matrix rank-2 update.
 *
 * Computes
 *    -\f$ A = A + alpha \times X Y^T + alpha \times Y X^T \f$
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag *ARMAS_LOWER* (*ARMAR_UPPER*) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      X, Y source vector
 * @param[in]      flags flag bits
 * @param[in]      conf configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvupdate2_sym(
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
    if (A->cols != A->rows || ny != nx) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_env_t *env = armas_getenv();
    // default precision here
    switch (conf->optflags) {
    case ARMAS_ONAIVE:
        update_syr2_unb(beta, A, alpha, x, y, flags);
        break;

    case ARMAS_ORECURSIVE:
    default:
        update_syr2_recursive(beta, A, alpha, x, y, flags, env->blas2min);
        break;

    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRED */
