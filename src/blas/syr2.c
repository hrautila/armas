
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
#if defined(armas_x_mvupdate_trm_unb) && defined(armas_x_mvupdate_rec)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "nosimd/mvec.h"

//! \endcond

static
void update_syr2_recursive(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags,
    int N,
    int min_mvec_size)
{
    armas_x_dense_t x0, y0, A0;

    if (N < min_mvec_size) {
        armas_x_mvupdate_trm_unb(beta, A, alpha, X, Y, flags);
        armas_x_mvupdate_trm_unb(__ONE, A, alpha, Y, X, flags);
        return;
    }

    armas_x_subvector_unsafe(&x0, X, 0, N/2);
    armas_x_subvector_unsafe(&y0, Y, 0, N/2);
    armas_x_submatrix_unsafe(&A0, A, 0, 0, N/2, N/2);
    if (N/2 < min_mvec_size) {
        armas_x_mvupdate_trm_unb(beta, &A0, alpha, &x0, &y0, flags);
        armas_x_mvupdate_trm_unb(__ONE, &A0, alpha, &y0, &x0, flags);
    } else {
        update_syr2_recursive(beta, &A0, alpha, &x0, &y0, flags, N/2, min_mvec_size);
    }

    if (flags & ARMAS_UPPER) {
        armas_x_subvector_unsafe(&x0, X, 0, N/2);
        armas_x_subvector_unsafe(&y0, Y, N/2, N-N/2);
        armas_x_submatrix_unsafe(&A0, A, 0, N/2, N/2, N-N/2);
        armas_x_mvupdate_rec(beta, &A0, alpha, &x0, &y0, flags);

        armas_x_subvector_unsafe(&x0, X, N/2, N-N/2);
        armas_x_subvector_unsafe(&y0, Y, 0, N/2);
        armas_x_mvupdate_rec(__ONE, &A0, alpha, &y0, &x0, flags);
    } else {
        armas_x_subvector_unsafe(&y0, Y, 0, N/2);
        armas_x_subvector_unsafe(&x0, X, N/2, N-N/2);
        armas_x_submatrix_unsafe(&A0, A, N/2, 0, N-N/2, N/2);
        armas_x_mvupdate_rec(beta, &A0, alpha, &x0, &y0, flags);

        armas_x_subvector_unsafe(&y0, Y, N/2, N-N/2);
        armas_x_subvector_unsafe(&x0, X, 0, N/2);
        armas_x_mvupdate_rec(__ONE, &A0, alpha, &y0, &x0, flags);
    }

    armas_x_subvector_unsafe(&y0, Y, N/2, N-N/2);
    armas_x_subvector_unsafe(&x0, X, N/2, N-N/2);
    armas_x_submatrix_unsafe(&A0, A, N/2, N/2, N-N/2, N-N/2);
    if (N-N/2 < min_mvec_size) {
        armas_x_mvupdate_trm_unb(beta, &A0, alpha, &x0, &y0, flags);
        armas_x_mvupdate_trm_unb(__ONE, &A0, alpha, &y0, &x0, flags);
    } else {
        update_syr2_recursive(beta, &A0, alpha, &x0, &y0, flags, N-N/2, min_mvec_size);
    }
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
    case ARMAS_ORECURSIVE:
        update_syr2_recursive(beta, A, alpha, x, y, flags, nx, env->blas2min);
        break;

    case ARMAS_ONAIVE:
    default:
        armas_x_mvupdate_trm_unb(beta, A, alpha, x, y, flags);
        armas_x_mvupdate_trm_unb(__ONE, A, alpha, y, x, flags);
        break;

    }
    return 0;
}

#else
#warning "Missing defines; no code!"

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRED */
