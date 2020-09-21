
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mvmult) && defined(armas_x_ext_mvmult_unsafe) && defined(armas_x_ext_mvupdate_unsafe)
#define ARMAS_PROVIDES 1
#endif
// if extended precision enabled and requested
#if defined(armas_x_ext_adot_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

int armas_x_ext_mvmult_unsafe(
    DTYPE beta,
    armas_x_dense_t *Y,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    int flags)
{
    int i, nY;
    armas_x_dense_t a0;
    DTYPE y0, c0;

    nY = armas_x_size(Y);

    if (flags & ARMAS_TRANS) {
        for (i = 0; i < nY; ++i) {
            armas_x_column(&a0, A, i);
            twoprod(&y0, &c0, beta, armas_x_get_at_unsafe(Y, i));
            armas_x_ext_adot_unsafe(&y0, &c0, alpha, &a0, X);
            armas_x_set_at_unsafe(Y, i, y0+c0);
        }
        return 0;
    }

    for (i = 0; i < nY; i++) {
        armas_x_row(&a0, A, i);
        twoprod(&y0, &c0, beta, armas_x_get_at_unsafe(Y, i));
        armas_x_ext_adot_unsafe(&y0, &c0, alpha, &a0, X);
        armas_x_set_at_unsafe(Y, i, y0+c0);
    }
    return 0;
}


/*
 * Compute error free translation Y + dY = beta * (Y + dY) + alpha * A * X
 *
 */
void armas_x_ext_mvmult_dx_unsafe(
    DTYPE beta,
    armas_x_dense_t *Y,
    armas_x_dense_t *dY,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *X,
    int flags)
{
    int i;
    armas_x_dense_t a0;
    DTYPE y0, c0;
    int nY = armas_x_size(Y);

    if (flags & ARMAS_TRANS) {
        for (i = 0; i < nY; ++i) {
            twoprod(&y0, &c0, beta, armas_x_get_at_unsafe(Y, i));
            c0 += beta * armas_x_get_at_unsafe(dY, i);
            armas_x_column(&a0, A, i);
            armas_x_ext_adot_unsafe(&y0, &c0, alpha, &a0, X);
            armas_x_set_at_unsafe(Y, i, y0);
            armas_x_set_at_unsafe(dY, i, c0);
        }
        return;
    }

    for (i = 0; i < nY; ++i) {
        twoprod(&y0, &c0, beta, armas_x_get_at_unsafe(Y, i));
        c0 += beta * armas_x_get_at_unsafe(dY, i);
        armas_x_row(&a0, A, i);
        armas_x_ext_adot_unsafe(&y0, &c0, alpha, &a0, X);
        armas_x_set_at_unsafe(Y, i, y0);
        armas_x_set_at_unsafe(dY, i, c0);
    }
}

/**
* @brief General matrix-vector multiply with extended precision
 *
 * Computes
 *   - \f$ Y = alpha \times A X + beta \times Y \f$
 *   - \f$ Y = alpha \times A^T X + beta \times Y  \f$   if *ARMAS_TRANS* set
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
 * @ingroup blasext
 */
int armas_x_ext_mvmult(
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

    armas_x_ext_mvmult_unsafe(beta, y, alpha, A, x, flags);
    return 0;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
