
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mvupdate_unsafe) \
    || defined(armas_x_ext_mvupdate_trm_unsafe) \
    || defined(armas_x_mvupdate2_sym_unsafe)
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

/*
 * Unblocked update of general M-by-N matrix. A[i,j] = A[i,j] + alpha*x[i]*y[j]
 */
int armas_x_ext_mvupdate_unsafe(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y)
{
    DTYPE ah, dx, yk;
    armas_x_dense_t ac;

    for (int j = 0; j < A->cols; ++j) {
        armas_x_column(&ac, A, j);
        yk = armas_x_get_at_unsafe(Y, j);
        twoprod(&ah, &dx, alpha, yk);
        armas_x_ext_axpby_dx_unsafe(beta, &ac, ah, dx, X);
    }
    return 0;
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
int armas_x_ext_mvupdate(
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

    armas_x_ext_mvupdate_unsafe(beta, A, alpha, x, y);
    return 0;
}



#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
