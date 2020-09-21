
// Copyright (c) Harri Rautila, 2015-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_axpy_unsafe) && defined(armas_x_ext_axpby_unsafe) \
    && defined(armas_x_ext_axpby) && defined(armas_x_ext_axpby_dx_unsafe)
#define ARMAS_PROVIDES 1
#endif
// extended precision enabled
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

int armas_x_ext_axpy_unsafe(
    armas_x_dense_t *Y,
    DTYPE alpha,
    const armas_x_dense_t *X)
{
    register int i, kx, ky;
    DTYPE y0, y1, x0, x1;
    DTYPE p0, p1, c0, c1;
    int xinc = X->rows == 1 ? X->step : 1;
    int yinc = Y->rows == 1 ? Y->step : 1;
    int N = armas_x_size(X);

    for (i = 0; i < N-1; i += 2) {
        twoprod(&x0, &p0, X->elems[(i+0)*xinc], alpha);
        twoprod(&x1, &p1, X->elems[(i+1)*xinc], alpha);

        twosum(&y0, &c0, x0, Y->elems[(i+0)*yinc]);
        twosum(&y1, &c1, x1, Y->elems[(i+1)*yinc]);

        Y->elems[(i+0)*yinc] = (p0 + c0) + y0;
        Y->elems[(i+1)*yinc] = (p1 + c1) + y1;
    }
    if (i == N)
        return 0;

    kx = i*xinc; ky = i*yinc;
    twoprod(&x0, &p0, X->elems[kx], alpha);
    twosum(&y0, &c0, x0, Y->elems[ky]);
    Y->elems[ky] = (p0 + c0) + y0;
    return 0;
}

int armas_x_ext_axpby_unsafe(
    DTYPE beta,
    armas_x_dense_t *Y,
    DTYPE alpha,
    const armas_x_dense_t *X)
{
    register int i, kx, ky, xinc, yinc, N;
    DTYPE y0, y1, p0, p1, x0, x1, c0, c1;

    if (beta == ONE)
        return armas_x_ext_axpy_unsafe(Y, alpha, X);

    xinc = X->rows == 1 ? X->step : 1;
    yinc = Y->rows == 1 ? Y->step : 1;
    N = armas_x_size(X);

    for (i = 0; i < N-1; i += 2) {
        twoprod(&x0, &p0, X->elems[(i+0)*xinc], alpha);
        twoprod(&x1, &p1, X->elems[(i+1)*xinc], alpha);

        twoprod(&y0, &c0, Y->elems[(i+0)*yinc], beta);
        twoprod(&y1, &c1, Y->elems[(i+1)*yinc], beta);
        p0 += c0;
        p1 += c1;

        twosum(&y0, &c0, x0, y0);
        twosum(&y1, &c1, x1, y1);

        Y->elems[(i+0)*yinc] = (p0 + c0) + y0;
        Y->elems[(i+1)*yinc] = (p1 + c1) + y1;
    }
    if (i == N)
        return 0;

    kx = i*xinc; ky = i*yinc;
    twoprod(&x0, &p0, X->elems[kx], alpha);
    twoprod(&y0, &c0, Y->elems[ky], beta);
    p0 += c0;
    twosum(&y0, &c0, x0, y0);
    Y->elems[ky] = p0 + c0 + y0;
    return 0;
}

// Compute y = beta*y + (alpha, dx) * x
int armas_x_ext_axpby_dx_unsafe(
    DTYPE beta,
    armas_x_dense_t *Y,
    DTYPE alpha,
    DTYPE dx,
    const armas_x_dense_t *X)
{
    register int i, xinc, yinc, N;
    DTYPE y0, p0, x0, c0, z0, y1, c1;

    xinc = X->rows == 1 ? X->step : 1;
    yinc = Y->rows == 1 ? Y->step : 1;
    N = armas_x_size(X);

    for (i = 0; i < N; ++i) {
        twoprod(&y0, &c0, Y->elems[i*yinc], beta);
        twoprod(&x0, &p0, X->elems[i*xinc], alpha);
        z0 = dx * X->elems[i*xinc];
        p0 += c0;

        twosum(&y1, &c1, x0, y0);
        Y->elems[i*yinc] = y1 + z0 + c1 + p0; //(p0 + c0) + y0;
    }
    return 0;
}

/**
 * @brief Compute \f$ y = beta*y + alpha*x \f$ in extended precision.
 *
 * @param[in]     beta scalar multiplier
 * @param[in,out] y target and source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     x source vector
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval <0 Failed, conf->error holds error code
 *
 * @ingroup blasext
 */
int armas_x_ext_axpby(
    DTYPE beta,
    armas_x_dense_t *y,
    DTYPE alpha,
    const armas_x_dense_t *x,
    armas_conf_t *conf)
{
    // only for column or row vectors
    if (!conf)
        conf = armas_conf_default();

    if (!(armas_x_isvector(x) && armas_x_isvector(y))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (armas_x_size(x) != armas_x_size(y)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_x_ext_axpby_unsafe(beta, y, alpha, x);
    return 0;
}
#else
#warning "Missing defines! No code!"
#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */
