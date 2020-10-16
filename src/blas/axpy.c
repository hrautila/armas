
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Scaled vector to vector addition.
//! @addtogroup blas
//! @{

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_axpy) && defined(armas_axpby) && defined(armas_axpby_unsafe)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

void armas_axpby_unsafe(
    DTYPE beta,
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *x)
{
    int i, kx, ky;
    DTYPE y0, y1, y2, y3;
    int N = armas_size(y);
    int yinc = y->rows == 1 ? y->step : 1;
    int xinc = x->rows == 1 ? x->step : 1;

    for (i = 0; i < N-3; i += 4) {
        y0 = y->elems[(i+0)*yinc] * beta;
        y1 = y->elems[(i+1)*yinc] * beta;
        y2 = y->elems[(i+2)*yinc] * beta;
        y3 = y->elems[(i+3)*yinc] * beta;
        y->elems[(i+0)*yinc] = y0 + alpha*x->elems[(i+0)*xinc];
        y->elems[(i+1)*yinc] = y1 + alpha*x->elems[(i+1)*xinc];
        y->elems[(i+2)*yinc] = y2 + alpha*x->elems[(i+2)*xinc];
        y->elems[(i+3)*yinc] = y3 + alpha*x->elems[(i+3)*xinc];
    }
    if (i == N)
        return;

    kx = i*xinc; ky = i*yinc;
    switch (N-i) {
    case 3:
        y0 = y->elems[ky] * beta;
        y->elems[ky] = y0 + alpha*x->elems[kx];
        kx += xinc; ky += yinc;
    case 2:
        y0 = y->elems[ky] * beta;
        y->elems[ky] = y0 + alpha*x->elems[kx];
        kx += xinc; ky += yinc;
    case 1:
        y0 = y->elems[ky] * beta;
        y->elems[ky] = y0 + alpha*x->elems[kx];
    }
}

/**
 * @brief Compute \f$ y = beta*y + alpha*x \f$
 *
 * @param[in]     beta scalar multiplier
 * @param[in,out] y target and source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     x source vector
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval < 0 Failed, conf->error holds error code
 */
int armas_axpby(
    DTYPE beta,
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *x,
    armas_conf_t *conf)
{
    // only for column or row vectors
    if (!conf)
        conf = armas_conf_default();

    if (!(armas_isvector(x) && armas_isvector(y))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (armas_size(x) != armas_size(y)) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    armas_axpby_unsafe(beta, y, alpha, x);
    return 0;
}

/**
 * @brief Compute \f$ y = y + alpha*x \f$
 *
 * @param[in,out] y target and source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     x source vector
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval < 0 Failed, conf->error holds error code
 */
int armas_axpy(
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *x,
    armas_conf_t *conf)
{
    return armas_axpby(ONE, y, alpha, x, conf);
}
//! @}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
