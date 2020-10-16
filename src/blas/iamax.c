
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Absolute maximum

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_iamax) && defined(armas_amax)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

// return index of max absolute value
static
int vec_iamax(const armas_dense_t *X,  int N)
{
    int i, ix, n, xinc;
    register ABSTYPE max, c0, c1;

    if (N <= 1)
        return 0;

    xinc = X->rows == 1 ? X->step : 1;
    max = 0.0;
    ix = 0;
    for (i = 0; i < N-1; i += 2) {
        c0 = ABS(X->elems[(i+0)*xinc]);
        c1 = ABS(X->elems[(i+1)*xinc]);
        if (c1 > c0) {
            n = 1;
            c0 = c1;
        }
        if (c0 > max) {
            ix = i+n;
            max = c0;
        }
        n = 0;
    }
    if (i < N) {
        c0 = ABS(X->elems[i*xinc]);
        ix = c0 > max ? N-1 : ix;
    }
    return ix;
}

static
int vec_iamin(const armas_dense_t *X,  int N)
{
    register int i, ix, n, xinc;
    register ABSTYPE min, c0, c1;

    if (N <= 1)
        return 0;

    xinc = X->rows == 1 ? X->step : 1;
    min = ABS(X->elems[0]);
    ix = 0;
    for (i = 0; i < N-1; i += 2) {
        c0 = ABS(X->elems[(i+0)*xinc]);
        c1 = ABS(X->elems[(i+1)*xinc]);
        if (c1 < c0) {
            n = 1;
            c0 = c1;
        }
        if (c0 < min) {
            ix = i+n;
            min = c0;
        }
        n = 0;
    }
    if (i < N) {
        c0 = ABS(X->elems[i*xinc]);
        ix = c0 < min ? N-1 : ix;
    }
    return ix;
}


static
int vec_iamax2(const armas_dense_t *X,  int N)
{
    int i, ix, xinc;
    DTYPE *ep;
    register ABSTYPE max, c0;

    if (N <= 1)
        return 0;

    xinc = X->rows == 1 ? X->step : 1;
    max = 0.0;
    ix = 0;
    for (i = 0, ep = X->elems; i < N; ++i, ep += xinc) {
        c0 = ABS(*ep);
        if (c0 > max) {
            ix = i;
            max = c0;
        }
    }
    return ix;
}

static
int vec_iamin2(const armas_dense_t *X,  int N)
{
    int i, ix, xinc;
    DTYPE *ep;
    register ABSTYPE min, c0;

    if (N <= 1)
        return 0;

    xinc = X->rows == 1 ? X->step : 1;
    ep = X->elems;
    min = ABS(*ep++);
    ix = 0;
    for (i = 1; i < N; ++i, ep += xinc) {
        c0 = ABS(*ep);
        if (c0 < min) {
            ix = i;
            min = c0;
        }
    }
    return ix;
}

/**
 * @brief Index of \f$ \max_{k} |x| \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @retval >= 0 index of maximum element
 * @retval -1  error, conf->error holds error code
 *
 * @ingroup blas
 */
int armas_iamax(const armas_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (armas_size(x) == 0) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    return vec_iamax2(x, armas_size(x));
}

/**
 * @brief Maximum absolute value of vector.
 */
ABSTYPE armas_amax(const armas_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    int imax = armas_iamax(x, conf);
    if (imax >= 0) {
        return ABS(armas_get_at_unsafe(x, imax));
    }
    return ZERO;
}


/**
 * @brief Index of \f$ \min_{k} |x| \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @retval >= 0 index of minimum element
 * @retval -1  error, conf->error holds error code
 *
 * @ingroup blas1
 */
int armas_iamin(const armas_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (armas_size(x) == 0) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    return vec_iamin2(x, armas_size(x));
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */
