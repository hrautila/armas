
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 *  @file
 * Absolute sum
 * @addtogroup blas
 * @{
 */

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_asum)
#define ARMAS_PROVIDES 1
#endif
// this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

// return sum of absolute values
static
ABSTYPE vec_asum(const armas_x_dense_t *X,  int N)
{
    register int i, k, xinc;
    register ABSTYPE c0, c1, c2, c3;
    register ABSTYPE z0, z1, z2, z3;

    xinc = X->rows == 1 ? X->step : 1;
    c0 = c1 = c2 = c3 = 0.0;
    for (i = 0; i < N-3; i += 4) {
        z0 = ABS(X->elems[(i+0)*xinc]);
        z1 = ABS(X->elems[(i+1)*xinc]);
        z2 = ABS(X->elems[(i+2)*xinc]);
        z3 = ABS(X->elems[(i+3)*xinc]);
        c0 += z0;
        c1 += z1;
        c2 += z2;
        c3 += z3;
    }
    if (i == N)
        goto update;

    k = i*xinc;
    switch (N-i) {
    case 3:
        c0 += ABS(X->elems[k]);
        k += xinc;
    case 2:
        c1 += ABS(X->elems[k]);
        k += xinc;
    case 1:
        c2 += ABS(X->elems[k]);
    }
update:
    return c0 + c1 + c2 + c3;
}

static
DTYPE vec_asum_kahan(const armas_x_dense_t *X, int N)
{
    register int k, xinc;
    register ABSTYPE c0, s0, c1, s1;
    register ABSTYPE t0, y0, t1, y1;

    xinc = X->rows == 1 ? X->step : 1;
    c0 = c1 = s0 = s1 = ZERO;
    for (k = 0; k < N-1; k += 2) {
        y0 = ABS(X->elems[(k+0)*xinc]) - c0;
        t0 = s0 + y0;
        c0 = (t0 - s0) - y0;
        s0 = t0;

        y1 = ABS(X->elems[(k+1)*xinc]) - c1;
        t1 = s1 + y1;
        c1 = (t1 - s1) - y1;
        s1 = t1;
    }
    if (k == N)
        return s0 + s1;

    y0 = ABS(X->elems[(k+0)*xinc]) - c0;
    t0 = s0 + y0;
    c0 = (t0 - s0) - y0;
    s0 = t0;
    return s0 + s1;
}

static inline
armas_x_dense_t *__subvec(armas_x_dense_t *x, const armas_x_dense_t *y, int K, int N)
{
    return armas_x_subvector_unsafe(x, y, K, N);
}

static
DTYPE vec_asum_recursive(const armas_x_dense_t *X, int n, int min_mvec)
{
    register DTYPE c0, c1, c2, c3;
    armas_x_dense_t x0;
    int n2 = n/2;
    int n4 = n/4;
    int nn = n - n2 - n4;

    if (n < min_mvec)
        return vec_asum(X, n);

    if (n/2 < min_mvec) {
        c0 = vec_asum(__subvec(&x0, X, 0, n2),   n2);
        c1 = vec_asum(__subvec(&x0, X, n2, n-n2), n-n2);
        return c0+c1;
    }

    if (n/4 < min_mvec) {
        c0 = vec_asum(__subvec(&x0, X, 0, n4),    n4);
        c1 = vec_asum(__subvec(&x0, X, n4, n2-n4),n2-n4);
        c2 = vec_asum(__subvec(&x0, X, n2, n4),   n4);
        c3 = vec_asum(__subvec(&x0, X, n2+n4, nn),nn);
        return c0 + c1 + c2 + c3;
    }

    c0 = vec_asum_recursive(__subvec(&x0, X, 0, n4),     n4, min_mvec);
    c1 = vec_asum_recursive(__subvec(&x0, X, n4, n2-n4), n2-n4, min_mvec);
    c2 = vec_asum_recursive(__subvec(&x0, X, n2, n4),    n4, min_mvec);
    c3 = vec_asum_recursive(__subvec(&x0, X, n2+n4, nn), nn, min_mvec);
    return c0 + c1 + c2 + c3;
}

/**
 * @brief Compute \f$ \sum_{i=0}^{len(x)-1} |x| \f$
 *
 * @retval Sum of absolute values of x elements
 */
ABSTYPE armas_x_asum(const armas_x_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (!armas_x_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return ZERO;
    }

    armas_env_t *env = armas_getenv();
    // this executed if extended precision not requested
    switch (conf->optflags & (ARMAS_ONAIVE|ARMAS_OKAHAN|ARMAS_ORECURSIVE)) {
    case ARMAS_OKAHAN:
        return vec_asum_kahan(x, armas_x_size(x));

    case ARMAS_ONAIVE:
        return vec_asum(x, armas_x_size(x));
    }
    return vec_asum_recursive(x, armas_x_size(x), env->blas1min);
}

#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

//! @}
