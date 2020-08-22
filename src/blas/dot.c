
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! dot product

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_dot) && defined(armas_x_adot)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

static
DTYPE vec_dot(const armas_x_dense_t *X,  const armas_x_dense_t *Y, int N)
{
    register int i, kx, ky;
    register DTYPE c0, c1, c2, c3, x0, x1, x2, x3;
    register int yinc = Y->rows == 1 ? Y->step : 1;
    register int xinc = X->rows == 1 ? X->step : 1;

    c0 = c1 = c2 = c3 = 0.0;
    for (i = 0; i < N-3; i += 4) {
        x0 = X->elems[(i+0)*xinc];
        x1 = X->elems[(i+1)*xinc];
        x2 = X->elems[(i+2)*xinc];
        x3 = X->elems[(i+3)*xinc];
        c0 += x0*Y->elems[(i+0)*yinc];
        c1 += x1*Y->elems[(i+1)*yinc];
        c2 += x2*Y->elems[(i+2)*yinc];
        c3 += x3*Y->elems[(i+3)*yinc];
    }
    if (i == N)
        goto update;

    kx = i*xinc;
    ky = i*yinc;
    switch (N-i) {
    case 3:
        c0 += X->elems[kx] * Y->elems[ky];
        kx += xinc; ky += yinc;
    case 2:
        c1 += X->elems[kx] * Y->elems[ky];
        kx += xinc; ky += yinc;
    case 1:
        c2 += X->elems[kx] * Y->elems[ky];
    }
update:
    return c0 + c1 + c2 + c3;
}

static inline
armas_x_dense_t *__subvec(armas_x_dense_t *x, const armas_x_dense_t *y, int K, int N)
{
    return armas_x_subvector_unsafe(x, y, K, N);
}

static
DTYPE vec_dot_recursive(const armas_x_dense_t *X, const armas_x_dense_t *Y, int n, int min_mvec_size)
{
    register DTYPE c0, c1, c2, c3;
    armas_x_dense_t x0, y0;
    int n2 = n/2;
    int n4 = n/4;
    int nn = n - n2 - n4;

    if (n < min_mvec_size)
        return vec_dot(X, Y, n);

    if (n/2 < min_mvec_size) {
        c0 = vec_dot(__subvec(&x0, X, 0, n2),     __subvec(&y0, Y, 0, n2),   n2);
        c1 = vec_dot(__subvec(&x0, X, n2, n-n2), __subvec(&y0, Y, n2, n-n2), n-n2);
        return c0+c1;
    }

    if (n/4 < min_mvec_size) {
        c0 = vec_dot(__subvec(&x0, X, 0, n4),      __subvec(&y0, Y, 0, n4),     n4);
        c1 = vec_dot(__subvec(&x0, X, n4, n2-n4 ), __subvec(&y0, Y, n4, n2-n4), n2-n4);
        c2 = vec_dot(__subvec(&x0, X, n2, n4),     __subvec(&y0, Y, n2, n4),    n4);
        c3 = vec_dot(__subvec(&x0, X, n2+n4, nn),  __subvec(&y0, Y, n2+n4, nn), nn);
        return c0 + c1 + c2 + c3;
    }

    c0 = vec_dot_recursive(__subvec(&x0, X, 0, n4),     __subvec(&y0, Y, 0, n4),     n4, min_mvec_size);
    c1 = vec_dot_recursive(__subvec(&x0, X, n4, n2-n4), __subvec(&y0, Y, n4, n2-n4), n2-n4, min_mvec_size);
    c2 = vec_dot_recursive(__subvec(&x0, X, n2, n4),    __subvec(&y0, Y, n2, n4),    n4, min_mvec_size);
    c3 = vec_dot_recursive(__subvec(&x0, X, n2+n4, nn), __subvec(&y0, Y, n2+n4, nn), nn, min_mvec_size);
    return c0 + c1 + c2 + c3;
}

void armas_x_adot_unsafe(DTYPE *value, DTYPE alpha, const armas_x_dense_t *x, const armas_x_dense_t *y)
{
    DTYPE dval;
    armas_env_t *env = armas_getenv();
    if (env->blas1min == 0) {
        dval = vec_dot(x, y, armas_x_size(y));
    } else {
        dval = vec_dot_recursive(y, x, armas_x_size(y), env->blas1min);
    }
    *value += dval * alpha;
}

DTYPE armas_x_dot_unsafe(const armas_x_dense_t *x, const armas_x_dense_t *y)
{
    DTYPE dval = ZERO;
    armas_x_adot_unsafe(&dval, ONE, x, y);
    return dval;
}

/**
 * @brief Updates value with inner product of two vectors scaled by constant, \f$ v = alpha*x^T*y \f$
 *
 * @param[in,out] value
 *    On entry initial value. On exit updated with the value of the inner
 *    product of x,y scaled  with constant alpha.
 * @param[in] alpha constant
 * @param[in] x vector
 * @param[in] y vector
 * @param[in] conf configuration block
 *
 * @return 0 for success, -1 for error.
 *
 * @ingroup blas1
 */
int armas_x_adot(DTYPE *value, DTYPE alpha, 
        const armas_x_dense_t *x, const armas_x_dense_t *y, armas_conf_t *conf)
{
    DTYPE dval;

    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (!armas_x_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (!armas_x_isvector(y)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (armas_x_size(x) != armas_x_size(y)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (!value) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    armas_env_t *env = armas_getenv();
    if (conf->optflags & ARMAS_ONAIVE || env->blas1min == 0) {
        dval = vec_dot(y, x, armas_x_size(y));
    } else {
        dval = vec_dot_recursive(y, x, armas_x_size(y), env->blas1min);
    }
    *value += alpha*dval;
    return 0;
}

/**
 * @brief Computes inner product of two vectors, \f$ v = x^T*y \f$
 *
 * @param[in] x vector
 * @param[in] y vector
 * @param[in] conf configuration block
 *
 * @ingroup blas1
 */
DTYPE armas_x_dot(const armas_x_dense_t *x, const armas_x_dense_t *y, armas_conf_t *conf)
{
    DTYPE dval = ZERO;
    if (armas_x_adot(&dval, ONE, x, y, conf) < 0) {
        return ZERO;
    }
    return dval;
}
#else
#warning "Not compiled; missing defines"
#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */
