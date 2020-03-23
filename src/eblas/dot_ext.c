
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_dot_unsafe) && \
    defined(armas_x_ext_adot_unsafe) && \
    defined(armas_x_ext_adot_dx_unsafe) && \
    defined(armas_x_ext_adot)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

DTYPE armas_x_ext_dot_unsafe(
    const armas_x_dense_t *X,  const armas_x_dense_t *Y)
{
    register int i, kx, ky;
    DTYPE s0, s1, c0, c1, p0, p1, h0, h1, z0, z1;
    int xinc = X->rows == 1 ? X->step : 1;
    int yinc = Y->rows == 1 ? Y->step : 1;
    int N = armas_x_size(X);

    s0 = s1 = 0.0;
    c0 = c1 = 0.0;
    for (i = 0; i < N-1; i += 2) {
        twoprod(&h0, &p0, X->elems[(i+0)*xinc], Y->elems[(i+0)*yinc]);
        twoprod(&h1, &p1, X->elems[(i+1)*xinc], Y->elems[(i+1)*yinc]);
        twosum(&s0, &z0, s0, h0);
        c0 += z0 + p0;
        twosum(&s1, &z1, s1, h1);
        c1 += z1 + p1;
    }
    if (i == N)
        goto update;

    kx = i*xinc;
    ky = i*yinc;
    twoprod(&h0, &p0, X->elems[kx], Y->elems[ky]);
    twosum(&s0, &z0, s0, h0);
    c0 += z0 + p0;

 update:
    twosum(&s0, &z0, s0, s1);
    c0 += z0;
    return s0 + (c0 + c1);
}

/**
 * @brief Compute inner product scaled by constant in extended precision.
 *
 * @param h, l
 *    On entry initial values. On exit h = h + s, l = l + u where (s + u) = alpha*x^T*y
 * @param alpha
 *    Scaling constant
 * @param x, y
 *    Input vectors
 *
 */
void armas_x_ext_adot_unsafe(
    DTYPE *h,
    DTYPE *l,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y)
{
    register int i;
    DTYPE s0, c0, p0, h0, z0;
    int xinc = X->rows == 1 ? X->step : 1;
    int yinc = Y->rows == 1 ? Y->step : 1;
    int N = armas_x_size(X);

    if (alpha == ZERO) {
        return;
    }

    s0 = c0 = ZERO;
    for (i = 0; i < N; ++i) {
        twoprod(&h0, &p0, X->elems[(i+0)*xinc], Y->elems[(i+0)*yinc]);
        twosum(&s0, &z0, s0, h0);
        c0 += z0 + p0;
    }

    twoprod(&h0, &p0, s0, alpha);
    fastsum(&s0, &z0, *h, h0);
    c0 = *l + p0 + alpha*c0 + z0;
    // protect against  h < l
    fastsum(h, l, s0, c0);
}

/**
 * @brief Compute inner product scaled by constant in extended precision.
 *
 * @param h, l
 *    On entry initial values. On exit h = h + s, l = l + u where (s + u) = alpha*(x^T*y + dx^T*y)
 * @param alpha
 *    Scaling constant
 * @param x, dx, y
 *    Input vectors
 *
 */
void armas_x_ext_adot_dx_unsafe(
    DTYPE *h,
    DTYPE *l,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *dX,
    const armas_x_dense_t *Y)
{
    register int i;
    DTYPE s, c, z, p, r, q;
    int xinc =   X->rows == 1 ? X->step : 1;
    int dxinc = dX->rows == 1 ? dX->step : 1;
    int yinc =   Y->rows == 1 ? Y->step : 1;
    int N = armas_x_size(X);

    if (alpha == ZERO) {
        return;
    }

    s = c = ZERO;
    // (x + dx) * y == h + (p + dx*y); h + p = x * y;
    for (i = 0; i < N; ++i) {
        twoprod(&p, &q, X->elems[(i+0)*xinc], Y->elems[(i+0)*yinc]);
        r = dX->elems[i*dxinc]*Y->elems[i*yinc];
        twosum(&s, &z, s, p);
        c += z + q + r;
    }
    twoprod(&p, &q, s, alpha);
    twosum(&s, &z, *h, p);
    c = *l + q + alpha*c + z;
    // protect against  h < l
    fastsum(h, l, s, c);
}

/**
 * @brief Compute inner product scaled by constant in extended precision.
 *
 * @param result
 *    On exit resul in extended precision.
 * @param alpha
 *    Scaling constant
 * @param x, y
 *    Input vectors
 *
 * @return Zero on success, -1 on errors.
 */
int armas_x_ext_adot(
    DTYPE *result,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    armas_conf_t *cf)
{
    DTYPE s, u;
    if (!cf)
        cf = armas_conf_default();
    if (!armas_x_isvector(x) || !armas_x_isvector(y)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (armas_x_size(x) != armas_x_size(y)) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
    s = u = ZERO;
    armas_x_ext_adot_unsafe(&s, &u, alpha, x, y);
    *result = s + u;
    return 0;
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */
