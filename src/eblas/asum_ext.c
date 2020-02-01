
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_asum) && defined(armas_x_ext_sum_unsafe)
#define ARMAS_PROVIDES 1
#endif
// if extended precision enabled
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"


static
void ext_asum2s(DTYPE *h, DTYPE *l, const armas_x_dense_t *X, int N)
{
    register int i, k;
    ABSTYPE c0, c1, z0, z1;
    register ABSTYPE a0, a1;
    int xinc = X->rows == 1 ? X->step : 1;

    c0 = c1 = ZERO;
    a0 = a1 = ZERO;
    for (i = 0; i < N-1; i += 2) {
        twosum(&c0, &z0, c0, ABS(X->elems[(i+0)*xinc]));
        twosum(&c1, &z1, c1, ABS(X->elems[(i+1)*xinc]));
        a0 += z0; a1 += z1;
    }

    if (i != N) {
        k = i*xinc;
        twosum(&c0, &z0, c0, ABS(X->elems[k]));
        a0 += z0;
    }

    twosum(&c0, &z0, c0, c1);
    a0 += a1 + z0;
    *h = c0;
    *l = a0;
}

static
void ext_sum2s(DTYPE *h, DTYPE *l, const armas_x_dense_t *X, int N)
{
    register int i, k;
    DTYPE c0, c1, c2, c3, z0, z1, z2, z3;
    register DTYPE a0, a1, a2, a3;
    int xinc = X->rows == 1 ? X->step : 1;

    c0 = c1 = c2 = c3 = ZERO;
    a0 = a1 = a2 = a3 = ZERO;
    for (i = 0; i < N-3; i += 4) {
        twosum(&c0, &z0, c0, X->elems[(i+0)*xinc]);
        twosum(&c1, &z1, c1, X->elems[(i+1)*xinc]);
        twosum(&c2, &z2, c2, X->elems[(i+2)*xinc]);
        twosum(&c3, &z3, c3, X->elems[(i+3)*xinc]);
        a0 += z0; a1 += z1; a2 +=z2; a3 += z3;
    }
    if (i == N)
        goto update;

    k = i*xinc;
    switch (N-i) {
    case 3:
        twosum(&c0, &z0, c0, X->elems[k]);
        a0 += z0;
        k  += xinc;
    case 2:
        twosum(&c1, &z1, c1, X->elems[k]);
        a1 += z1;
        k  += xinc;
    case 1:
        twosum(&c2, &z2, c2, X->elems[k]);
        a2 += z2;
    }

 update:
    a0 += a1 + a2 + a3;
    twosum(&c0, &z1, c0, c1);
    a0 += z1;
    twosum(&c0, &z1, c0, c2);
    a0 += z1;
    twosum(&c0, &z1, c0, c3);
    a0 += z1;

    *h = c0;
    *l = a0;
}

/**
 * @brief Compute sum(x)
 *
 * @retval sum of x elements
 *
 * @ingroup blas1
 */
DTYPE armas_x_ext_sum_unsafe(const armas_x_dense_t *X)
{
    DTYPE h, l;
    ext_sum2s(&h, &l, X, armas_x_size(X));
    return h+l;
}

/**
 * @brief Compute alpha*sum(x) or alpha*sum(abs(x)) with extended internal precission
 *
 * @param result
 *    On exit, result of the computation.
 * @param alpha
 *    Constant scalar
 * @param X
 *    Vector
 * @param flags
 *    If ARMAS_ABS set the sum of absolute values is computed.
 *
 * @ingroup blas1ext
 */
int armas_x_ext_sum(DTYPE *result, DTYPE alpha, const armas_x_dense_t *X, int flags, armas_conf_t *cf)
{
    DTYPE h, l, q;
    if (!cf)
        cf = armas_conf_default();

    if (!armas_x_isvector(X)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (!result) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    if (flags & ARMAS_ABS) {
        ext_asum2s(&h, &l, X, armas_x_size(X));
    } else {
        ext_sum2s(&h, &l, X, armas_x_size(X));
    }

    if (alpha != ONE) {
        twoprod(result, &q, alpha, h);
        *result += q + alpha*l;
    } else {
        *result = h + l;
    }
    return 0;
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

