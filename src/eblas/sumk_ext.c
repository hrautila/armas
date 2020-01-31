
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_sumk)
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
void ext_sumk(DTYPE *res, const armas_x_dense_t *x, int K)
{
    int i, k, N = armas_size(x);
    DTYPE q[8], s, alpha;

    for (i = 0; i < K-1; ++i) {
        s = armas_x_get_at_unsafe(x, i);
        for (k = 0; k < i; ++k) {
            twosum(&q[k], &s, q[k], s);
        }
        q[i] = s;
    }
    for (i = K-1; i < N; ++i) {
        alpha = armas_x_get_at_unsafe(x, i);
        for (k = 0; k < K-1; ++k) {
            twosum(&q[k], &alpha, q[k], alpha);
        }
        s += alpha;
    }
    for (i = 0; i < K-2; ++i) {
        alpha = q[i];
        for (k = i+1; k < K-1; ++k) {
            twosum(&q[k], &alpha, q[k], alpha);
        }
        s += alpha;
    }
    *res = s + q[K-1];
}

/**
 * @brief Compute sum(x)
 *
 * @retval sum of x elements
 *
 * @ingroup blas1
 */
DTYPE armas_x_ext_sumk_unsafe(const armas_x_dense_t *X, int K)
{
    DTYPE h, l;
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
int armas_x_ext_sumk(DTYPE *result, const armas_x_dense_t *X, int K, armas_conf_t *cf)
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

    return 0;
}

#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

