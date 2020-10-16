
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_sumk)
#define ARMAS_PROVIDES 1
#endif
// if extended precision enabled
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

static
void ext_sumk(DTYPE *res, const armas_dense_t *x, int K)
{
    int i, k, N = armas_size(x);
    DTYPE q[8], s, alpha;

    for (i = 0; i < K-1; ++i) {
        s = armas_get_at_unsafe(x, i);
        for (k = 0; k < i; ++k) {
            twosum(&q[k], &s, q[k], s);
        }
        q[i] = s;
    }
    for (i = K-1; i < N; ++i) {
        alpha = armas_get_at_unsafe(x, i);
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
 * @ingroup blasext
 */
DTYPE armas_ext_sumk_unsafe(const armas_dense_t *X, int K)
{
    DTYPE res;
    ext_sum(&res, X, K);
    return res;
}

/*
 * @brief Compute sum(x) with extended internal precission
 *
 * @param result
 *    On exit, result of the computation.
 * @param X
 *    Vector
 * @param K
 *    Precission level
 * @param cf
  *
 * @ingroup blasext
 */
int armas_ext_sumk(DTYPE *result, const armas_dense_t *X, int K, armas_conf_t *cf)
{
    DTYPE h, l, q;
    if (!cf)
        cf = armas_conf_default();

    if (!armas_isvector(X)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (!result) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    ext_sum(result, X, K);
    return 0;
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

