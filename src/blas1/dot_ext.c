
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__vec_dot_ext) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

DTYPE __vec_dot_ext(const mvec_t *X,  const mvec_t *Y, int N)
{
    register int i, kx, ky;
    DTYPE s0, s1, c0, c1, p0, p1, h0, h1, z0, z1;

    s0 = s1 = 0.0;
    c0 = c1 = 0.0;
    for (i = 0; i < N-1; i += 2) {
        twoprod(&h0, &p0, X->md[(i+0)*X->inc], Y->md[(i+0)*Y->inc]);
        twoprod(&h1, &p1, X->md[(i+1)*X->inc], Y->md[(i+1)*Y->inc]);
        twosum(&s0, &z0, s0, h0);
        c0 += z0 + p0;
        twosum(&s1, &z1, s1, h1);
        c1 += z1 + p1;
    }    
    if (i == N)
        goto update;

    kx = i*X->inc;
    ky = i*Y->inc;
    twoprod(&h0, &p0, X->md[kx], Y->md[ky]);
    twosum(&s0, &z0, s0, h0);
    c0 += z0 + p0;

 update:
    twosum(&s0, &z0, s0, s1);
    c0 += z0;
    return s0 + (c0 + c1);
}


#if 0
/**
 * @brief Computes inner product of two vectors.
 *
 * @param[in] x vector
 * @param[in] y vector
 * @param[in] conf configuration block
 *
 * @ingroup blas1
 */
DTYPE __armas_ex_dot(const __armas_dense_t *x, const __armas_dense_t *y, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        if (conf) conf->error = ARMAS_ENEED_VECTOR;
        return __ZERO;
    }
    if (y->cols != 1 && y->rows != 1) {
        if (conf) conf->error = ARMAS_ENEED_VECTOR;
        return __ZERO;
    }
    if (__armas_size(x) != __armas_size(y)) {
        if (conf) conf->error = ARMAS_ESIZE;
        return __ZERO;
    }

    const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
    const mvec_t Y = {y->elems, (y->rows == 1 ? y->step : 1)};

    return __vec_dot_ext(&Y, &X, __armas_size(y));
}
#endif

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

