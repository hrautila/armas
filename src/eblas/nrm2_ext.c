
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ex_nrm2) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

// return vector norm; naive version
static inline
ABSTYPE __vec_nrm2_ext(const mvec_t *X,  int N)
{
    register int i, k;
    ABSTYPE c0, c1, s0, s1, h0, h1;
    ABSTYPE z0, z1, p0, p1, q0, q1;

    z0 = X->md[0];
    z1 = X->md[X->inc];
    twoprod(&s0, &c0, __ABS(z0), __ABS(z1));
    twoprod(&s1, &c1, __ABS(z0), __ABS(z1));
    for (i = 4; i < N-1; i += 2) {
        z0 = X->md[(i+0)*X->inc];
        z1 = X->md[(i+1)*X->inc];
        twoprod(&h0, &q0, __ABS(z0), __ABS(z0));
        twoprod(&h1, &q1, __ABS(z1), __ABS(z1));
        twosum(&s0, &p0, s0, h0);
        c0 += p0 + q0;
        twosum(&s1, &p1, s1, h1);
        c1 += p1 + q1;
    }    
    if (i == N)
        goto update;

    k = i*X->inc;
    z2 = __ABS(X->md[k]);
    twoprod(&h0, &q0, __ABS(z2), __ABS(z2));
    twosum(&s0, &p0, s0, h0);
    c0 += p0 + q0;

 update:
    twosum(&s0, &p0, s0, s1);
    c0 += p0;
    return __SQRT(s0 + c0 + c1);
}

/*
 * Nick Higham in Accurrancy and Precision:
 *   For about half of all machine numbers x, value of x^2 either
 *   underflows or overflows
 *
 * Overflow is avoided by summing squares of scaled numbers and
 * then multiplying then with the scaling factor. Following is
 * is by Hammarling and included in BLAS reference libary.
 */

static inline
ABSTYPE __vec_nrm2_ext_scaled(const mvec_t *X,  int N)
{
    register int i;
    ABSTYPE a0, sum, scale, oneps, p0, q, c;

    sum = __ABSONE;
    scale = __ABSZERO;
    c = __ABSZERO;
    for (i = 0; i < N; i += 1) {
        a0 = __ABS(X->md[(i+0)*X->inc]);
        if (a0 != __ZERO) {
            if (a0 > scale) {
                // compute: sum = __ONE + sum * ((scale/a0)*(scale/a0));
                p0 = prod2s({scale, __ONE/a0, scale, __ONE/a0, sum}, 5);
                twosum(&sum, &q, p0, __ONE);
                c += q;
                scale = a0;
                oneps = __ONE/scale;
            } else {
                // compute: sum = sum + (a0/scale)*(a0/scale);
                p0 = prod2s({a0, oneps, a0, oneps}, 4);
                twosum(&sum, &q, p0, sum);
                c += q;
            }
        }
    }    
    return scale*__SQRT(sum + c);
}

/**
 * @brief Norm2 of vector
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @ingroup xblas1
 */
ABSTYPE armas_x_ex_nrm2(const armas_x_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return __ABSZERO;
    }

    if (armas_x_size(x) == 0) {
        return __ABSZERO;
    }
    if (armas_x_size(x) == 1) {
        return __ABS(x->elems[0]);
    }
    mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
    if (conf && (conf->optflags & ARMAS_SNAIVE)) {
        return __vec_nrm2_ext(&X, armas_x_size(x));
    }
    return __vec_nrm2_ext_scaled(&X, armas_x_size(x));
}

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

