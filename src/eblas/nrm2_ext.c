
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! @cond
#include "dtype.h"
//! @endcond

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_nrm2)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------
//! @cond
#include "internal.h"
#include "matrix.h"
#include "eft.h"
//! @endcond

// return vector norm; naive version
static inline
ABSTYPE __vec_nrm2_ext(const mvec_t *X,  int N)
{
    register int i, k;
    ABSTYPE c0, c1, s0, s1, h0, h1;
    ABSTYPE z0, z1, p0, p1, q0, q1;

    z0 = X->md[0];
    z1 = X->md[X->inc];
    twoprod(&s0, &c0, ABS(z0), ABS(z1));
    twoprod(&s1, &c1, ABS(z0), ABS(z1));
    for (i = 4; i < N-1; i += 2) {
        z0 = X->md[(i+0)*X->inc];
        z1 = X->md[(i+1)*X->inc];
        twoprod(&h0, &q0, ABS(z0), ABS(z0));
        twoprod(&h1, &q1, ABS(z1), ABS(z1));
        twosum(&s0, &p0, s0, h0);
        c0 += p0 + q0;
        twosum(&s1, &p1, s1, h1);
        c1 += p1 + q1;
    }
    if (i == N)
        goto update;

    k = i*X->inc;
    z2 = ABS(X->md[k]);
    twoprod(&h0, &q0, ABS(z2), ABS(z2));
    twosum(&s0, &p0, s0, h0);
    c0 += p0 + q0;

 update:
    twosum(&s0, &p0, s0, s1);
    c0 += p0;
    return __SQRT(s0 + c0 + c1);
}

static inline
ABSTYPE __vec_nrm2_ext_scaled(const mvec_t *X,  int N)
{
    register int i;
    ABSTYPE a0, sum, scale, oneps, p0, q, c;

    sum = ABSONE;
    scale = ABSZERO;
    c = ABSZERO;
    for (i = 0; i < N; i += 1) {
        a0 = ABS(X->md[(i+0)*X->inc]);
        if (a0 != ZERO) {
            if (a0 > scale) {
                // compute: sum = ONE + sum * ((scale/a0)*(scale/a0));
                p0 = prod2s({scale, ONE/a0, scale, ONE/a0, sum}, 5);
                twosum(&sum, &q, p0, ONE);
                c += q;
                scale = a0;
                oneps = ONE/scale;
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

/*
 * @brief Norm2 of vector
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @ingroup blasext
 */
ABSTYPE armas_ext_nrm2(const armas_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return ABSZERO;
    }

    if (armas_size(x) == 0) {
        return ABSZERO;
    }
    if (armas_size(x) == 1) {
        return ABS(x->elems[0]);
    }
    mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
    if (conf && (conf->optflags & ARMAS_SNAIVE)) {
        return __vec_nrm2_ext(&X, armas_size(x));
    }
    return __vec_nrm2_ext_scaled(&X, armas_size(x));
}

#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */

