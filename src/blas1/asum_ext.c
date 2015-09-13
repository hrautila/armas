
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__vec_asum_ext) && defined(__vec_sum_ext)
#define __ARMAS_PROVIDES 1
#endif
// if extended precision enabled
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"


ABSTYPE __vec_asum_ext(const mvec_t *X,  int N)
{
    register int i, k;
    ABSTYPE c0, c1, z0, z1; 
    register ABSTYPE a0, a1;

    c0 = __ABS(X->md[0]);
    c1 = __ABS(X->md[1*X->inc]);
    a0 = a1 = __ZERO;
    for (i = 2; i < N-1; i += 2) {
        twosum(&c0, &z0, c0, __ABS(X->md[(i+0)*X->inc]));
        twosum(&c1, &z1, c1, __ABS(X->md[(i+1)*X->inc]));
        a0 += z0; a1 += z1; 
    }    

    if (i != N) {
        k = i*X->inc;
        twosum(&c0, &z0, c0, __ABS(X->md[k]));
        a0 += z0;
    }

    twosum(&c0, &z0, c0, c1);
    a0 += a1 + z0;
    return c0 + a0;
}

// -----------------------------------------------------------------------------------
DTYPE __vec_sum_ext(const mvec_t *X,  int N)
{
    register int i, k;
    DTYPE c0, c1, c2, c3, z0, z1, z2, z3;
    register DTYPE a0, a1, a2, a3;

    c0 = X->md[0];
    c1 = X->md[X->inc];
    c2 = X->md[2*X->inc];
    c3 = X->md[3*X->inc];
    a0 = a1 = a2 = a3 = 0.0;
    for (i = 4; i < N-3; i += 4) {
        twosum(&c0, &z0, c0, X->md[(i+0)*X->inc]);
        twosum(&c1, &z1, c1, X->md[(i+1)*X->inc]);
        twosum(&c2, &z2, c2, X->md[(i+2)*X->inc]);
        twosum(&c3, &z3, c3, X->md[(i+3)*X->inc]);
        a0 += z0; a1 += z1; a2 +=z2; a3 += z3;
    }    
    if (i == N)
        goto update;
  
    k = i*X->inc;
    switch (N-i) {
    case 3:
        twosum(&c0, &z0, c0, X->md[k]);
        a0 += z0;
        k  += X->inc;
    case 2:
        twosum(&c1, &z1, c1, X->md[k]);
        a1 += z1;
        k  += X->inc;
    case 1:
        twosum(&c2, &z2, c2, X->md[k]);
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
    return c0 + a0;
}


#if 0
static inline
DTYPE __vec_asum_ext0(const mvec_t *X,  int N)
{
    register int i;
    DTYPE c0, z0;
    register DTYPE a0;

    c0 = __ABS(X->md[0]);
    a0 = 0.0;
    for (i = 1; i < N; i++) {
        twosum(&c0, &z0, c0, __ABS(X->md[(i+0)*X->inc]));
        a0 += z0; 
    }    
    return c0 + a0;
}

static inline
DTYPE __vec_sum_ext0(const mvec_t *X,  int N)
{
    register int i, k;
    DTYPE c0, c1, c2, c3, z0, z1, z2, z3;
    register DTYPE a0, a1, a2, a3;

    c0 = X->md[0];
    a0 = 0.0;
    for (i = 1; i < N; i++) {
        twosum(&c0, &z0, c0, X->md[(i+0)*X->inc]);
        a0 += z0; 
    }    
    return c0 + a0;
}


/**
 * @brief Compute sum(abs(x)) with extended internal precission
 *
 * @retval sum of absolute values of x elements
 *
 * @ingroup blas1ext
 */
ABSTYPE __armas_ex_asum(const __armas_dense_t *x, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return __ZERO;
    }

    const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};

    return __vec_asum_ext(&X, __armas_size(x));
}


/**
 * @brief Compute sum(x)
 *
 * @retval sum of x elements
 *
 * @ingroup blas1
 */
DTYPE __armas_ex_sum(const __armas_dense_t *x, armas_conf_t *conf)
{
    DTYPE s;
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return __ZERO;
    }

    const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  
    return __vec_sum_ext(&X, __armas_size(x));
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

