
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mvsolve_trm_unsafe) && defined(armas_x_ext_mvsolve_trm_w) && \
    defined(armas_x_ext_mvsolve_trm)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 =  alpha*b'0/a00
 *  b1 = (alpha*b'1 - a10*b0)/a11
 *  b2 = (alpha*b'2 - a20*b0 - a21*b1)/a22
 */
static inline
void __trsv_ext_unb_ll(
    armas_x_dense_t *X,
    armas_x_dense_t *dX,
    const armas_x_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_x_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_x_get_at_unsafe(X, 0);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_x_get_at_unsafe(dX, 0);
    if (!unit) {
        ak = armas_x_get_unsafe(Ac, 0, 0);
        approx_twodiv(&s0, &r0, s0, ak);
        fastsum(&s0, &u0, s0, u0/ak + r0);
    }
    armas_x_set_at_unsafe(X,  0, s0);
    armas_x_set_at_unsafe(dX, 0, u0);

    for (i = 1; i < Ac->cols; ++i) {
        armas_x_subvector_unsafe(&x0, X, 0, i);
        armas_x_subvector_unsafe(&dx0, dX, 0, i);
        armas_x_submatrix_unsafe(&a0, Ac, i, 0, 1, i);

        xk = armas_x_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_x_ext_adot_dx_unsafe(&s0, &u0, -__ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_x_set_at_unsafe(X,  i, s0);
            armas_x_set_at_unsafe(dX, i, u0);
            continue;
        }
        // if not unit diagonal then here
        ak = armas_x_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_x_set_at_unsafe(X,  i, s0);
        armas_x_set_at_unsafe(dX, i, u0);
    }
}


/*
 * LEFT-LOWER-TRANSPOSE
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = (alpha*b'0 - a10*b1 - a20*b2)/a00
 *  b1 =          (alpha*b'1 - a21*b2)/a11
 *  b2 =                     alpha*b'2/a22
 */
static inline
void __trsv_ext_unb_llt(
    armas_x_dense_t *X,
    armas_x_dense_t *dX,
    const armas_x_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_x_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_x_get_at_unsafe(X, Ac->cols-1);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_x_get_at_unsafe(dX, Ac->cols-1);
    if (!unit) {
      ak = armas_x_get_unsafe(Ac, Ac->cols-1, Ac->cols-1);
      approx_twodiv(&p0, &r0, s0, ak);
      fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_x_set_at_unsafe(X,  Ac->cols-1, s0);
    armas_x_set_at_unsafe(dX, Ac->cols-1, u0);

    for (i = Ac->cols-2; i >= 0; --i) {
        armas_x_subvector_unsafe(&x0, X, i+1, Ac->cols-1-i);
        armas_x_subvector_unsafe(&dx0, dX, i+1, Ac->cols-1-i);
        armas_x_submatrix_unsafe(&a0, Ac, i+1, i, Ac->cols-1-i, i);

        xk = armas_x_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_x_ext_adot_dx_unsafe(&s0, &u0, -__ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_x_set_at_unsafe(dX, i, u0);
            armas_x_set_at_unsafe(X, i, s0);
            continue;
        }

        ak = armas_x_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_x_set_at_unsafe(X,  i, s0);
        armas_x_set_at_unsafe(dX, i, u0);
    }
}

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = (alpha*b'0 - a01*b1 - a02*b2)/a00
 *    b1 =          (alpha*b'1 - a12*b2)/a11
 *    b2 =                     alpha*b'2/a22
 */
static inline
void __trsv_ext_unb_lu(
    armas_x_dense_t *X,
    armas_x_dense_t *dX,
    const armas_x_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_x_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, c0, ak, xk;

    s0 = armas_x_get_at_unsafe(X, Ac->cols-1);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_x_get_at_unsafe(dX, Ac->cols-1);
    if (!unit) {
        ak = armas_x_get_unsafe(Ac, Ac->cols-1, Ac->cols-1);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_x_set_at_unsafe(X,  Ac->cols-1, s0);
    armas_x_set_at_unsafe(dX, Ac->cols-1, u0);

    for (i = Ac->cols-2; i >= 0; --i) {
        p0 = r0 = __ZERO;
        armas_x_subvector_unsafe(&x0, X, i+1, Ac->cols-1-i);
        armas_x_subvector_unsafe(&dx0, dX, i+1, Ac->cols-1-i);
        armas_x_submatrix_unsafe(&a0, Ac, i, i+1, 1, Ac->cols-1-i);

        xk = armas_x_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_x_ext_adot_dx_unsafe(&s0, &u0, -__ONE, &x0, &dx0, &a0);

        if (unit) {
            fastsum(&s0, &c0, s0, u0);
            armas_x_set_at_unsafe(X,  i, s0);
            armas_x_set_at_unsafe(dX, i, u0);
            continue;
        }

        ak = armas_x_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_x_set_at_unsafe(X,  i, s0);
        armas_x_set_at_unsafe(dX, i, u0);
    }
}

/*
 * LEFT-UPPER-TRANSPOSE
 *
 *  b0    a00|a01|a02  b'0
 *  b1 =   0 |a11|a12  b'1
 *  b2     0 | 0 |a22  b'2
 *
 *  b0 =  alpha*b'0/a00
 *  b1 = (alpha*b'1 - a01*b0)/a11
 *  b2 = (alpha*b'2 - a02*b0 - a12*b1)/a22
 */
static inline
void __trsv_ext_unb_lut(
    armas_x_dense_t *X,
    armas_x_dense_t *dX,
    const armas_x_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_x_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_x_get_at_unsafe(X, 0);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_x_get_at_unsafe(dX, 0);
    if (!unit) {
        ak = armas_x_get_unsafe(Ac, 0, 0);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_x_set_at_unsafe(X,  0, s0);
    armas_x_set_at_unsafe(dX, 0, u0);

    for (i = 1; i < Ac->cols; ++i) {
        armas_x_subvector_unsafe(&x0, X, 0, i);
        armas_x_subvector_unsafe(&dx0, dX, 0, i);
        armas_x_submatrix_unsafe(&a0, Ac, 0, i, i, 1);

        xk = armas_x_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_x_ext_adot_dx_unsafe(&s0, &u0, -__ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_x_set_at_unsafe(X,  i, s0);
            armas_x_set_at_unsafe(dX, i, u0);
            continue;
        }

        ak = armas_x_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_x_set_at_unsafe(X,  i, s0);
        armas_x_set_at_unsafe(dX, i, u0);
    }
}

int armas_x_ext_mvsolve_trm_unsafe(
    armas_x_dense_t *X,
    armas_x_dense_t *dX,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        __trsv_ext_unb_lut(X, dX, A, alpha, unit);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        __trsv_ext_unb_llt(X, dX, A, alpha, unit);
        break;
    case ARMAS_UPPER:
        __trsv_ext_unb_lu(X, dX, A, alpha, unit);
        break;
    case ARMAS_LOWER:
    default:
        __trsv_ext_unb_ll(X, dX, A, alpha, unit);
        break;
    }
    return 0;
}

int armas_x_ext_mvsolve_trm_w(
    armas_x_dense_t *x,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_wbuf_t *wb,
    armas_conf_t *cf)
{
    armas_x_dense_t dX;
    if (!cf)
        cf = armas_conf_default();

    int N = armas_x_size(x);
    if (!armas_x_isvector(x)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (A->cols != N || A->cols != A->rows) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
    if (wb && wb->bytes == 0) {
        wb->bytes = sizeof(DTYPE) * N;
        return 0;
    }
    if (armas_wbytes(wb) < sizeof(DTYPE)*N) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }
    armas_x_make(&dX, A->cols, 1, A->cols, (DTYPE *)armas_wptr(wb));
    armas_x_ext_mvsolve_trm_unsafe(x, &dX, alpha, A, flags);
    return 0;
}

int armas_x_ext_mvsolve_trm(
    armas_x_dense_t *x,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_conf_t *cf)
{
    int err = 0;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!armas_walloc(&wb, A->cols*sizeof(DTYPE))) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }
    err = armas_x_ext_mvsolve_trm_w(x, alpha, A, flags, &wb, cf);
    armas_wrelease(&wb);
    return err;
}

#else
#warning "Missing defines. No code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
