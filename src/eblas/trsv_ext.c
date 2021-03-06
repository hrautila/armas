
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_mvsolve_trm_unsafe) && defined(armas_ext_mvsolve_trm_w) && \
    defined(armas_ext_mvsolve_trm)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
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
static
void trsv_ext_unb_ll(
    armas_dense_t *X,
    armas_dense_t *dX,
    const armas_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_get_at_unsafe(X, 0);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_get_at_unsafe(dX, 0);
    if (!unit) {
        ak = armas_get_unsafe(Ac, 0, 0);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_set_at_unsafe(X,  0, s0);
    armas_set_at_unsafe(dX, 0, u0);

    for (i = 1; i < Ac->cols; ++i) {
        armas_subvector_unsafe(&x0, X, 0, i);
        armas_subvector_unsafe(&dx0, dX, 0, i);
        armas_submatrix_unsafe(&a0, Ac, i, 0, 1, i);

        xk = armas_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_ext_adot_dx_unsafe(&s0, &u0, -ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_set_at_unsafe(X,  i, s0);
            armas_set_at_unsafe(dX, i, u0);
            continue;
        }
        // if not unit diagonal then here
        ak = armas_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_set_at_unsafe(X,  i, s0);
        armas_set_at_unsafe(dX, i, u0);
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
static
void trsv_ext_unb_llt(
    armas_dense_t *X,
    armas_dense_t *dX,
    const armas_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_get_at_unsafe(X, Ac->cols-1);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_get_at_unsafe(dX, Ac->cols-1);
    if (!unit) {
        ak = armas_get_unsafe(Ac, Ac->cols - 1, Ac->cols - 1);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0 / ak + r0);
    }
    armas_set_at_unsafe(X, Ac->cols - 1, s0);
    armas_set_at_unsafe(dX, Ac->cols-1, u0);

    for (i = Ac->cols-2; i >= 0; --i) {
        armas_subvector_unsafe(&x0, X, i+1, Ac->cols-1-i);
        armas_subvector_unsafe(&dx0, dX, i+1, Ac->cols-1-i);
        armas_submatrix_unsafe(&a0, Ac, i+1, i, Ac->cols-1-i, i);

        xk = armas_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_ext_adot_dx_unsafe(&s0, &u0, -ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_set_at_unsafe(dX, i, u0);
            armas_set_at_unsafe(X, i, s0);
            continue;
        }

        ak = armas_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_set_at_unsafe(X,  i, s0);
        armas_set_at_unsafe(dX, i, u0);
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
static
void trsv_ext_unb_lu(
    armas_dense_t *X,
    armas_dense_t *dX,
    const armas_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, c0, ak, xk;

    s0 = armas_get_at_unsafe(X, Ac->cols-1);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_get_at_unsafe(dX, Ac->cols-1);
    if (!unit) {
        ak = armas_get_unsafe(Ac, Ac->cols-1, Ac->cols-1);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_set_at_unsafe(X,  Ac->cols-1, s0);
    armas_set_at_unsafe(dX, Ac->cols-1, u0);

    for (i = Ac->cols-2; i >= 0; --i) {
        p0 = r0 = ZERO;
        armas_subvector_unsafe(&x0, X, i+1, Ac->cols-1-i);
        armas_subvector_unsafe(&dx0, dX, i+1, Ac->cols-1-i);
        armas_submatrix_unsafe(&a0, Ac, i, i+1, 1, Ac->cols-1-i);

        xk = armas_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_ext_adot_dx_unsafe(&s0, &u0, -ONE, &x0, &dx0, &a0);

        if (unit) {
            fastsum(&s0, &c0, s0, u0);
            armas_set_at_unsafe(X,  i, s0);
            armas_set_at_unsafe(dX, i, u0);
            continue;
        }

        ak = armas_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_set_at_unsafe(X,  i, s0);
        armas_set_at_unsafe(dX, i, u0);
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
static
void trsv_ext_unb_lut(
    armas_dense_t *X,
    armas_dense_t *dX,
    const armas_dense_t *Ac,
    DTYPE alpha,
    int unit)
{
    register int i;
    armas_dense_t x0, dx0, a0;
    DTYPE s0, u0, p0, r0, ak, xk;

    s0 = armas_get_at_unsafe(X, 0);
    twoprod(&s0, &u0, s0, alpha);
    u0 = u0 + alpha * armas_get_at_unsafe(dX, 0);
    if (!unit) {
        ak = armas_get_unsafe(Ac, 0, 0);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
    }
    armas_set_at_unsafe(X,  0, s0);
    armas_set_at_unsafe(dX, 0, u0);

    for (i = 1; i < Ac->cols; ++i) {
        armas_subvector_unsafe(&x0, X, 0, i);
        armas_subvector_unsafe(&dx0, dX, 0, i);
        armas_submatrix_unsafe(&a0, Ac, 0, i, i, 1);

        xk = armas_get_at_unsafe(X, i);
        twoprod(&s0, &u0, xk, alpha);
        armas_ext_adot_dx_unsafe(&s0, &u0, -ONE, &x0, &dx0, &a0);

        if (unit) {
            armas_set_at_unsafe(X,  i, s0);
            armas_set_at_unsafe(dX, i, u0);
            continue;
        }

        ak = armas_get_unsafe(Ac, i, i);
        approx_twodiv(&p0, &r0, s0, ak);
        fastsum(&s0, &u0, p0, u0/ak + r0);
        armas_set_at_unsafe(X,  i, s0);
        armas_set_at_unsafe(dX, i, u0);
    }
}

int armas_ext_mvsolve_trm_unsafe(
    armas_dense_t *X,
    armas_dense_t *dX,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        trsv_ext_unb_lut(X, dX, A, alpha, unit);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        trsv_ext_unb_llt(X, dX, A, alpha, unit);
        break;
    case ARMAS_UPPER:
        trsv_ext_unb_lu(X, dX, A, alpha, unit);
        break;
    case ARMAS_LOWER:
    default:
        trsv_ext_unb_ll(X, dX, A, alpha, unit);
        break;
    }
    return 0;
}

/**
  * @brief Triangular matrix-vector solve in extended precision
 *
 * Computes
 *    - \f$ X = alpha \times A^{-1} X \f$
 *    - \f$ X = alpha \times A^{-T} X \f$  if *ARMAS_TRANS* set
 *
 * where A is upper (lower) triangular matrix defined with flag bits *ARMAS_UPPER*
 * (*ARMAS_LOWER*).
 *
 * @param[in,out] x Target and source vector
 * @param[in]     alpha Scalar multiplier
 * @param[in]     A Matrix
 * @param[in]     flags Operand flags
 * @param[in]     wb  Working space for intermediate results.
 * @param[in]     cf  Configuration block
 *
 * @retval  0 Success
 * @retval <0 Failed
 *
 * @ingroup blasext
 */
int armas_ext_mvsolve_trm_w(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    armas_wbuf_t *wb,
    armas_conf_t *cf)
{
    armas_dense_t dX;
    if (!cf)
        cf = armas_conf_default();

    int N = armas_size(x);
    if (!armas_isvector(x)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (A->cols != N || A->cols != A->rows) {
        cf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    if (wb && wb->bytes == 0) {
        wb->bytes = sizeof(DTYPE) * N;
        return 0;
    }
    if (armas_wbytes(wb) < sizeof(DTYPE)*N) {
        cf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    armas_make(&dX, A->cols, 1, A->cols, (DTYPE *)armas_wptr(wb));
    armas_ext_mvsolve_trm_unsafe(x, &dX, alpha, A, flags);
    return 0;
}

/**
 * @brief Triangular matrix-vector solve in extended precision
 *
 * Convenience function to call solver without explicit workspace.
 *
 * @ingroup blasext
 */
int armas_ext_mvsolve_trm(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    armas_conf_t *cf)
{
    int err = 0;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!armas_walloc(&wb, A->cols*sizeof(DTYPE))) {
        cf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    err = armas_ext_mvsolve_trm_w(x, alpha, A, flags, &wb, cf);
    armas_wrelease(&wb);
    return err;
}

#else
#warning "Missing defines. No code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
