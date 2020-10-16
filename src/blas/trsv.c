
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Triangular solve

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mvsolve_trm) && defined(armas_mvsolve_trm_unsafe)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(armas_mvmult_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "partition.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = (b'0 - a01*b1 - a02*b2)/a00
 *    b1 =          (b'1 - a12*b2)/a11
 *    b2 =                     b'2/a22
 */
static
void trsv_lu(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    xk = alpha * armas_get_at_unsafe(x, A->cols-1);
    if (!unit) {
        xk /= armas_get_unsafe(A, A->cols-1, A->cols-1);
    }
    armas_set_at_unsafe(x, A->cols-1, xk);

    for (i = A->cols-2; i >= 0; --i) {
        xk = armas_get_at_unsafe(x, i);
        armas_subvector_unsafe(&x0, x, i+1, A->cols-1-i);
        armas_submatrix_unsafe(&a0, A, i, i+1, 1, A->cols-1-i);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = (alpha*xk - dot);
        if (!unit) {
            xk /= armas_get_unsafe(A, i, i);
        }
        armas_set_at_unsafe(x, i, xk);
    }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0
 *  b1 =   0 |a11|a12  b'1
 *  b2     0 | 0 |a22  b'2
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a01*b0)/a11
 *  b2 = (b'2 - a02*b0 - a12*b1)/a22
 */
static
void trsv_lut(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    xk = alpha * armas_get_at_unsafe(x, 0);
    if (!unit)
        xk /= armas_get_unsafe(A, 0, 0);
    armas_set_at_unsafe(x, 0, xk);

    for (i = 1; i < A->cols; ++i) {
        xk = armas_get_at_unsafe(x, i);
        armas_subvector_unsafe(&x0, x, 0, i);
        armas_submatrix_unsafe(&a0, A, 0, i, i, 1);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = (alpha*xk - dot);
        if (!unit) {
            xk /= armas_get_unsafe(A, i, i);
        }
        armas_set_at_unsafe(x, i, xk);
    }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a10*b0)/a11
 *  b2 = (b'2 - a20*b0 - a21*b1)/a22
 */
static
void trsv_ll(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    xk = alpha * armas_get_at_unsafe(x, 0);
    if (!unit)
        xk /= armas_get_unsafe(A, 0, 0);
    armas_set_at_unsafe(x, 0, xk);

    for (i = 1; i < A->cols; ++i) {
        xk = armas_get_at_unsafe(x, i);
        armas_subvector_unsafe(&x0, x, 0, i);
        armas_submatrix_unsafe(&a0, A, i, 0, 1, i);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = (alpha*xk - dot);
        if (!unit) {
            xk /= armas_get_unsafe(A, i, i);
        }
        armas_set_at_unsafe(x, i, xk);
    }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = (b'0 - a10*b1 - a20*b2)/a00
 *  b1 =          (b'1 - a21*b2)/a11
 *  b2 =                     b'2/a22
 */
static
void trsv_llt(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    xk = alpha * armas_get_at_unsafe(x, A->cols-1);
    if (!unit) {
        xk /= armas_get_unsafe(A, A->cols-1, A->cols-1);
    }
    armas_set_at_unsafe(x, A->cols-1, xk);

    for (i = A->cols-2; i >= 0; --i) {
        xk = armas_get_at_unsafe(x, i);
        armas_subvector_unsafe(&x0, x, i+1, A->cols-1-i);
        armas_submatrix_unsafe(&a0, A, i+1, i, A->cols-1-i, 1);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = (alpha*xk - dot);
        if (!unit) {
            xk /= armas_get_unsafe(A, i, i);
        }
        armas_set_at_unsafe(x, i, xk);
    }
}

static
void trsv_unb(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        trsv_lut(X, alpha, A, flags);
        break;
    case ARMAS_UPPER:
        trsv_lu(X, alpha, A, flags);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        trsv_llt(X, alpha, A, flags);
        break;
    case ARMAS_LOWER:
    default:
        trsv_ll(X, alpha, A, flags);
        break;
    }
}

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0           --> x0 = alpha*trsv(x'0, A00)
 *    x'1 = A01*x0 + A11*x1  --> x1 = trsv(alpha*x'1 - A01*x0)
 *  lower:
 *    x'0 = A00*x0           --> x0 = alpha8trsv(x'0, A00)
 *    x'1 = A10*x0 + A11*x1  --> x1 = trsv(alpha*x'1 - A10*x0, A11)
 *
 *   Forward substitution.
 */
static
void trsv_forward_recursive(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mvec_size)
{
    armas_dense_t ATL, ATR, ABL, ABR, xT, xB;

    if (A->cols < min_mvec_size) {
        trsv_unb(X, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, A->rows/2, ARMAS_PTOP);

    // top part
    trsv_forward_recursive(&xT, alpha, &ATL, flags, min_mvec_size);
    // update bottom with top
    if (flags & ARMAS_UPPER) {
        armas_mvmult_unsafe(alpha, &xB, -ONE, &ATR, &xT, ARMAS_TRANS);
    } else {
        armas_mvmult_unsafe(alpha, &xB, -ONE, &ABL, &xT, 0);
    }
    // bottom part
    trsv_forward_recursive(&xB, ONE, &ABR, flags, min_mvec_size);
}

/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0 + A01*x1  --> x0 = trsv(alpha*x'0 - A01*x1, A00)
 *    x'1 = A11*x1           --> x1 = alpha*trsv(x'1, A11)
 *  lower:
 *    x'0 = A00*x0 + A10*x1  --> x0 = trsv(alpha*x'0 - A10*x1, A00)
 *    x'1 = A11*x1           --> x1 = alpha*trsv(x'1, A11)
 *
 *   Backward substitution.
 */
static
void trsv_backward_recursive(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mvec_size)
{
    armas_dense_t ATL, ATR, ABL, ABR, xT, xB;

    if (A->cols < min_mvec_size) {
        trsv_unb(X, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, A->rows/2, ARMAS_PTOP);

    // bottom part
    trsv_backward_recursive(&xB, alpha, &ABR, flags, min_mvec_size);
    // update top with bottom
    if (flags & ARMAS_UPPER) {
        armas_mvmult_unsafe(alpha, &xT, -ONE, &ATR, &xB, 0);
    } else {
        armas_mvmult_unsafe(alpha, &xT, -ONE, &ABL, &xB, ARMAS_TRANS);
    }
    // top part
    trsv_backward_recursive(&xT, ONE, &ATL, flags, min_mvec_size);
}

void armas_mvsolve_trm_unsafe(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    armas_env_t *env = armas_getenv();

    if (A->cols < env->blas2min || env->blas2min == 0) {
        trsv_unb(X, alpha, A, flags);
        return;
    }

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_UPPER:
        trsv_backward_recursive(X, alpha, A, flags, env->blas2min);
        break;

    case ARMAS_UPPER|ARMAS_TRANS:
    case ARMAS_LOWER:
    default:
        trsv_forward_recursive(X, alpha, A, flags, env->blas2min);
        break;
    }
}


/**
 * @brief Triangular matrix-vector solve
 *
 * Computes
 *    - \f$ X = alpha \times A^{-1} X \f$
 *    - \f$ X = alpha \times A^{-T} X \f$  if *ARMAS_TRANS* set
 *
 * where A is upper (lower) triangular matrix defined with flag bits *ARMAS_UPPER*
 * (*ARMAS_LOWER*).
 *
 * @param[in,out] x target and source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     A matrix
 * @param[in]     flags operand flags
 * @param[in]     conf  configuration block
 *
 * @retval  0 Success
 * @retval <0 Failed
 *
 * @ingroup blas
 */
int armas_mvsolve_trm(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    armas_conf_t *conf)
{
    int nx = armas_size(x);

    if (armas_size(A) == 0 || nx == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (A->cols != nx || A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    armas_env_t *env = armas_getenv();
    if ((conf->optflags & ARMAS_ONAIVE) || env->blas2min == 0) {
        trsv_unb(x, alpha, A, flags);
        return 0;
    }

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_UPPER:
        trsv_backward_recursive(x, alpha, A, flags, env->blas2min);
        break;
    case ARMAS_UPPER|ARMAS_TRANS:
    case ARMAS_LOWER:
    default:
        trsv_forward_recursive(x, alpha, A, flags, env->blas2min);
        break;
    }
    return 0;
}
#else
#warning "Missing defines; No code;"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
