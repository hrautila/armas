
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Triangular multiply

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mvupdate_trm)// && defined(__trmv_recursive)
#define ARMAS_PROVIDES 1
#endif
// this module requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "partition.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0
 *     0 |a11|a12  b1
 *     0 | 0 |a22  b2
 *
 *    b00 = a00*b0 + a01*b1 + a02*b2
 *    b10 =          a11*b1 + a12*b2
 *    b20 =                   a22*b2
 */
static
void trmv_lu(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    for (i = 0; i < A->cols; i++) {
        xk = armas_get_at_unsafe(x, i);
        armas_submatrix_unsafe(&a0, A, i, i+unit, 1, A->cols-i-unit);
        armas_subvector_unsafe(&x0, x, i+unit, A->cols-i-unit);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = unit ? xk + dot : dot;
        armas_set_at_unsafe(x, i, alpha*xk);
    }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0
 *  b1 =   0 |a11|a12  b'1
 *  b2     0 | 0 |a22  b'2
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static
void trmv_lut(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    for (i = A->cols-1; i >= 0; i--) {
        xk = armas_get_at(x, i);
        armas_submatrix_unsafe(&a0, A, 0, i, i-unit+1, 1);
        armas_subvector_unsafe(&x0, x, 0, i-unit+1);
        dot = armas_dot_unsafe(&x0, &a0);
        xk = unit ? xk + dot : dot;
        armas_set_at_unsafe(x, i, alpha*xk);
    }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = a00*b'0
 *  b1 = a10*b'0 + a11*b'1
 *  b2 = a20*b'0 + a21*b'1 + a22*b'2
 */
static
void trmv_ll(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    for (i = A->cols-1; i >= 0; i--) {
        xk = armas_get_at(x, i);
        armas_submatrix_unsafe(&a0, A, i, 0, 1, i-unit+1);
        armas_subvector_unsafe(&x0, x, 0, i-unit+1);
        dot = armas_dot_unsafe(&x0, &a0);
        xk = unit ? xk + dot : dot;
        armas_set_at_unsafe(x, i, alpha*xk);
    }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 */
static
void trmv_llt(
    armas_dense_t *x,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE xk, dot;
    armas_dense_t a0, x0;

    for (i = 0; i < A->cols; i++) {
        xk = armas_get_at_unsafe(x, i);
        armas_submatrix_unsafe(&a0, A, i+unit, i, A->cols-i-unit, 1);
        armas_subvector_unsafe(&x0, x, i+unit, A->cols-i-unit);
        dot = armas_dot_unsafe(&a0, &x0);
        xk = unit ? xk + dot : dot;
        armas_set_at_unsafe(x, i, alpha*xk);
    }
}

static
void trmv_unb(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        trmv_lut(X, alpha, A, flags);
        break;
    case ARMAS_UPPER:
        trmv_lu(X, alpha, A, flags);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        trmv_llt(X, alpha, A, flags);
        break;
    case ARMAS_LOWER:
    default:
        trmv_ll(X, alpha, A, flags);
        break;
    }
}

/*
 *     A00 | A01   x0
 *     ---------   --
 *     A10 | A11   x1
 *
 *  For UPPER                              LOWER-TRANS
 *  x0  = alpha*A00*x0 + alpha*A01*x1      x0 = alpha*A00^T*x1 + alpha*A10^T*x1
 *  x1  = alpha*A11*x1                     x1 = alpha*A11^T*x1
 */
static
void trmv_forward_recursive(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mvec_size)
{
    armas_dense_t ATL, ATR, ABL, ABR, xT, xB;

    if (A->cols < min_mvec_size) {
        trmv_unb(X, alpha, A, flags);
        return;
    }
    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, A->rows/2, ARMAS_PTOP);

    // top part
    trmv_forward_recursive(&xT, alpha, &ATL, flags, min_mvec_size);
    // update top with bottom
    if (flags & ARMAS_UPPER) {
        armas_mvmult_unsafe(ONE, &xT, alpha, &ATR, &xB, 0);
    } else {
        armas_mvmult_unsafe(ONE, &xT, alpha, &ABL, &xB, ARMAS_TRANS);
    }
    // bottom part
    trmv_forward_recursive(&xB, alpha, &ABR, flags, min_mvec_size);
}

/*
 *     A00 | A01   x0
 *     ---------   --
 *     A10 | A11   x1
 *
 *  For UPPER - TRANS                      LOWER
 *  x0  = alpha*A00^T*x0                   x0 = alpha*A00*x0
 *  x1  = alpha*A01^T*x0 + alpha*A11*x1    x1 = alpha*A10*x0 + alpha*A11*x1
 */
static
void trmv_backward_recursive(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mvec_size)
{
    armas_dense_t ATL, ATR, ABL, ABR, xT, xB;

    if (A->cols < min_mvec_size) {
        trmv_unb(X, alpha, A, flags);
        return;
    }
    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    vec_partition_2x1(
        &xT,
        &xB, /**/ X, A->rows/2, ARMAS_PTOP);

    // bottom part
    trmv_backward_recursive(&xB, alpha, &ABR, flags, min_mvec_size);
    // update bottom with top
    if (flags & ARMAS_UPPER) {
        armas_mvmult_unsafe(ONE, &xB, alpha, &ATR, &xT, ARMAS_TRANS);
    } else {
        armas_mvmult_unsafe(ONE, &xB, alpha, &ABL, &xT, 0);
    }
    // top part
    trmv_backward_recursive(&xT, alpha, &ATL, flags, min_mvec_size);
}

#if defined(armas_mvmult_trm_unsafe)
void armas_mvmult_trm_unsafe(
    armas_dense_t *X,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    armas_env_t *env = armas_getenv();

    if (A->cols < env->blas2min || env->blas2min == 0) {
        trmv_unb(X, alpha, A, flags);
        return;
    }

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_UPPER:
        trmv_forward_recursive(X, alpha, A, flags, env->blas2min);
        break;

    case ARMAS_UPPER|ARMAS_TRANS:
    case ARMAS_LOWER:
    default:
        trmv_backward_recursive(X, alpha, A, flags, env->blas2min);
        break;
    }
}
#endif

/**
 * @brief Triangular matrix-vector multiply
 *
 * Computes
 *    - \f$ X = alpha \times A X \f$
 *    - \f$ X = alpha \times A^T X  \f$   if *ARMAS_TRANS* set
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
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas
 */
int armas_mvmult_trm(
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

    // normal precision here
    if ((conf->optflags & ARMAS_ONAIVE) || env->blas2min == 0) {
        trmv_unb(x, alpha, A, flags);
    } else {
        switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
        case ARMAS_LOWER|ARMAS_TRANS:
        case ARMAS_UPPER:
            trmv_forward_recursive(x, alpha, A, flags, env->blas2min);
            break;
        case ARMAS_UPPER|ARMAS_TRANS:
        case ARMAS_LOWER:
        default:
            trmv_backward_recursive(x, alpha, A, flags, env->blas2min);
            break;
        }
    }
    return 0;
}
#else
#warning "No code; undefined requirements"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
