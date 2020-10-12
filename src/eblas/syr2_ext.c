
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mvupdate2_sym_unsafe) && defined(armas_x_ext_mvupdate2_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

// A = A + alpha*x.T*y + alpha*y.T*x
int armas_x_ext_mvupdate2_sym_unsafe(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    DTYPE p0, p1, r, s, c0, c1, q0, q1, xk, yk;
    register int i, j;

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        for (i = 0; i < A->rows; ++i) {
            for (j = i; j < A->cols; ++j) {
                xk = armas_x_get_at_unsafe(X, i);
                yk = armas_x_get_at_unsafe(Y, j);
                // p + r = X[i]*Y[j]
                twoprod(&p0, &r, xk, yk);
                // p + c = alpha*p
                twoprod(&p0, &c0, alpha, p0);
                c0 += alpha*r;

                xk = armas_x_get_at_unsafe(X, j);
                yk = armas_x_get_at_unsafe(Y, i);
                // p + r = X[j]*Y[i]
                twoprod(&p1, &r, xk, yk);
                // p + c = alpha*p
                twoprod(&p1, &c1, alpha, p1);
                c0 += alpha*r;
                // p0 + q0 = alpha*x[i]*y[j] + alpha*y[i]*x[j]
                twosum(&p0, &q0, p0, p1);
                q0 += c0;

                // p1 + q1 = beta*A[i,j]
                twoprod(&p1, &q1, beta, armas_x_get_unsafe(A, i, j));
                twosum(&s, &c0, p1, p0);
                armas_x_set_unsafe(A, i, j,  s + (c0 + q0 + q1));
            }
        }
        break;
    case ARMAS_LOWER:
    default:
        for (j = 0; j < A->cols; ++j) {
            for (i = 0; i < j+1; ++i) {
                xk = armas_x_get_at_unsafe(X, i);
                yk = armas_x_get_at_unsafe(Y, j);
                // p + r = X[i]*Y[j]
                twoprod(&p0, &r, xk, yk);
                // p + c = alpha*p
                twoprod(&p0, &c0, alpha, p0);
                c0 += alpha*r;

                xk = armas_x_get_at_unsafe(X, j);
                yk = armas_x_get_at_unsafe(Y, i);
                // p + r = X[j]*Y[i]
                twoprod(&p1, &r, xk, yk);
                // p + c = alpha*p
                twoprod(&p1, &c1, alpha, p1);
                c0 += alpha*r;
                // p0 + q0 = alpha*x[i]*y[j] + alpha*y[i]*x[j]
                twosum(&p0, &q0, p0, p1);
                q0 += c0;

                // p1 + q1 = beta*A[i,j]
                twoprod(&p1, &q1, beta, armas_x_get_unsafe(A, i, j));
                twosum(&s, &c0, p1, p0);
                armas_x_set_unsafe(A, i, j,  s + (c0 + q0 + q1));
            }
        }
        break;
    }
    return 0;
}


/**
 * @brief Symmetric matrix rank-2 update in extended precision.
 *
 * Computes
 *    -\f$ A = A + alpha \times X Y^T + alpha \times Y X^T \f$
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag *ARMAS_LOWER* (*ARMAR_UPPER*) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * @param[in]      beta scalar
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      x, y source vector
 * @param[in]      flags flag bits
 * @param[in]      conf configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blasext
 */
int armas_x_ext_mvupdate2_sym(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    const armas_x_dense_t *y,
    int flags,
    armas_conf_t *conf)
{
    int nx = armas_x_size(x);
    int ny = armas_x_size(y);

    if (armas_x_size(A) == 0 || nx == 0 || ny == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    if (!armas_x_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (!armas_x_isvector(y)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (A->cols != A->rows || ny != nx) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    armas_x_ext_mvupdate2_sym_unsafe(beta, A, alpha, x, y, flags);
    return 0;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
