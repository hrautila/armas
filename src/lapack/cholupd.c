
// Copyright (c) Harri Rautila, 2016-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Rank update Cholesky factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_cholupdate)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas) && defined(armas_x_gvrot_vec)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#include "sym.h"
//! \endcond

#define MAX_ITERS 5

/*
 * Update of Cholesky factorization.
 *
 *   A + x*x^T = (L, x) Q (L^T)
 *                        (x^T)
 */
static
int unblk_cholupdate_lower(armas_x_dense_t * A, armas_x_dense_t * X,
                           armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a11, a21, A22;
    armas_x_dense_t XL, XR, X0, x1, X2;
    DTYPE c, s, r, a11val, x1val;

    EMPTY(x1);
    EMPTY(a11);
    EMPTY(A00);
    EMPTY(XL);
    EMPTY(X0);

    mat_partition_2x2(
        &ATL, __nil, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &XL, &XR, /**/ X, 0, ARMAS_PLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        // ---------------------------------------------------------------------
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_1x2to1x3(
            &XL, /**/ &X0, &x1, &X2, /**/ X, 1, ARMAS_PRIGHT);
        // ---------------------------------------------------------------------
        a11val = armas_x_get_unsafe(&a11, 0, 0);
        x1val = armas_x_get_unsafe(&x1, 0, 0);
        armas_x_gvcompute(&c, &s, &r, a11val, x1val);

        armas_x_set_unsafe(&a11, 0, 0, r);
        armas_x_gvrot_vec(&a21, &X2, c, s);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_1x3to1x2(
            &XL, &XR, /**/ &X0, &x1, /**/ X, ARMAS_PRIGHT);
    }
    return ATL.rows;
}


static
int unblk_cholupdate_upper(armas_x_dense_t * A, armas_x_dense_t * X,
                           armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a11, a12, A22;
    armas_x_dense_t XL, XR, X0, x1, X2;
    DTYPE c, s, r, a11val, x1val;

    EMPTY(x1);
    EMPTY(a11);
    EMPTY(A00);
    EMPTY(XL);
    EMPTY(X0);

    mat_partition_2x2(
        &ATL, __nil, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &XL, &XR, /**/ X, 0, ARMAS_PLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        // ---------------------------------------------------------------------
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_1x2to1x3(
            &XL, /**/ &X0, &x1, &X2, /**/ X, 1, ARMAS_PRIGHT);
        // ---------------------------------------------------------------------
        a11val = armas_x_get_unsafe(&a11, 0, 0);
        x1val = armas_x_get_unsafe(&x1, 0, 0);
        armas_x_gvcompute(&c, &s, &r, a11val, x1val);

        armas_x_set_unsafe(&a11, 0, 0, r);
        armas_x_gvrot_vec(&a12, &X2, c, s);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_1x3to1x2(
            &XL, &XR, /**/ &X0, &x1, /**/ X, ARMAS_PRIGHT);
    }
    return ATL.rows;
}


/**
 * @brief Rank update of unpivoted Cholesky factorization
 *
 * Computes Chol(A + x*x^T) = LL^T + xx^T or U^TU + xx^T
 *
 * @param[in,out] A
 *    On entry, original factorization. On exit, updated factorization.
 * @param[in,out] X
 *    On entry, update vector. On exit, contents of X are destroyed.
 * @param[in] flags
 *    Indicator flags, lower (ARMAS_LOWER) or upper (ARMAS_UPPER)
 *    triangular matrix
 * @param[in,out] conf
 *    Configuration block.
 *
 * @retval  0 ok
 * @retval -1 error
 */
int armas_x_cholupdate(armas_x_dense_t * A, armas_x_dense_t * X, int flags,
                       armas_conf_t * conf)
{
    armas_x_dense_t Xrow;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(A) == 0 || armas_x_size(X) == 0)
        return 0;

    // private functions expect row vector
    if (X->cols == 1) {
        armas_x_col_as_row(&Xrow, X);
    } else {
        armas_x_make(&Xrow, X->rows, X->cols, X->step, armas_x_data(X));
    }

    if (flags & ARMAS_UPPER) {
        unblk_cholupdate_upper(A, &Xrow, conf);
    } else {
        unblk_cholupdate_lower(A, &Xrow, conf);
    }

    return 0;
}


#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
