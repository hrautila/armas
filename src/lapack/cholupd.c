
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Rank update Cholesky factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_cholupdate) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_blas) && defined(__armas_gvrot_vec)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
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
int __unblk_cholupdate_lower(__armas_dense_t *A, __armas_dense_t *X, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABL, ABR, A00, a11, a21, A22;
    __armas_dense_t XL, XR, X0, x1, X2;
    DTYPE c, s, r, a11val, x1val;

    EMPTY(x1); EMPTY(a11); EMPTY(A00); EMPTY(XL);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __partition_1x2(&XL,    &XR,  /**/  X, 0, ARMAS_PLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11, __nil,
                               __nil,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_1x2to1x3(&XL, /**/ &X0, &x1, &X2,  /**/ X, 1, ARMAS_PRIGHT);
        // ---------------------------------------------------------------------------
        a11val = __armas_get_unsafe(&a11, 0, 0);
        x1val  = __armas_get_unsafe(&x1, 0, 0);
        __armas_gvcompute(&c, &s, &r, a11val, x1val);

        __armas_set_unsafe(&a11, 0, 0, r);
        __armas_gvrot_vec(&a21, &X2, c, s);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/ &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __continue_1x3to1x2(&XL, &XR,    /**/ &X0,  &x1, /**/ X, ARMAS_PRIGHT);
    }
    return ATL.rows;
}


static
int __unblk_cholupdate_upper(__armas_dense_t *A, __armas_dense_t *X, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABL, ABR, A00, a11, a12, A22;
    __armas_dense_t XL, XR, X0, x1, X2;
    DTYPE c, s, r, a11val, x1val;

    EMPTY(x1); EMPTY(a11); EMPTY(A00); EMPTY(XL);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __partition_1x2(&XL,    &XR,  /**/  X, 0, ARMAS_PLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11,  &a12,
                               __nil, __nil,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_1x2to1x3(&XL, /**/ &X0, &x1, &X2,  /**/ X, 1, ARMAS_PRIGHT);
        // ---------------------------------------------------------------------------
        a11val = __armas_get_unsafe(&a11, 0, 0);
        x1val  = __armas_get_unsafe(&x1, 0, 0);
        __armas_gvcompute(&c, &s, &r, a11val, x1val);

        __armas_set_unsafe(&a11, 0, 0, r);
        __armas_gvrot_vec(&a12, &X2, c, s);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/ &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __continue_1x3to1x2(&XL, &XR,    /**/ &X0,  &x1, /**/ X, ARMAS_PRIGHT);
    }
    return ATL.rows;
}


/**
 * \brief Rank update of unpivoted Cholesky factorization 
 *
 * Computes Chol(A + x*x^T) = LL^T + xx^T or U^TU + xx^T
 *
 * \param[in,out] A
 *    On entry, original factorization. On exit, updated factorization.
 * \param[in,out] X
 *    On entry, update vector. On exit, contents of X are destroyed.
 * \param[in] flags
 *    Indicator flags, lower (ARMAS_LOWER) or upper (ARMAS_UPPER) triangular matrix
 * \param[in,out] conf
 *    Configuration block.
 *  
 * \retval  0 ok
 * \retval -1 error
 */
int __armas_cholupdate(__armas_dense_t *A, __armas_dense_t *X, int flags, armas_conf_t *conf)
{
    __armas_dense_t Xrow;
    
    if (!conf)
        conf = armas_conf_default();
    
    if (__armas_size(A) == 0 || __armas_size(X) == 0)
        return 0;

    // private functions expect row vector
    if (X->cols == 1) {
        __armas_col_as_row(&Xrow, X);
    } else {
        __armas_make(&Xrow, X->rows, X->cols, X->step, __armas_data(X));
    }

    if (flags & ARMAS_UPPER) {
        __unblk_cholupdate_upper(A, &Xrow, conf);
    } else {
        __unblk_cholupdate_lower(A, &Xrow, conf);
    }

    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

