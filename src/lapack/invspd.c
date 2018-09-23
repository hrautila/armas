
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_cholinverse) && defined(armas_x_cholinverse_w) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

// symmetric positive definite matrix inverse
static
int __unblk_invspd_lower(armas_x_dense_t *A, armas_x_dense_t *W, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a21, A22, l21;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(a11); EMPTY(ATL); EMPTY(A00);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11, __nil,
                               __nil,  &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        // inverse upper part; 
        a11val = armas_x_get_unsafe(&a11, 0, 0);
        if (a11val == __ZERO) {
            conf->error = ARMAS_ESINGULAR;
            return -1;
        }

        armas_x_submatrix(&l21, W, 0, 0, armas_x_size(&a21), 1);
        armas_x_copy(&l21, &a21, conf);

        // a21 = - A22*l21 / l11
        armas_x_mvmult_sym(__ZERO, &a21, -__ONE/a11val, &A22, &l21, ARMAS_LOWER, conf);

        // a11 = (u11.-1 - a21.T*l21)/l11 ; l11 = u11 = a11
        udot = armas_x_dot(&a21, &l21, conf);
        a11val  = (__ONE/a11val - udot)/a11val;
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}

/*
 * precondition:
 *      A = L*L.T && L = TriL(A)
 * invariant : 
 *      A*L = L.-T
 * pre-update : 
 *      A22*L22 = L22.-T
 * post-update:
 *      (A11 A21.T) (L11    ) = (L11.-T L21.-T)
 *      (A21 A22  ) (L21 L22)   (   0   L22.-T)
 *
 *      (1)  A11*L11 + A21.T*L21 = L11.-T --> A11 = L11.-T*L11.-1 - A21.T*L21*L11.-1
 *      (2)  A21*L11 + A22*L21   = 0      --> A21 = (-A22*L21)*L11.-1
 */
static
int __blk_invspd_lower(armas_x_dense_t *A, armas_x_dense_t *W, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A21, A22, L21, w2;
    int err = 0;
    
    EMPTY(A00); EMPTY(ATL);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &A11, __nil,
                               __nil,  &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------

        // workspaces;
        armas_x_make(&w2, lb, 1, lb, armas_x_data(W));
        armas_x_make(&L21, A21.rows, A11.cols, A21.rows, &armas_x_data(W)[lb]);
        armas_x_mcopy(&L21, &A21);
        
        // A21 = - A22*L21*L11.-1
        armas_x_mult_sym(__ZERO, &A21, -__ONE, &A22, &L21, ARMAS_LOWER, conf);
        armas_x_solve_trm(&A21, __ONE, &A11, ARMAS_LOWER|ARMAS_RIGHT, conf);
        
        // A11*L11 + A21.T*L21 = L11.-T -> A11 = inv(A11) - A21.T*L21*L11.-1
        armas_x_solve_trm(&L21, __ONE, &A11, ARMAS_LOWER|ARMAS_RIGHT, conf);
        if (__unblk_invspd_lower(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        // rank update
        armas_x_update_trm(__ONE, &A11, -__ONE, &A21, &L21, ARMAS_TRANSA|ARMAS_LOWER, conf);

        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}


static
int __unblk_invspd_upper(armas_x_dense_t *A, armas_x_dense_t *W, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, A22, u12;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(a11); EMPTY(ATL); EMPTY(A00);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11,  &a12,
                               __nil, __nil,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        // inverse upper part; 
        a11val = armas_x_get_unsafe(&a11, 0, 0);
        if (a11val == __ZERO) {
            conf->error = ARMAS_ESINGULAR;
            return -1;
        }

        armas_x_submatrix(&u12, W, 0, 0, armas_x_size(&a12), 1);
        armas_x_copy(&u12, &a12, conf);

        // a12 = - A22.T*u12 / u11
        armas_x_mvmult_sym(__ZERO, &a12, -__ONE/a11val, &A22, &u12, ARMAS_UPPER|ARMAS_TRANSA, conf);

        // a11 = (u11.-1 - a12.T*u12)/l11 ; l11 = u11 = a11
        udot = armas_x_dot(&a12, &u12, conf);
        a11val  = (__ONE/a11val - udot)/a11val;
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}

/*
 * precondition:
 *      A = U.T*U && L = TriU(A)
 * invariant : 
 *      U*A = U.-T
 * pre-update : 
 *      U22*A22 = U22.-T
 * post-update:
 *      (U11 U12)(A11   A12) = (U11.-T       )
 *      (    U22)(A12.T A22)   (U12.-T U22.-T)
 *
 *      (1)  U11*A11 + U12*A12.T = U11.-T --> A11 = U11.-1*U11.-T - U11.-1*U12*A12.T
 *      (2)  U11*A12 + U12*A22   = 0      --> A12 = U11.-1*(-U12*A22)
 */
static
int __blk_invspd_upper(armas_x_dense_t *A, armas_x_dense_t *W, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A22, U12, w2;
    int err = 0;
    
    EMPTY(A00); EMPTY(ATL);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &A11,  &A12,
                               __nil, __nil,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        // workspaces;
        armas_x_make(&w2, lb, 1, lb, armas_x_data(W));
        armas_x_make(&U12, A11.rows, A12.cols, A11.rows, &armas_x_data(W)[lb]);
        armas_x_mcopy(&U12, &A12);

        // A12 = - U11.-1*U12*A22
        armas_x_mult_sym(__ZERO, &A12, -__ONE, &A22, &U12, ARMAS_RIGHT|ARMAS_UPPER, conf);
        armas_x_solve_trm(&A12, __ONE, &A11, ARMAS_LEFT|ARMAS_UPPER, conf);

        // A11 = inv(A11) - U11.-1*U12*A12.T
        armas_x_solve_trm(&U12, __ONE, &A11, ARMAS_LEFT|ARMAS_UPPER, conf);
        if (__unblk_invspd_upper(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        armas_x_update_trm(__ONE, &A11, -__ONE, &U12, &A12, ARMAS_UPPER|ARMAS_TRANSB, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}


/**
 * \brief Computes the inverse of a positive definite symmetric NxN matrix.
 *
 * \param[in,out] A
 *      On entry, the lower (or upper) Cholesky factorization of matrix A. On exit the inverse of A
 *      stored on lower (or upper) part of matrix A.
 * \param[in]  flags
 *      Indicator flags ARMAS_UPPER or ARMAS_LOWER.
 * \param[out] wb
 *      Workspace 
 * \param[in]  conf
 *      Configuration block
 *
 * \retval  0 Succes
 * \retval -1 Error, error code set in conf.error
 *
 */
int armas_x_cholinverse_w(armas_x_dense_t *A,
                          int flags,
                          armas_wbuf_t *wb,
                          armas_conf_t *conf)
{
    int lb, err = 0;
    size_t wsmin, wsz;
    armas_x_dense_t Wt;

    if (!conf)
        conf = armas_conf_default();
    
    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    if (wb && wb->bytes == 0) {
        if (conf->lb > 0 && A->rows > conf->lb)
            wb->bytes = conf->lb * A->rows * sizeof(DTYPE);
        else
            wb->bytes = A->rows * sizeof(DTYPE);
        return 0;
    }
    
    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    wsmin = A->rows * sizeof(DTYPE);
    wsz   = armas_wbytes(wb);
    if (wsz < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    lb = conf->lb;
    wsz /= sizeof(DTYPE);
    if (A->rows > lb) {
        if (wsz < lb * A->rows) {
            lb = (wsz / A->rows) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    armas_x_make(&Wt, wsz, 1, wsz, (DTYPE *)armas_wptr(wb));
    wsz = armas_wpos(wb);

    if (lb == 0 || A->rows <= lb) {
        if (flags & ARMAS_LOWER)
            err = __unblk_invspd_lower(A, &Wt, conf);
        else 
            err = __unblk_invspd_upper(A, &Wt, conf);
    } else {
        if (flags & ARMAS_LOWER)
            err = __blk_invspd_lower(A, &Wt, lb, conf);
        else 
            err = __blk_invspd_upper(A, &Wt, lb, conf);
    }
    return err;
}

int armas_x_cholinverse(armas_x_dense_t *A,
                        int flags,
                        armas_conf_t *conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (!conf)
        conf = armas_conf_default();

    if (armas_x_cholinverse_w(A, flags, &wb, conf) < 0)
        return -1;
    
    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    }
    else
        wbs = ARMAS_NOWORK;
                
    err = armas_x_cholinverse_w(A, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;

}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
