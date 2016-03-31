
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_ldlinverse_sym) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_blas) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <assert.h>
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

#include "sym.h"


/*
 * precondition:
 *      A = L*L.T && L = TriL(A)
 * invariant : 
 *      A*L = L.-T D.-1
 * pre-update : 
 *      A22*L22 = L22.-T*D2.-1
 * post-update:
 *      (a11 a21.T) ( I     ) = ( I l21.-T) (d1.-1   0  )
 *      (a21 A22  ) (l21 L22)   ( 0 l22.-T) (  0   D2.-1)
 *
 *      (1)  a11 + a21.T*l21 = d1.-1 --> a11 = d1.-1 - a21.T*L21
 *      (2)  a21 + A22*l21   = 0     --> a21 = -A22*L21
 *
 * post-update: (alternatively if invariant is A*L*D = L.-T
 *      (a11 a21.T) ( I     ) (d1  0 ) = ( I l21.-T)
 *      (a21 A22  ) (l21 L22) (0   D2)   ( 0 l22.-T)
 *
 *      (a11 a21.T) (    d1     0 ) = ( I l21.-T)
 *      (a21 A22  ) (l21*d1 L22*D2)   ( 0 l22.-T)
 *
 *      (1)  a11*d1 + a21.T*l21*d1 = I --> a11 = d1.-1 - a21.T*L21
 *      (2)  a21*d1 + A22*l21*d1   = 0 --> a21 = -A22*l21
 *
 */
static
int __unblk_invldl_lower(__armas_dense_t *A, __armas_dense_t *W, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, a11, a21, A22, l21;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(ATL); EMPTY(a11); EMPTY(A00);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    while (ATL.rows > 0 && ATL.cols > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11, __nil,
                               __nil,  &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        __armas_make(&l21, a21.rows, 1, a21.rows, __armas_data(W));
        __armas_mcopy(&l21, &a21);
        // a21 = - A22*l21 
        __armas_mvmult_sym(&a21, &A22, &l21, -__ONE, __ZERO, ARMAS_LOWER|ARMAS_LEFT, conf);
        // a11 = a11 - a21.T*l21
        udot    = __armas_dot(&a21, &l21, conf);
        a11val  = __ONE/__armas_get_unsafe(&a11, 0, 0);
        __armas_set_unsafe(&a11, 0, 0, a11val - udot);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}


/*
 * precondition:
 *   A = L*L.T && L = TriL(A)
 * invariant : 
 *   A*L = L.-T D.-1
 * pre-update : 
 *   A22*L22 = L22.-T*D2.-1
 * post-update:
 *   (A11 A21.T) (L11    ) = (L11.-T L21.-T) (D1.-1   0  )
 *   (A21 A22  ) (L21 L22)   (   0   L22.-T) (  0   D2.-1)
 *
 *   (1)  A11*L11 + A21.T*L21 = L11.-T*D1.-1 --> A11 = L11.-T*D1.1*L11.-1 - A21.T*L21*L11.-1
 *   (2)  A21*L11 + A22*L21   = 0            --> A21 = (-A22*L21)*L11.-1
 */
static
int __blk_invldl_lower(__armas_dense_t *A, __armas_dense_t *W, int lb, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, A11, A21, A22, L21, w2;
    int err = 0;
    
    EMPTY(ATL); EMPTY(A00);

    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &A11, __nil,
                               __nil,  &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        // workspaces;
        __armas_make(&w2, lb, 1, lb, __armas_data(W));
        __armas_make(&L21, A21.rows, A11.cols, A21.rows, &__armas_data(W)[lb]);
        __armas_mcopy(&L21, &A21);
        
        // A21 = - A22*L21*L11.-1
        __armas_mult_sym(&A21, &A22, &L21, - __ONE, __ZERO, ARMAS_LOWER, conf);
        __armas_solve_trm(&A21, &A11, __ONE, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
        
        // A11*L11 + A21.T*L21 = L11.-T -> A11 = inv(A11) - A21.T*L21*L11.-1
        __armas_solve_trm(&L21, &A11, __ONE, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
        if (__unblk_invldl_lower(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        // rank update
        __armas_update_trm(&A11, &A21, &L21, -__ONE, __ONE, ARMAS_TRANSA|ARMAS_LOWER, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}


/*
 * precondition:
 *   A = U*D*U.T && U = TriU(A)
 * invariant : 
 *   A*U*D = U.-T 
 * pre-update : 
 *   A00*u00*d1 = U00.-T
 * post-update:
 *   (A00   a01) (U00 u01)(D0  0) = ( U00.-T  0)
 *   (a01.T a11) ( 0   I )( 0 d1)   ( u01.-t  I)
 *
 *   (A00   a01) (U00*D0  u01*d1) = (U00.-T 0)
 *   (a01.T a11) (   0      d1  ) = (u01.-T I)
 *
 *   (1) A00*u01*d1 + a01*d1   = 0  --> a01 = -A00*u01   (*)
 *   (2) a01.T*u01*d1 + a11*d1 = I  --> a11 = d1.-1 - a01.T*u01 (*)
 *
 */
static
int __unblk_invldl_upper(__armas_dense_t *A, __armas_dense_t *W, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, a01, a11, A22, u01;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(A00); EMPTY(a11);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    while (ABR.rows > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,   &a01, __nil,
                               __nil,  &a11, __nil,
                               __nil, __nil,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------
        __armas_make(&u01, a01.rows, 1, a01.rows, __armas_data(W));
        __armas_mcopy(&u01, &a01);
        
        __armas_mvmult_sym(&a01, &A00, &u01, -__ONE, __ZERO, ARMAS_UPPER, conf);
        a11val = __ONE/__armas_get_unsafe(&a11, 0, 0);
        udot   = __armas_dot(&a01, &u01, conf);
        __armas_set_unsafe(&a11, 0, 0, a11val - udot);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

/*
 * precondition:
 *   A = U*D*U.T && U = TriU(A)
 * invariant : 
 *   A*U*D = U.-T
 * pre-update : 
 *   A00*U00*D0 = U00.-T
 * post-update:
 *   (A00   A01)(U00 U01)(D0  0) = (U00.-T   0   )
 *   (A01.T A11)( 0  U11)(0  D1)    U01.-T U11.-T)
 *
 *   (A00   A01)(U00*D0 U01*D1) = (U00.-T   0   )
 *   (A01.T A11)( 0     U11*D1)   (U01.-T U11.-T)
 *
 *   (1)  A00*U01*D1 + A01*U11*D1   = 0      -->  A01 = -A00*U01*U11.-1
 *   (2)  A01.T*U01*D1 + A11*U11*D1 = U11.-T -->  A11 = U11.-T*D1.-1*U11.-1 - A01.T*U01*U11.-1
 *                                           -->  A11 = inv(A11) - A01.T*U01*U11.-1
 */
static
int __blk_invldl_upper(__armas_dense_t *A, __armas_dense_t *W, int lb, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, A01, A11, A22, U01, w2;
    int err = 0;
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    
    while (ABR.rows > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,   &A01, __nil,
                               __nil,  &A11, __nil,
                               __nil, __nil,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------
        // workspaces;
        __armas_make(&w2, lb, 1, lb, __armas_data(W));
        __armas_make(&U01, A01.rows, A11.cols, A01.rows, &__armas_data(W)[lb]);
        __armas_mcopy(&U01, &A01);

        // A01 = - A00*U01*U11.-1
        __armas_mult_sym(&A01, &A00, &U01, -__ONE, __ZERO, ARMAS_LEFT|ARMAS_UPPER, conf);
        __armas_solve_trm(&A01, &A11, __ONE, ARMAS_RIGHT|ARMAS_UPPER|ARMAS_UNIT, conf);

        // A11 = inv(A11) - A01.T*U01*U11.-1
        __armas_solve_trm(&U01, &A11, __ONE, ARMAS_RIGHT|ARMAS_UPPER|ARMAS_UNIT, conf);
        if (__unblk_invldl_upper(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        __armas_update_trm(&A11, &A01, &U01, -__ONE, __ONE, ARMAS_UPPER|ARMAS_TRANSA, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}


/**
 * \brief Inverse of symmetric LDL^T or UDY^T factorized matrix
 *
 * \param[in,out] A
 *     On entry, the LDL^T (UDU^T) factorized symmetric matrix. On exit inverse of the
 *     matrix. 
 * \param[out]  W
 *     Workspace, miminum space needed is N elements.
 * \param[in] P
 *     Pivot vector
 * \param[in] flags
 *     Indicator flags, if ARMAS_LOWER (ARMAS_UPPER) is set then the elements of the 
 *     symmetric matrix are store in the lower (upper) triangular part of the matrix. 
 *     The strictly  upper (lower) part is not accessed.
 * \param[in,out] conf
 *     Blocking configuration, error status
 * 
 * \retval  0   OK
 * \retval -1   failure, conf.error holds actual error code
 */
int __armas_ldlinverse_sym(__armas_dense_t *A, __armas_dense_t *W, 
                           armas_pivot_t *P, int flags, armas_conf_t *conf)
{
    int lb, ws, err = 0;

    if (!conf)
        conf = armas_conf_default();
    
    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    lb = conf->lb;
    ws = __armas_size(W);
    if (ws < A->rows) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (lb != 0 && lb > ws/A->rows) {
        // adjust blocking factor to workspace
        lb = ws/A->rows;
        if (lb < 4) {
            lb = 0;
        }  else {        
            lb &= ~0x3; // make multiple of 4
        }
    }

    if (lb == 0 || A->rows <= lb) {
        if (flags & ARMAS_LOWER)
            err = __unblk_invldl_lower(A, W, conf);
        else 
            err = __unblk_invldl_upper(A, W, conf);
    } else {
        if (flags & ARMAS_LOWER)
            err = __blk_invldl_lower(A, W, lb, conf);
        else 
            err = __blk_invldl_upper(A, W, lb, conf);
    }

    if (P != ARMAS_NOPIVOT) {
        if (flags & ARMAS_UPPER) {
            __armas_pivot(A, P, ARMAS_PIVOT_UPPER|ARMAS_PIVOT_FORWARD, conf);
        } else {
            __armas_pivot(A, P, ARMAS_PIVOT_LOWER|ARMAS_PIVOT_BACKWARD, conf);
        }
    }
    return err;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
