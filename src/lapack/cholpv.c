
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__cholfactor_pv) && defined(__cholsolve_pv)
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
#include "pivot.h"

#include "sym.h"

/*
 * (1) Sven Hammarling , Nicholas J. Higham, and Craig Lucas
 *     LAPACK-Style Codes for Pivoted Cholesky and QR Updating
 * (2) Lapack working notes #161
 *
 *  From (1) stopping criteria 3.5
 *
 *  MAX {diag(A_k)} < N*eps* MAX{diag(A_0)}
 *  where 
 *     diag(A_k) is the diagonal of remaining submatrix on k'th round
 *     diag(A_0) is the diagonal of initial matrix
 */


static 
DTYPE __zeros(int i, int j)
{
    return __ZERO;
}

/*
 *  ( a11  a12 )   ( l11   0 )( l11  l21.t )
 *  ( a21  A22 )   ( l21 L22 )( 0    L22.t )
 *
 *   a11  =   l11*l11                => l11 = sqrt(a11)
 *   a21  =   l21*l11                => l21 = a21/l11
 *   A22  =   l21*l21.t + L22*L22.t  => A22 = A22 - l21*l21.T
 */
static
int __unblk_cholpv_lower(armas_x_dense_t *A, armas_pivot_t *P, DTYPE xstop, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a11, a21, A22, D;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(ABL);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11, __nil,
                               __nil,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        armas_x_diag(&D, &ABR, 0);
        im = armas_x_iamax(&D, conf);
        a11val = armas_x_get_at_unsafe(&D, im);
        if (a11val <= xstop) {
            // stopping criteria satisfied; zero trailing matrix ABR;
            armas_x_set_values(&ABR, __zeros, ARMAS_LOWER);
            // return rank
            return ATL.rows;
        }
        // pivoting
        if (im != 0) {
            __apply_bkpivot_lower(&ABR, 0, im, conf);
            __swap_rows(&ABL, 0, im, conf);
        }
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im + ATL.rows + 1);

        a11val = armas_x_get_unsafe(&a11, 0, 0);
        a11val = __SQRT(a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        armas_x_scale(&a21, __ONE/a11val, conf);
        armas_x_mvupdate_sym(&A22, -__ONE, &a21, ARMAS_LOWER, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return ATL.rows;
}

// compute D - diag(x*x.T);
static inline
void __update_diag(armas_x_dense_t *D, armas_x_dense_t *X)
{
    DTYPE d, x;
    int k;
    for (k = 0; k < armas_x_size(D); k++) {
        d  = armas_x_get_at_unsafe(D, k);
        x  = armas_x_get_at_unsafe(X, k);
        d -= x*x;
        armas_x_set_at_unsafe(D, k, d);
    }
}

/*
 * Compute pivoting Cholesky factorization for first N columns without updating the 
 * trailing matrix; 
 */
static
int __unblk_cholpv_lower_ncol(armas_x_dense_t *A, armas_x_dense_t *D,
                              armas_pivot_t *P, DTYPE xstop, int ncol, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a10, a11, A20, a21, A22;
    armas_x_dense_t DT, DB, D0, d1, D2;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(ABL);
    
    // make copy of diagonal elements;
    armas_x_diag(&D0, A, 0);
    armas_x_copy(D, &D0, conf);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __partition_2x1(&DT,
                    &DB,  /**/ D, 0, ARMAS_PTOP);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ncol-- > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL, /**/
                               &A00,  __nil, __nil,
                               &a10,  &a11, __nil,
                               &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_2x1to3x1(&DT, /**/
                               &D0, &d1, &D2,  /**/  D, 1, ARMAS_PBOTTOM);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        im = armas_x_iamax(&DB, conf);
        a11val = armas_x_get_at_unsafe(&DB, im);
        if (a11val <= xstop) {
            armas_x_set_values(&ABR, __zeros, ARMAS_LOWER);
            return ATL.cols;
        }
        // pivot??
        if (im != 0) {
            __apply_bkpivot_lower(&ABR, 0, im, conf);
            __swap_rows(&ABL, 0, im, conf);
            a11val = armas_x_get_at_unsafe(&DB, im);
            armas_x_set_at_unsafe(&DB, im, armas_x_get_at_unsafe(&DB, 0));
            armas_x_set_at_unsafe(&DB, 0,  a11val);
        }
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im+ATL.rows+1);

        a11val = armas_x_get_at_unsafe(&d1, 0);
        a11val = __SQRT(a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // update a21 with prevous; a21 := a21 - A20*a10
        armas_x_mvmult(__ONE, &a21, -__ONE, &A20, &a10, 0, conf);
        armas_x_scale(&a21, __ONE/a11val, conf);
        // update diagonal; diag(A22) = D2 - diag(a21*a21.T); 
        if (ncol > 0)
            __update_diag(&D2, &a21);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __continue_3x1to2x1(&DT,
                            &DB,         /**/  &D0, &d1,   /**/ D, ARMAS_PBOTTOM);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return ATL.cols;
}


/*
 * Blocked pivoting Cholesky factorization
 */
static
int __blk_cholpv_lower(armas_x_dense_t *A, armas_x_dense_t *W, 
                       armas_pivot_t *P, DTYPE xstop, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, A11, A21, A22, D;
    armas_pivot_t pT, pB, p0, p1, p2;
    int k, e, np;

    EMPTY(A00); EMPTY(A11);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ABR.rows > lb) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &A11, __nil,
                               __nil,  &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        armas_x_make(&D, ABR.rows, 1, ABR.rows, armas_x_data(W));
        // if non-zero then stopping criteria satisfied with 
        e = __unblk_cholpv_lower_ncol(&ABR, &D, &p1, xstop, A11.cols, conf);
        // partial pivot of rows on left side; update pivot indexes;
        armas_x_pivot(&ABL, &p1, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_FORWARD, conf);
        np = e < A11.cols ? e : A11.cols;
        for (k = 0; k < np; k++) {
            armas_pivot_set_unsafe(&p1, k, armas_pivot_get_unsafe(&p1, k)+ATL.rows);
        }
        if (e  != A11.cols) {
            return e + ATL.rows;
        }
        // A22 = A22 - A21*A21.T
        armas_x_update_sym(__ONE, &A22, -__ONE, &A21, ARMAS_LOWER, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        e = __unblk_cholpv_lower(&ABR, &pB, xstop, conf);
        // pivot left side
        armas_x_pivot(&ABL, &pB, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_FORWARD, conf);
        np = e < ABR.cols ? e : ABR.cols;
        for (k = 0; k < np; k++) {
            armas_pivot_set_unsafe(&pB, k, armas_pivot_get_unsafe(&pB, k)+ATL.rows);
        }
        if (e != ABR.cols) {
            return e + ATL.rows;
        }
    }
    // full rank;
    return A->cols;
}


/*
 * BOTTOM to TOP
 *  ( A00  a01 )   ( U00 u01 )( U00.t  0  )
 *  ( a10  a11 )   (  0  u11 )( u01.t u11 )
 *
 *   a11  =   u11*u11                 => u11 = sqrt(a11)
 *   a01  =   u01*u11                 => u01 = a01/u11
 *   A00  =   u01*u01.t + U00*U00.>T  => A00 = A00 - a01*a01.t
 *
 * TOP to BOTTOM (use this)
 *  ( a11  a12.T )   ( u11  0   )( u11 u12.T )
 *  ( a12  A22   )   ( u12 U22.T)(  0  U22   )
 *
 *   a11   = u11*u11               => sqrt(u11)
 *   a12.T = u11*u12.T             => u12 = a12/u11
 *   A22   = u12*u12.T + U22.T*U22 => A22 = A22 - u12*u12.T
 */
static
int __unblk_cholpv_upper(armas_x_dense_t *A, armas_pivot_t *P, DTYPE xstop, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a11, a12, A22, D; 
    armas_pivot_t pT, pB, p0, p1, p2;
    int im;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(ATR);
    
   __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ABR.rows > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &a11,  &a12,
                               __nil, __nil,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        armas_x_diag(&D, &ABR, 0);
        im = armas_x_iamax(&D, conf); 
        a11val = armas_x_get_at_unsafe(&D, im);
        if (a11val <= xstop) {
            // not full rank
            armas_x_set_values(&ABR, __zeros, ARMAS_UPPER);
            return ATL.cols;
        }
        // pivoting
        if (im != 0) {
            __apply_bkpivot_upper_top(&ABR, 0, im, conf);
            __swap_cols(&ATR, 0, im, conf);
        }
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im + ATL.rows + 1);

        a11val = armas_x_get_unsafe(&a11, 0, 0);
        a11val = __SQRT(a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // u12 = a12/a11
        armas_x_scale(&a12, __ONE/a11val, conf);
        armas_x_mvupdate_sym(&A22, -__ONE, &a12, ARMAS_UPPER, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
     return ATL.cols;
}


/*
 * Compute pivoting U^TU factorization for first N columns without updating the 
 * trailing matrix; Y is A.rows,ncol matrix that will hold U21*D1
 */
static
int __unblk_cholpv_upper_ncol(armas_x_dense_t *A, armas_x_dense_t *D, 
                              armas_pivot_t *P, DTYPE xstop, int ncol, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a01, A02, a11, a12, A22;
    armas_x_dense_t DT, DB, D0, d1, D2;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(ATR);
    
    // make copy of diagonal elements;
    armas_x_diag(&D0, A, 0);
    armas_x_copy(D, &D0, conf);
    
    __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __partition_2x1(&DT,
                    &DB,  /**/ D, 0, ARMAS_PTOP);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ncol-- > 0) {
        // ---------------------------------------------------------------------------
        __repartition_2x2to3x3(&ATL, /**/
                               &A00,  &a01,  &A02,
                               __nil, &a11,  &a12,
                               __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_2x1to3x1(&DT, /**/
                               &D0, &d1, &D2,  /**/  D, 1, ARMAS_PBOTTOM);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        im = armas_x_iamax(&DB, conf);
        a11val = armas_x_get_at_unsafe(&DB, im);
        if (a11val <= xstop) {
            armas_x_set_values(&ABR, __zeros, ARMAS_UPPER);
            return ATL.cols;
        }
        if (im != 0) {
            // pivot
            __apply_bkpivot_upper_top(&ABR,  0, im, conf);
            __swap_cols(&ATR, 0, im, conf);
            a11val = armas_x_get_at_unsafe(&DB, im);
            armas_x_set_at_unsafe(&DB, im, armas_x_get_at_unsafe(&DB, 0));
            armas_x_set_at_unsafe(&DB, 0,  a11val);
        }
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im + ATL.rows + 1);

        a11val = armas_x_get_at_unsafe(&d1, 0);
        a11val = __SQRT(a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // update a01 with prevous; a12 := a12 - A02.T*a01
        armas_x_mvmult(__ONE, &a12, -__ONE, &A02, &a01, ARMAS_TRANS, conf);
        armas_x_scale(&a12, __ONE/a11val, conf);
        if (ncol > 0)
            __update_diag(&D2, &a12);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __continue_3x1to2x1(&DT,
                            &DB,         /**/  &D0, &d1,   /**/ D, ARMAS_PBOTTOM);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return ATL.cols;
}


static
int __blk_cholpv_upper(armas_x_dense_t *A, armas_x_dense_t *W,
                       armas_pivot_t *P, DTYPE xstop, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, A11, A12, A22, D;
    armas_pivot_t pT, pB, p0, p1, p2;
    int k, e, np;

    EMPTY(A00); EMPTY(A11);
    
    __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);


    while (ABR.rows > lb) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil,  &A11,  &A12,
                               __nil, __nil,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        armas_x_make(&D, ABR.cols, 1, ABR.cols, armas_x_data(W));
        
        e =__unblk_cholpv_upper_ncol(&ABR, &D, &p1, xstop, A11.cols, conf);
        // pivot colums above; update pivot indexes
        armas_x_pivot(&ATR, &p1, ARMAS_PIVOT_COLS|ARMAS_PIVOT_FORWARD, conf);
        np = e < A11.cols ? e : A11.cols;
        for (k = 0; k < np; k++) {
            armas_pivot_set_unsafe(&p1, k, armas_pivot_get_unsafe(&p1, k)+ATL.rows);
        }
        if (e != A11.cols) {
            return e + ATL.cols;
        }
        armas_x_update_sym(__ONE, &A22, -__ONE, &A12, ARMAS_UPPER|ARMAS_TRANS, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    if (ABR.cols > 0) {
        e = __unblk_cholpv_upper(&ABR, &pB, xstop, conf);
        armas_x_pivot(&ATR, &pB, ARMAS_PIVOT_COLS|ARMAS_PIVOT_FORWARD, conf);
        np = e < ABR.cols ? e : ABR.cols;
        for (k = 0; k < np; k++) {
            armas_pivot_set_unsafe(&pB, k, armas_pivot_get_unsafe(&pB, k)+ATL.rows);
        }
        if (e != ABR.cols)
            return e + ATL.cols;

    }
    // full rank; 
    return A->cols;
}


/**
 * \brief Compute pivoting Cholesky factorization of symmetric positive definite matrix
 *
 * \param A
 *     On entry symmetric matrix store on lower (upper) triangular part. On exit
 *     the P^T(LL^T)P or P^T(U^TU)P factorization of where L (U) is lower (upper) triangular
 *     matrix.
 * \param W
 *     Working space for blocked implementation. If null or zero sized then unblocked
 *     algorithm used. Blocked algorithm needs workspace of size N elements.
 * \param flags
 *     Indicator bits, lower (upper) storage if ARMAS_LOWER (ARMAS_UPPER) set.
 * \param conf
 *     Configuration block
 *
 * \retval 0  ok; rank not changed
 * \retval >0 ok; computed rank of pivoted matrix
 * \retval <0 error
 */
int __cholfactor_pv(armas_x_dense_t *A, armas_x_dense_t *W,
                    armas_pivot_t *P, int flags, armas_conf_t *conf)
{
    int ws, lb, err = 0;
    DTYPE xstop;
    armas_x_dense_t D;

    if (!conf)
        conf = armas_conf_default();

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (A->rows != armas_pivot_size(P)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    lb = conf->lb;
    ws = armas_x_size(W);
    if (ws < A->cols)
        ws = 0;

    armas_x_diag(&D, A, 0);
    xstop = armas_x_amax(&D, conf) * A->rows * __EPS;

    if (lb == 0 || ws == 0 || A->rows <= lb) {
        // unblocked code
        if (flags & ARMAS_UPPER) {
            err = __unblk_cholpv_upper(A, P, xstop, conf);
        } else {
            err = __unblk_cholpv_lower(A, P, xstop, conf);
        }
    } else {
        // blocked version
        if (flags & ARMAS_UPPER) {
            err = __blk_cholpv_upper(A, W, P, xstop, lb, conf);
        } else {
            err = __blk_cholpv_lower(A, W, P, xstop, lb, conf);
        }
    }
    // if full rank; 
    return err == A->cols ? 0 : err;
}


/**
 * \brief Solve X = A*B or X = A.T*B where A is symmetric matrix
 *
 * \param[in,out] B
 *    On entry, input values. On exit, the solutions matrix
 * \param[in] A
 *    The LDL.T (UDU.T) factorized symmetric matrix
 * \param[in] flags
 *    Indicator flags, lower (ARMAS_LOWER) or upper (ARMAS_UPPER) triangular matrix
 * \param[in,out] conf
 *    Configuration block.
 *  
 * \retval  0 ok
 * \retval -1 error
 */
int __cholsolve_pv(armas_x_dense_t *B, armas_x_dense_t *A, 
                   armas_pivot_t *P, int flags, armas_conf_t *conf)
{
    int pivot1_dir, pivot2_dir;
    if (!conf)
        conf = armas_conf_default();
    
    if (armas_x_size(B) == 0 || armas_x_size(A) == 0)
        return 0;

    if (A->rows != A->cols || A->cols != B->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (B->rows != P->npivots) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    pivot1_dir = ARMAS_PIVOT_FORWARD;
    pivot2_dir = ARMAS_PIVOT_BACKWARD;

    armas_x_pivot(B, P, ARMAS_PIVOT_ROWS|pivot1_dir, conf);

    /*
     * A*X = (LL^T)*X = B -> X = L^-T*(L^-1*B)
     * A*X = (U^TU)*X = B -> X = U^-1*(U^-T*B)
     */
    if (flags & ARMAS_UPPER) {
        armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANS, conf);
        armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_UPPER, conf);
    } else {
        armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_LOWER, conf);
        armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANS, conf);
    }

    armas_x_pivot(B, P, ARMAS_PIVOT_ROWS|pivot2_dir, conf);

    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

