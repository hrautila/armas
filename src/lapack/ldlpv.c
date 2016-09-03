
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ldlfactor) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mvupdate_trm) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "pivot.h"

#include "ldl.h"

extern
int __ldlfactor_np(armas_x_dense_t *A, armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern
int __ldlsolve_np(armas_x_dense_t *A, armas_x_dense_t *W, int flags, armas_conf_t *conf);


/*
 *  ( a11  a12 )   ( 1   0   )( d1  0  )( 1  l21.t )
 *  ( a21  A22 )   ( l21 L22 )(  0  D2 )( 0  L22.t )
 *
 *   a11  =   d1
 *   a21  =   l21*d1                       => a21 = a21/d1
 *   A22  =   l21*d1*l21.t + L22*D2*L22.t  => A22 = A22 - l21*d1*l21t
 */
static
int __unblk_ldlpv_lower(armas_x_dense_t *A, armas_pivot_t *P, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a11, a21, A22, D;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im, err = 0;
    DTYPE a11val;

    EMPTY(a11); EMPTY(ATL); EMPTY(ABL); EMPTY(A00);
    
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
        if ( (im = armas_x_iamax(&D, conf)) != 0) {
            // pivot
            __apply_bkpivot_lower(&ABR, 0, im, conf);
            __swap_rows(&ABL, 0, im, conf);
        }
        // A22 = A22 - l21*d11*l21.T = A22 - a21*a21.T/a11
        a11val = __ONE/armas_x_get_unsafe(&a11, 0, 0);
        armas_x_mvupdate_trm(&A22, &a21, &a21, -a11val, ARMAS_LOWER, conf);
        // l21 = a21/a11
        armas_x_scale(&a21, a11val, conf);
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im + ATL.rows + 1);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return err;
}

// compute D - alpha*diag(x.T*x);
static inline
void __update_diag(armas_x_dense_t *D, armas_x_dense_t *X, DTYPE alpha)
{
    DTYPE d, x;
    int k;
    for (k = 0; k < armas_x_size(D); k++) {
        d = armas_x_get_at_unsafe(D, k);
        x = armas_x_get_at_unsafe(X, k);
        d -= (alpha*x)*x;
        armas_x_set_at_unsafe(D, k, d);
    }
}

/*
 * Compute pivoting LDL factorization for first N columns without updating the 
 * trailing matrix; Y is A.rows,ncol matrix that will hold L21*D1
 */
static
int __unblk_ldlpv_lower_ncol(armas_x_dense_t *A, armas_x_dense_t *Y, 
                             armas_pivot_t *P, int ncol, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a10, a11, A20, a21, A22, D;
    armas_x_dense_t DT, DB, D0, d1, D2;
    armas_x_dense_t YTL, YBL, YBR, Y00, y10, y11, Y20, y21, Y22;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im, err = 0;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(ABL); EMPTY(y11); EMPTY(Y00); EMPTY(YBL); 
    
    // make copy of diagonal elements; last column of Y matrix; will be overwriten in the end
    armas_x_column(&D, Y, Y->cols-1);
    armas_x_diag(&D0, A, 0);
    armas_x_copy(&D, &D0, conf);
    
    __partition_2x2(&ATL, __nil,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
    __partition_2x2(&YTL, __nil,
                    &YBL, &YBR,   /**/  Y, 0, 0, ARMAS_PTOPLEFT);
    __partition_2x1(&DT,
                    &DB,  /**/ &D, 0, ARMAS_PTOP);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PTOP);

    while (ncol-- > 0) {
        __repartition_2x2to3x3(&ATL, /**/
                               &A00,  __nil, __nil,
                               &a10,  &a11, __nil,
                               &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_2x2to3x3(&YTL, /**/
                               &Y00,  __nil, __nil,
                               &y10,  &y11, __nil,
                               &Y20,  &y21,  &Y22,  /**/  Y, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_2x1to3x1(&DT, /**/
                               &D0, &d1, &D2,  /**/  &D, 1, ARMAS_PBOTTOM);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------------
        if ( (im = armas_x_iamax(&DB, conf)) != 0) {
            // pivot
            __apply_bkpivot_lower(&ABR, 0, im, conf);
            __swap_rows(&ABL, 0, im, conf);
            __swap_rows(&YBL, 0, im, conf);
            a11val = armas_x_get_at_unsafe(&DB, im);
            armas_x_set_at_unsafe(&DB, im, armas_x_get_at_unsafe(&DB, 0));
            armas_x_set_at_unsafe(&DB, 0,  a11val);
        }
        // ---------------------------------------------------------------------------
        a11val = armas_x_get_at_unsafe(&d1, 0);
        // update a21 with prevous; a21 := a21 - A20*y10
        armas_x_mvmult(&a21, &A20, &y10, -__ONE, __ONE, 0, conf);
        // this will overwrite D2 when ncol == 0; 
        armas_x_copy(&y21, &a21, conf);
        // update diagonal; diag(A22) = D2 - diag(a21*a21.T)/a11; 
        if (ncol > 0)
            __update_diag(&D2, &a21, __ONE/a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // a21 = a21/a11
        armas_x_scale(&a21, __ONE/a11val, conf);
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im+ATL.rows+1);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __continue_3x3to2x2(&YTL, __nil,
                            &YBL,  &YBR, /**/  &Y00, &y11, &Y22,   Y, ARMAS_PBOTTOMRIGHT);
        __continue_3x1to2x1(&DT,
                            &DB,         /**/  &D0, &d1,   /**/ &D, ARMAS_PBOTTOM);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return err;
}


/*
 *
 */
static
int __blk_ldlpv_lower(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, A11, A21, A22, YL, Y11, Y21;
    armas_pivot_t pT, pB, p0, p1, p2;
    int k, err = 0;

    EMPTY(A11); EMPTY(A00);
    
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
        armas_x_make(&YL, ABL.rows, A21.cols, ABL.rows, armas_x_data(W));
        
        __unblk_ldlpv_lower_ncol(&ABR, &YL, &p1, A11.cols, conf);
        __partition_2x1(&Y11,
                        &Y21, /**/ &YL, A11.rows, ARMAS_PTOP);

        // A22 = A22 - L21*D1*L21.T   (Y21 is L21*D1, A21 is L21)
        armas_x_update_trm(&A22, &Y21, &A21, -__ONE, __ONE, ARMAS_LOWER|ARMAS_TRANSB, conf);

        // pivot rows on left side; update pivot indexes;
        armas_x_pivot(&ABL, &p1, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_FORWARD, conf);
        for (k = 0; k < lb; k++) {
            armas_pivot_set_unsafe(&p1, k, armas_pivot_get_unsafe(&p1, k)+ATL.rows);
        }
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, __nil,
                            &ABL,  &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        __unblk_ldlpv_lower(&ABR, &p2, conf);
        armas_x_pivot(&ABL, &p2, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_FORWARD, conf);
        for (k = 0; k < ABR.cols; k++) {
            armas_pivot_set_unsafe(&p2, k, armas_pivot_get_unsafe(&p2, k)+ATL.rows);
        }
    }
    return err;
}


/*
 *  ( A00  a01 )   ( U00 u01 )( D0  0  )( U00.t 0 )
 *  ( a10  a11 )   (  0   1  )(  0  d1 )( u01.t 1 )
 *
 *   a11  =   d1
 *   a01  =   u01*d1                       => a01 = a01/a11
 *   A00  =   u01*d1*u01.t + U00*D1*U00.t  => A00 = A00 - a01*a01.t/a11
 */
static
int __unblk_ldlpv_upper(armas_x_dense_t *A, armas_pivot_t *P, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a11, a01, A22, D;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im, err = 0;
    DTYPE a11val;

    EMPTY(a11); EMPTY(ATL); EMPTY(ATR);
    
    __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PBOTTOM);

    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,   &a01, __nil,
                               __nil,  &a11, __nil,
                               __nil, __nil,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------------
        armas_x_diag(&D, &ATL, 0);
        if ( (im = armas_x_iamax(&D, conf)) != ATL.rows-1) {
            // pivot
            __apply_bkpivot_upper(&ATL, ATL.rows-1, im, conf);
            __swap_rows(&ATR, ATL.rows-1, im, conf);
        }
        // ---------------------------------------------------------------------------
        // A00 = A00 - u01*d1*u01.T = A00 - a01*a01.T/a11
        a11val = __ONE/armas_x_get_unsafe(&a11, 0, 0);
        armas_x_mvupdate_trm(&A00, &a01, &a01, -a11val, ARMAS_UPPER, conf);
        // u01 = a01/a11
        armas_x_scale(&a01, a11val, conf);
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im + 1);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }
    return err;
}



/*
 * Compute pivoting UDU^T factorization for last N columns without updating the 
 * trailing matrix; Y is A.rows,ncol matrix that will hold U21*D1
 */
static
int __unblk_ldlpv_upper_ncol(armas_x_dense_t *A, armas_x_dense_t *Y, 
                             armas_pivot_t *P, int ncol, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a01, A02, a11, a12, A22, D;
    armas_x_dense_t DT, DB, D0, d1, D2;
    armas_x_dense_t YTL, YTR, YBR, Y00, y01, Y02, y11, y12, Y22;
    armas_pivot_t pT, pB, p0, p1, p2;
    int im, err = 0;
    DTYPE a11val;

    EMPTY(a11); EMPTY(A00); EMPTY(Y00); EMPTY(YTL); EMPTY(YTR);
    EMPTY(ATL); EMPTY(ATR);
    
    // make copy of diagonal elements; first column of Y matrix; will be overwriten in the end
    armas_x_column(&D, Y, 0);
    armas_x_diag(&D0, A, 0);
    armas_x_copy(&D, &D0, conf);
    
    __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    __partition_2x2(&YTL,  &YTR,
                    __nil, &YBR,   /**/  Y, 0, 0, ARMAS_PBOTTOMRIGHT);
    __partition_2x1(&DT,
                    &DB,  /**/ &D, 0, ARMAS_PBOTTOM);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PBOTTOM);

    while (ncol-- > 0) {
        __repartition_2x2to3x3(&ATL, /**/
                               &A00,  &a01,  &A02,
                               __nil, &a11,  &a12,
                               __nil, __nil, &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        __repartition_2x2to3x3(&YTL, /**/
                               &Y00,  &y01,  &Y02,
                               __nil, &y11,  &y12,
                               __nil, __nil, &Y22,  /**/  Y, 1, ARMAS_PTOPLEFT);
        __repartition_2x1to3x1(&DT, /**/
                               &D0, &d1, &D2,  /**/  &D, 1, ARMAS_PTOP);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------------
        if ( (im = armas_x_iamax(&DT, conf)) != ATL.rows-1) {
            // pivot
            __apply_bkpivot_upper(&ATL, ATL.rows-1, im, conf);
            __swap_rows(&ATR, ATL.rows-1, im, conf);
            __swap_rows(&YTR, ATL.rows-1, im, conf);
            a11val = armas_x_get_at_unsafe(&DT, im);
            armas_x_set_at_unsafe(&DT, im, armas_x_get_at_unsafe(&DT, ATL.rows-1));
            armas_x_set_at_unsafe(&DT, ATL.rows-1,  a11val);
        }
        // ---------------------------------------------------------------------------
        a11val = armas_x_get_at_unsafe(&d1, 0);
        // update a01 with prevous; a01 := a01 - A02*y12
        armas_x_mvmult(&a01, &A02, &y12, -__ONE, __ONE, 0, conf);
        // this will overwrite D0 when ncol == 0; 
        armas_x_copy(&y01, &a01, conf);
        // update diagonal; diag(A00) = D0 - diag(a01*a01.T)/a11; 
        if (ncol > 0)
            __update_diag(&D0, &a01, __ONE/a11val);
        armas_x_set_unsafe(&a11, 0, 0, a11val);
        // a01 = a01/a11
        armas_x_scale(&a01, __ONE/a11val, conf);
        // save pivot; positive index 
        armas_pivot_set(&p1, 0, im+1);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
        __continue_3x3to2x2(&YTL,  &YTR,
                            __nil, &YBR, /**/  &Y00, &y11, &Y22,   Y, ARMAS_PTOPLEFT);
        __continue_3x1to2x1(&DT,
                            &DB,         /**/  &D0, &d1,   /**/ &D, ARMAS_PTOP);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }
    return err;
}


static
int __blk_ldlpv_upper(armas_x_dense_t *A, armas_x_dense_t *W,
                      armas_pivot_t *P, int lb, armas_conf_t *conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, A01, A11, A22, Y, Y11, Y01;
    armas_pivot_t pT, pB, p0, p1, p2;
    int err = 0;

    EMPTY(ATL); EMPTY(A11);
    
    __partition_2x2(&ATL,  &ATR,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PBOTTOM);
    __pivot_2x1(&pT,
                &pB,  /**/  P, 0, ARMAS_PBOTTOM);


    while (ATL.rows > lb) {
        __repartition_2x2to3x3(&ATL,
                               &A00,   &A01, __nil,
                               __nil,  &A11, __nil,
                               __nil, __nil,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        __pivot_repart_2x1to3x1(&pT,  /**/
                                &p0, &p1, &p2,       /**/ P, lb, ARMAS_PTOP);
        // ---------------------------------------------------------------------------
        armas_x_make(&Y, ATL.rows, A01.cols, ATL.rows, armas_x_data(W));
        
        __unblk_ldlpv_upper_ncol(&ATL, &Y, &p1, A11.cols, conf);
        __partition_2x1(&Y01,
                        &Y11, /**/ &Y, A11.rows, ARMAS_PBOTTOM);

        // A00 = A00 - U01*D1*U01.T   (Y01 is U01*D1, A01 is U01)
        armas_x_update_trm(&A00, &Y01, &A01, -__ONE, __ONE, ARMAS_UPPER|ARMAS_TRANSB, conf);

        // pivot rows on right side;
        armas_x_pivot(&ATR, &p1, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_BACKWARD, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  &ATR,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
        __pivot_cont_3x1to2x1(&pT,
                              &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }
    if (ATL.rows > 0) {
        __unblk_ldlpv_upper(&ATL, &pT, conf);
        armas_x_pivot(&ATR, &pT, ARMAS_PIVOT_ROWS|ARMAS_PIVOT_BACKWARD, conf);
    }
    return err;
}

/**
 * \brief Compute pivoting LDLT factoring of symmetric matrix
 *
 * \param A
 *     On entry symmetric matrix store on lower (upper) triangular part. On exit
 *     the LDL^T (UDU^T) factorization of where L (U) is lower (upper) triangular
 *     matrix with unit diagonal and D is stored on diagonal entries.
 * \param W
 *     Working space for blocked implementation. If null or zero sized then unblocked
 *     algorithm used.
 * \param P
 *     Pivot vector. Size must be equal to rows/columns of input matrix. Non-pivoting
 *     algorithm is used if P is ARMAS_NOPIVOT.
 * \param flags
 *     Indicator bits, lower (upper) storage if ARMAS_LOWER (ARMAS_UPPER) set.
 * \param conf
 *     Configuration block
 *
 * \retval 0  ok
 * \retval -1 error
 */
int armas_x_ldlfactor(armas_x_dense_t *A, armas_x_dense_t *W,
                      armas_pivot_t *P, int flags, armas_conf_t *conf)
{
    int ws, ws_opt, lb, err = 0;
    if (!conf)
        conf = armas_conf_default();

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (P == ARMAS_NOPIVOT) {
        return __ldlfactor_np(A, W, flags, conf);
    }

    lb = conf->lb;
    ws = armas_x_size(W);
    ws_opt = __ws_opt(A->rows, lb);

    if (ws > 0 && ws < ws_opt) {
        lb = __new_lb(A->rows, lb, ws);
    }

    if (lb == 0 || ws == 0 || A->rows <= lb) {
        // unblocked code
        if (flags & ARMAS_UPPER) {
            err = __unblk_ldlpv_upper(A, P, conf);
        } else {
            err = __unblk_ldlpv_lower(A, P, conf);
        }
    } else {
        // blocked version
        if (flags & ARMAS_UPPER) {
            err = __blk_ldlpv_upper(A, W, P, lb, conf);
        } else {
            err = __blk_ldlpv_lower(A, W, P, lb, conf);
        }
    }
    return err;
}

#if defined(armas_x_ldlsolve)
/**
 * \brief Solve X = A*B or X = A.T*B where A is symmetric matrix
 *
 * \param[in,out] B
 *    On entry, input values. On exit, the solutions matrix
 * \param[in] A
 *    The LDL^T (UDU^T) factorized symmetric matrix
 * \param[in] P
 *    The pivot vector. If P is ARMAS_NOPIVOT then non-pivoting factorization is assumed.
 * \param[in] flags
 *    Indicator flags, lower (ARMAS_LOWER) or upper (ARMAS_UPPER) triangular matrix
 * \param[in,out] conf
 *    Configuration block.
 *  
 * \retval  0 ok
 * \retval -1 error
 */
int armas_x_ldlsolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf)
{
    int pivot1_dir, pivot2_dir;
    if (!conf)
        conf = armas_conf_default();
    
    if (armas_x_size(B) == 0 || armas_x_size(A) == 0)
        return 0;

    if (P == ARMAS_NOPIVOT) 
        return __ldlsolve_np(B, A, flags, conf);
    
    if (A->rows != A->cols || A->cols != B->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (A->rows != P->npivots) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    pivot1_dir = flags & ARMAS_UPPER ? ARMAS_PIVOT_BACKWARD : ARMAS_PIVOT_FORWARD;
    pivot2_dir = flags & ARMAS_UPPER ? ARMAS_PIVOT_FORWARD : ARMAS_PIVOT_BACKWARD;

    armas_x_pivot(B, P, ARMAS_PIVOT_ROWS|pivot1_dir, conf);

    if (flags & ARMAS_TRANS) {
        // X = L.-1*(D.-1*(L.-T*B))
        armas_x_solve_trm(B, A, __ONE, flags|ARMAS_UNIT|ARMAS_TRANS|ARMAS_LEFT, conf);
        armas_x_solve_diag(B, A, ARMAS_LEFT, conf);
        armas_x_solve_trm(B, A, __ONE, flags|ARMAS_UNIT|ARMAS_LEFT, conf);
    } else {
        // X = L.-T*(D.-1*(L.-1*B))
        armas_x_solve_trm(B, A, __ONE, flags|ARMAS_UNIT|ARMAS_LEFT, conf);
        armas_x_solve_diag(B, A, ARMAS_LEFT, conf);
        armas_x_solve_trm(B, A, __ONE, flags|ARMAS_UNIT|ARMAS_TRANS, conf);
    }

    armas_x_pivot(B, P, ARMAS_PIVOT_ROWS|pivot2_dir, conf);

    return 0;
}

#endif

#if defined(armas_x_ldlfactor_ws)
/**
 * \brief Compute worksize for blocked factorization.
 */
int armas_x_ldlfactor_ws(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
    return __ws_opt(A->rows, conf->lb);
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

