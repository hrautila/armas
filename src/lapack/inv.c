
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_inverse) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external functions
#if defined(__unblk_inverse_upper) && defined(__armas_blas)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

extern int __unblk_inverse_upper(__armas_dense_t *A, int flags, armas_conf_t *conf);

// general matrix inverse; 
// method: A = LU then solve  A*L = U^-1 for A^-1. This version uses fused triangular upper
// matrix inverse within the loop.
static
int __unblk_inverse_fused(__armas_dense_t *A, __armas_dense_t *W, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, a01, a11, a21, A22, l21;
    __armas_dense_t AL, AR, A0, a1, A2;
    int err = 0;
    DTYPE a11val;

    EMPTY(A0); EMPTY(a11); EMPTY(ATL); EMPTY(AL);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    __partition_1x2(&AL, &AR,      /**/  A, 0, ARMAS_PRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  &a01, __nil,
                               __nil, &a11,  __nil,
                               __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        __repartition_1x2to1x3(&AL, &A0, &a1, &A2,  /**/  A, 1, ARMAS_PLEFT);
        // ---------------------------------------------------------------------------
        // inverse upper part; 
        a11val = __armas_get_unsafe(&a11, 0, 0);
        if (a11val == __ZERO) {
            if (err == 0) {
                conf->error = ARMAS_ESINGULAR;
                err = -1;
            }
            goto next;
        }
        // fused upper triangular matrix inverse
        // 1. a11 = 1.0/a11
        a11val = __ONE/a11val;
        __armas_set_unsafe(&a11, 0, 0, a11val);
        // 2. a01 = -a11val*A00.-1*a01
        __armas_mvsolve_trm(&a01, &A00, -a11val, ARMAS_UPPER, conf);

        // inverse A; l21 := a21; a21 = 0, we can compute full column a1 of A^-1.
        __armas_submatrix(&l21, W, 0, 0, __armas_size(&a21), 1);
        __armas_copy(&l21, &a21, conf);
        __armas_scale(&a21, __ZERO, conf);

        // a1 := a1 - A2*l21
        __armas_mvmult(&a1, &A2, &l21, -__ONE, __ONE, 0, conf);
        a11val = __armas_get(&a11, 0, 0);
        // ---------------------------------------------------------------------------
    next:
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
        __continue_1x3to1x2(&AL, &AR,    /**/  &A0,  &a1,  A, ARMAS_PLEFT);
    }
    return err;
}

static
int __blk_inverse_fused(__armas_dense_t *A, __armas_dense_t *W, int lb, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, A01, A11, A21, A22, LB, L11, L21;
    __armas_dense_t AL, AR, A0, A1, A2, AB;
    int e, err = 0;

    EMPTY(A0); EMPTY(A11); EMPTY(ATL); EMPTY(AL);

    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
    __partition_1x2(&AL, &AR,      /**/  A, 0, ARMAS_PRIGHT);
    
    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  &A01, __nil,
                               __nil, &A11, __nil,
                               __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        __repartition_1x2to1x3(&AL, &A0, &A1, &A2,  /**/  A, lb, ARMAS_PLEFT);
        // ---------------------------------------------------------------------------
        // fused inverse upper triangular; 
        // A11 := A11^-1  (use internal function)
        e = __unblk_inverse_upper(&A11, 0, conf);
        if (err == 0)
            err = e;
        // A01 := A01*A11
        __armas_mult_trm(&A01, &A11, __ONE, ARMAS_RIGHT|ARMAS_UPPER, conf);
        // A01 := -A00^-1*A01
        __armas_solve_trm(&A01, &A00, - __ONE, ARMAS_LEFT|ARMAS_UPPER, conf);

        // inverse A; copy A11,A21 to workspace
        __merge2x1(&AB, &A11, &A21);
        __armas_make(&LB, AB.rows, A11.cols, AB.rows, __armas_data(W));
        __armas_mcopy(&LB, &AB);
        __partition_2x1(&L11, &L21, &LB, A11.rows, ARMAS_PTOP);
        // zero strictly lower triangular part of A11 and all of A21 
        __armas_mscale(&A11, __ZERO, ARMAS_LOWER|ARMAS_UNIT);
        __armas_mscale(&A21, __ZERO, 0);

        // A1 := A1 - A2*L21
        __armas_mult(&A1, &A2, &L21, -__ONE, __ONE, 0, conf);
        // A1 := A1*L11.-1
        __armas_solve_trm(&A1, &L11, __ONE, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
        __continue_1x3to1x2(&AL, &AR,    /**/  &A0,  &A1,  A, ARMAS_PLEFT);
    }
    return err;
}

/**
 * \brief Computes the inverse of a general NxN matrix.
 *
 * \param[in,out] A
 *      On entry, the LU factorization of matrix A = LU. On exit the inverse of A
 * \param[out] W
 *      Workspace 
 * \param[in]  P
 *      Rows pivots of LU factorization
 * \param[in]  conf
 *      Configuration block
 *
 * \retval  0 Succes
 * \retval -1 Error, error code set in conf.error
 *
 */
int __armas_inverse(__armas_dense_t *A, __armas_dense_t *W, armas_pivot_t *P, armas_conf_t *conf)
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
        err = __unblk_inverse_fused(A, W, conf);
    } else {
        err = __blk_inverse_fused(A, W, lb, conf);
    }

    if (err == 0 && P) {
        // apply row pivots
    }
    return err;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

