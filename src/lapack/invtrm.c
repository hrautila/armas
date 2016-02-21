
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_inverse_trm) && defined(__unblk_inverse_upper)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_blas) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

/**
 * These are forward versions, iteration of matrix from top left to bottom right, which
 * is the natural memory order for column major matrix.
 *
 * Alternative would be backward iteration, from bottom right to top left. For that see
 * general matrix inverse, upper triangular matrix inverse is there fused within the 
 * loop of general matrix inverse.
 */

int __unblk_inverse_upper(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, a01, a11, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(a11);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

    if (flags & ARMAS_UNIT)
        a11val =  __ONE;

    while (ABR.rows > 0 && ABR.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  &a01,  __nil,
                               __nil, &a11,  __nil,
                               __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------
        if (!(flags & ARMAS_UNIT)) {
            a11val = __armas_get_unsafe(&a11, 0, 0);
            if (a11val == __ZERO) {
                if (err == 0) {
                    conf->error = ARMAS_ESINGULAR;
                    err = -1;
                }
                goto next;
            }
            // a11 = 1.0/a11
            a11val = __ONE/a11val;
            __armas_set_unsafe(&a11, 0, 0, a11val);
        } 
        // a01 = -a11val*A00*a01
        __armas_mvmult_trm(&a01, &A00, -a11val, ARMAS_UPPER, conf);
        // ---------------------------------------------------------------------------
    next:
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

static
int __blk_inverse_upper(__armas_dense_t *A, int flags, int lb, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, A01, A11, A22;
    int err = 0;

    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  &A01,  __nil,
                               __nil, &A11,  __nil,
                               __nil, __nil, &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------
        // A01 := A00*A01
        __armas_mult_trm(&A01, &A00, __ONE, ARMAS_LEFT|ARMAS_UPPER|flags, conf);
        // A01 := -A01*A11.-1
        __armas_solve_trm(&A01, &A11, -__ONE, ARMAS_RIGHT|ARMAS_UPPER|flags, conf);
        // inv(&A11)
        if (__unblk_inverse_upper(&A11, flags, conf) != 0 && err == 0)
            err = -1;
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

static
int __unblk_inverse_lower(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, a11, a21, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(ATL); EMPTY(a11); EMPTY(A00);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);

    if (flags & ARMAS_UNIT)
        a11val =  __ONE;

    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil, &a11,  __nil,
                               __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        if (!(flags & ARMAS_UNIT)) {
            a11val = __armas_get_unsafe(&a11, 0, 0);
            if (a11val == __ZERO) {
                if (err == 0) {
                    conf->error = ARMAS_ESINGULAR;
                    err = -1;
                }
                goto next;
            }
            // a11 = 1.0/a11
            a11val = __ONE/a11val;
            __armas_set_unsafe(&a11, 0, 0, a11val);
        } 
        // a21 = -a11val*A22*a21
        __armas_mvmult_trm(&a21, &A22, -a11val, ARMAS_LOWER, conf);
        // ---------------------------------------------------------------------------
    next:
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}


static
int __blk_inverse_lower(__armas_dense_t *A, int flags, int lb, armas_conf_t *conf)
{
    __armas_dense_t ATL, ABR, A00, A11, A21, A22;
    int err = 0;

    EMPTY(A00); EMPTY(ATL);
    
    __partition_2x2(&ATL,  __nil,
                    __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  __nil, __nil,
                               __nil, &A11,  __nil,
                               __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------------
        // A21 := A22*A21
        __armas_mult_trm(&A21, &A22, __ONE, ARMAS_LEFT|ARMAS_LOWER|flags, conf);
        // A21 := -A21*A11.-1
        __armas_solve_trm(&A21, &A11, -__ONE, ARMAS_RIGHT|ARMAS_LOWER|flags, conf);
        // inv(&A11)
        if (__unblk_inverse_lower(&A11, flags, conf) != 0 && err == 0)
            err = -1;
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL,  __nil,
                            __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    }
    return err;
}

/**
 * \brief Compute inverse of a triangular matrix
 *
 * \param[in,out] A
 *    On entry the upper (lower) triangular matrix. On exit the inverse
 *    of the input matrix.
 * \param[in] flags
 *    Input matrix is upper (lower) if ARMAS_UPPER (ARMAS_LOWER) set. 
 *    And unit diagonal if ARMAS_UNIT is set.
 * \param[in,out] conf
 *    Configuration block.
 *
 * \returns
 *    0 for succes, -1 for error
 */
int __armas_inverse_trm(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
    int err = 0;
    if (!conf)
        conf = armas_conf_default();

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }    

    if (A->rows == 0)
        return 0;

    if (conf->lb == 0 || A->cols <= conf->lb) {
        if (flags & ARMAS_LOWER) {
            err = __unblk_inverse_lower(A, flags, conf);
        } else {
            err = __unblk_inverse_upper(A, flags, conf);
        }
    } else {
        if (flags & ARMAS_LOWER) {
            err = __blk_inverse_lower(A, flags, conf->lb, conf);
        } else {
            err = __blk_inverse_upper(A, flags, conf->lb, conf);
        }
    }
    return err;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

