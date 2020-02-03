
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! General matrix factorization

//! \ingroup lapack

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_lufactor)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas2) && defined(armas_x_blas3)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "pivot.h"
//! \endcond

static
int unblk_lufactor_nopiv(armas_x_dense_t * A, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22;
    DTYPE a11val;

    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // a21 = a21/a11
        a11val = armas_x_get(&a11, 0, 0);
        armas_x_scale(&a21, ONE/a11val, conf);

        // A22 = A22 - a21*a12
        armas_x_mvupdate(ONE, &A22, -ONE, &a21, &a12, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return 0;
}

static
int blk_lufactor_nopiv(armas_x_dense_t * A, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A21, A22;

    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // A11 = LU(A11)
        unblk_lufactor_nopiv(&A11, conf);
        // A12 = trilu(A11)*A12.-1
        armas_x_solve_trm(&A12, ONE, &A11,
                          ARMAS_LEFT | ARMAS_LOWER | ARMAS_UNIT, conf);
        // A21 = A21.-1*triu(A11)
        armas_x_solve_trm(&A21, ONE, &A11, ARMAS_RIGHT | ARMAS_UPPER, conf);
        // A22 = A22 - A21*A12
        armas_x_mult(ONE, &A22, -ONE, &A21, &A12, ARMAS_NONE, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }

    // last block with unblocked 
    if (ABR.rows > 0 && ABR.cols > 0) {
        unblk_lufactor_nopiv(&ABR, conf);
    }
    return 0;
}

/*
 * Unblocked factorization with partial (row) pivoting. 'Left looking' version.
 */
static
int unblk_lufactor(armas_x_dense_t * A, armas_pivot_t * P, int offset,
                     armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a01, a10, a11, a21, A20, A22;
    armas_x_dense_t AL, AR, A0, a1, A2, aB0;
    armas_pivot_t pT, pB, p0, p1, p2;
    int pi, err = 0;
    DTYPE a11val, aa;

    EMPTY(a11);
    EMPTY(A0);
    EMPTY(AL);

    mat_partition_2x2(&ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_1x2(&AL, &AR, /**/ A, 0, ARMAS_PLEFT);
    pivot_2x1(&pT, &pB, /**/ P, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_1x2to1x3(
            &AL, &A0, &a1, &A2, /**/ A, 1, ARMAS_PRIGHT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // A. apply previous pivots and updates to current column
        armas_x_pivot_rows(&a1, &p0, ARMAS_PIVOT_FORWARD, conf);
        // a. a01 = trilu(A00)\a01
        armas_x_mvsolve_trm(&a01, ONE, &A00,
                            ARMAS_LEFT | ARMAS_LOWER | ARMAS_UNIT, conf);
        // b. a11 = a11 - a10*a01
        a11val = armas_x_get(&a11, 0, 0);
        aa = armas_x_dot(&a10, &a01, conf);
        armas_x_set(&a11, 0, 0, a11val - aa);
        // c. a21 = a21 - A20*a01
        err = armas_x_mvmult(ONE, &a21, -ONE, &A20, &a01, ARMAS_NONE, conf);
        // HERE:
        // current column has been updated with effects of earlier computations.
        // B. find pivot index on vector ( a11 ) == ABR[:,0] (1st column of ABR)
        //                               ( a21 )
        armas_x_column(&aB0, &ABR, 0);
        pi = armas_x_pivot_index(&aB0, conf);
        armas_pivot_set(&p1, 0, pi);
        armas_x_pivot_rows(&aB0, &p1, ARMAS_PIVOT_FORWARD, conf);

        // a21 = a21 / a11
        a11val = armas_x_get(&a11, 0, 0);
        if (a11val == 0.0) {
            conf->error = ARMAS_ESINGULAR;
            err = -1;
        } else {
            armas_x_scale(&a21, ONE/a11val, conf);
        }
        // apply pivots on left columns
        armas_x_pivot_rows(&ABL, &p1, ARMAS_PIVOT_FORWARD, conf);
        // pivot index to origin of matrix row numbers
        armas_pivot_set(&p1, 0, pi + ATL.rows);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_1x3to1x2(
            &AL, &AR, /**/ &A0, &a1, /**/ A, ARMAS_PRIGHT);
        pivot_cont_3x1to2x1(
            &pT, &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        // here A.rows < A.cols; handle the right columns
        armas_x_pivot_rows(&ATR, P, ARMAS_PIVOT_FORWARD, conf);
        armas_x_solve_trm(&ATR, ONE, &ATL,
                          ARMAS_LEFT | ARMAS_UNIT | ARMAS_LOWER, conf);
    }
    return err;
}


/*
 * Blocked LU factorization with row pivoting. 'Left looking' version, pivots
 * updated on current block and left blocks.
 */
static
int blk_lufactor(armas_x_dense_t * A, armas_pivot_t * P, int lb,
                   armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, A01, A10, A11, A21, A20, A22;
    armas_x_dense_t AL, AR, A0, A1, A2, AB0;
    armas_pivot_t pT, pB, p0, p1, p2;
    int k, pi, err = 0;

    EMPTY(A0);
    EMPTY(AL);
    EMPTY(AB0);

    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &AL, &AR, /**/ A, 0, ARMAS_PLEFT);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            &A10, &A11, __nil,
            &A20, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_1x2to1x3(
            &AL, &A0, &A1, &A2, /**/ A, A11.cols, ARMAS_PRIGHT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, A11.cols, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // A. apply previous pivots and updates to current column
        armas_x_pivot_rows(&A1, &p0, ARMAS_PIVOT_FORWARD, conf);
        // a. A01 = trilu(A00) \ A01
        armas_x_solve_trm(&A01, ONE, &A00,
                          ARMAS_LEFT | ARMAS_LOWER | ARMAS_UNIT, conf);
        // b. A11 = A11 - A10*A01
        armas_x_mult(ONE, &A11, -ONE, &A10, &A01, ARMAS_NONE, conf);
        // c. A21 = A21 - A20*A01
        armas_x_mult(ONE, &A21, -ONE, &A20, &A01, ARMAS_NONE, conf);
        // HERE:
        // current block has been updated with effects of earlier computations.
        // B. factor ( A11 )
        //           ( A21 )
        mat_merge2x1(&AB0, &A11, &A21);
        err = unblk_lufactor(&AB0, &p1, ATL.rows, conf);

        // apply pivots on left columns
        armas_x_pivot_rows(&ABL, &p1, ARMAS_PIVOT_FORWARD, conf);

        // shift pivot indicies to origin of matrix row numbers
        for (k = 0; k < armas_pivot_size(&p1); k++) {
            pi = armas_pivot_get(&p1, k);
            armas_pivot_set(&p1, k, pi + ATL.rows);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_1x3to1x2(
            &AL, &AR, /**/ &A0, &A1, /**/ A, ARMAS_PRIGHT);
        pivot_cont_3x1to2x1(
            &pT, &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        // here A.rows < A.cols; handle the right columns
        armas_x_pivot_rows(&ATR, P, ARMAS_PIVOT_FORWARD, conf);
        armas_x_solve_trm(&ATR, ONE, &ATL,
                          ARMAS_LEFT | ARMAS_UNIT | ARMAS_LOWER, conf);
    }
    return err;
}


/**
 * \brief LU factorization of a general MxN matrix
 *
 * Compute an LU factorization of a general M-by-N matrix using
 * partial pivoting with row interchanges.
 *
 * \param[in,out]  A
 *      On entry, the M-by-N matrix to be factored. On exit the factors
 *      L and U from factorization A = P*L*U, the unit diagonal elements
 *      of L are not stored.
 * \param[out]  P
 *      On exit the pivot indices. If `null` then factorization computed without
 *      pivoting.
 * \param[in,out]  conf
 *      Blocking configuration.
 *
 * \retval 0  success 
 * \retval -1 error and `conf.error` holds last error code.
 *
 * Compatible with lapack.xGETRF
 */
int armas_x_lufactor(armas_x_dense_t * A, armas_pivot_t * P,
                     armas_conf_t * cf)
{
    int err = 0;
    if (!cf)
        cf = armas_conf_default();

    armas_env_t *env = armas_getenv();
    if (env->lb == 0 || A->cols <= env->lb || A->rows <= env->lb) {
        if (P) {
            err = unblk_lufactor(A, P, 0, cf);
        } else {
            err = unblk_lufactor_nopiv(A, cf);
        }
    } else {
        if (P) {
            err = blk_lufactor(A, P, env->lb, cf);
        } else {
            err = blk_lufactor_nopiv(A, env->lb, cf);
        }
    }
    return err;
}

/**
 * \brief Solves system of linear equations
 *
 * Solve a system of linear equations \f$ AX = B \f$ or \f$ A^TX = B \f$
 * with general N-by-N* matrix A using the LU factorization computed
 * by `lufactor()`.
 *
 * \param[in,out] B
 *      On entry, the right hand side matrix B. On exit, the solution matrix X.
 * \param[in] A
 *      The factor L and U from the factorization A = P*L*U as computed
 *      by `lufactor()`.
 * \param[in] P
 *      The pivot indices from `lufactor()`, if NULL then no pivoting applied
 *      to matrix B.
 * \param[in] flags
 *      The indicator of the form of the system of equations.
 *      If *ARMAS_TRANS*  set then system is transposed.
 * \param[in,out] conf
 *      Blocking configuration
 *
 * \retval 0  success
 * \retval -1 error and `conf.error` holds last error code.
 *
 * Compatible with lapack.DGETRS.
 */
int armas_x_lusolve(armas_x_dense_t * B, armas_x_dense_t * A,
                    armas_pivot_t * P, int flags, armas_conf_t * cf)
{
    int ok;
    if (!cf)
        cf = armas_conf_default();

    ok = B->rows == A->cols && A->rows == A->cols;
    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
    if (P) {
        // pivots; apply to B.
        armas_x_pivot_rows(B, P, ARMAS_PIVOT_FORWARD, cf);
    }

    if (flags & ARMAS_TRANS) {
        // solve A.T*X = B; X = A.-T*B == (L.T*U.T).-1*B == U.-T*(L.-T*B)
        armas_x_solve_trm(
            B, ONE, A, ARMAS_LEFT | ARMAS_LOWER | ARMAS_UNIT | ARMAS_TRANSA, cf);
        armas_x_solve_trm(
            B, ONE, A, ARMAS_LEFT | ARMAS_UPPER | ARMAS_TRANSA, cf);
    } else {
        // solve A*X = B;  X = A.-1*B == (L*U).-1*B == U.-1*(L.-1*B)
        armas_x_solve_trm(
            B, ONE, A, ARMAS_LEFT | ARMAS_LOWER | ARMAS_UNIT, cf);
        armas_x_solve_trm(
            B, ONE, A, ARMAS_LEFT | ARMAS_UPPER, cf);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
