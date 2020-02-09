
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdreduce) && defined(armas_x_trdreduce_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_householder) && defined(armas_x_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"
//! \endcond

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

/*
 * (1) G.Van Zee, R. van de Geijn
 *       Algorithms for Reducing a Matrix to Condensed Form
 *       2010, Flame working note #53 
 */

/*
 * Tridiagonal reduction of LOWER triangular symmetric matrix, zero elements below 1st
 * subdiagonal:
 *
 *   A =  (1 - tau*u*u.t)*A*(1 - tau*u*u.T)
 *     =  (I - tau*( 0   0   )) (a11 a12) (I - tau*( 0  0   ))
 *        (        ( 0  u*u.t)) (a21 A22) (        ( 0 u*u.t))
 *
 *  a11, a12, a21 not affected
 *
 *  from LEFT:
 *    A22 = A22 - tau*u*u.T*A22
 *  from RIGHT:
 *    A22 = A22 - tau*A22*u.u.T
 *
 *  LEFT and RIGHT:
 *    A22   = A22 - tau*u*u.T*A22 - tau*(A22 - tau*u*u.T*A22)*u*u.T
 *          = A22 - tau*u*u.T*A22 - tau*A22*u*u.T + tau*tau*u*u.T*A22*u*u.T
 *    [x    = tau*A22*u (vector)]  (SYMV)
 *    A22   = A22 - u*x.T - x*u.T + tau*u*u.T*x*u.T
 *    [beta = tau*u.T*x (scalar)]  (DOT)
 *          = A22 - u*x.T - x*u.T + beta*u*u.T
 *          = A22 - u*(x - 0.5*beta*u).T - (x - 0.5*beta*u)*u.T
 *    [w    = x - 0.5*beta*u]      (AXPY)
 *          = A22 - u*w.T - w*u.T  (SYR2)
 *
 * Result of reduction for N = 5:
 *    ( d  .  .  . . )
 *    ( e  d  .  . . )
 *    ( v1 e  d  . . )
 *    ( v1 v2 e  d . )
 *    ( v1 v2 v3 e d )
 */
static
int unblk_trdreduce_lower(armas_x_dense_t * A, armas_x_dense_t * tauq,
                          armas_x_dense_t * W, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a21, A22;
    armas_x_dense_t tT, tB, tq0, tq1, tq2, y21;
    DTYPE v0, beta, tauval;

    EMPTY(A00);
    EMPTY(a11);

    if (armas_x_size(tauq) == 0)
        return 0;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        // --------------------------------------------------------------------
        // temp vector for this round
        armas_x_make(&y21, A22.rows, 1, A22.rows, armas_x_data(W));

        // compute householder to zero subdiagonal entries
        armas_x_compute_householder_vec(&a21, &tq1, conf);
        tauval = armas_x_get(&tq1, 0, 0);

        // set subdiagonal to unit
        v0 = armas_x_get(&a21, 0, 0);
        armas_x_set(&a21, 0, 0, 1.0);

        // y21 := tauq*A22*a21
        armas_x_mvmult_sym(ZERO, &y21, tauval, &A22, &a21, ARMAS_LOWER, conf);
        // beta := tauq*a21.T*y21
        beta = tauval * armas_x_dot(&a21, &y21, conf);
        // y21 := y21 - 0.5*beta*a21
        armas_x_axpy(&y21, -HALF * beta, &a21, conf);
        // A22 := A22 - a21*y21.T - y21*a21.T
        armas_x_mvupdate2_sym(ONE, &A22, -ONE, &a21, &y21, ARMAS_LOWER, conf);
        // restore subdiagonal
        armas_x_set(&a21, 0, 0, v0);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
    }
    return 0;
}

/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
static
int unblk_trdbuild_lower(armas_x_dense_t * A, armas_x_dense_t * tauq,
                         armas_x_dense_t * Y, armas_x_dense_t * W,
                         armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
    armas_x_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
    armas_x_dense_t tT, tB, tq0, tq1, tq2, w12;
    DTYPE beta, tauval, aa, v0 = ZERO;
    int k, err = 0;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(Y00);
    EMPTY(y11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &YTL, __nil,
        __nil, &YBR, /**/ Y, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, 0, ARMAS_PTOP);

    for (k = 0; k < Y->cols; k++) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &YTL,
            &Y00, __nil, __nil,
            &y10, &y11, __nil,
            &Y20, &y21, &Y22, /**/ Y, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        armas_x_submatrix(&w12, Y, 0, 0, 1, Y00.cols);

        if (Y00.cols > 0) {
            // a11 := a11 - a10*y10 - y10*a10
            aa = armas_x_dot(&a10, &y10, conf);
            aa += armas_x_dot(&y10, &a10, conf);
            armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
            // a21 := a21 - A20*y10
            armas_x_mvmult(ONE, &a21, -ONE, &A20, &y10, ARMAS_NONE, conf);
            // a21 := a21 - Y20*a10
            armas_x_mvmult(ONE, &a21, -ONE, &Y20, &a10, ARMAS_NONE, conf);

            // restore subdiagonal value
            armas_x_set(&a10, 0, a10.cols - 1, v0);
        }
        // compute householder to zero subdiagonal entries
        armas_x_compute_householder_vec(&a21, &tq1, conf);
        tauval = armas_x_get(&tq1, 0, 0);

        // set subdiagonal to unit
        v0 = armas_x_get(&a21, 0, 0);
        armas_x_set(&a21, 0, 0, 1.0);

        // y21 := tauq*A22*a21
        armas_x_mvmult_sym(ZERO, &y21, tauval, &A22, &a21, ARMAS_LOWER, conf);
        // w12 := A20.T*a21
        armas_x_mvmult(ZERO, &w12, ONE, &A20, &a21, ARMAS_TRANS, conf);
        // y21 := y21 - tauq*Y20*(A20.T*a21)
        armas_x_mvmult(ONE, &y21, -tauval, &Y20, &w12, ARMAS_NONE, conf);
        // w12 := Y20.T*a21
        armas_x_mvmult(ZERO, &w12, ONE, &Y20, &a21, ARMAS_TRANS, conf);
        // y21 := y21 - tauq*A20*(Y20.T*a21)
        armas_x_mvmult(ONE, &y21, -tauval, &A20, &w12, ARMAS_NONE, conf);

        // beta := tauq*a21.T*y21
        beta = tauval * armas_x_dot(&a21, &y21, conf);
        // y21 := y21 - 0.5*beta*a21
        armas_x_axpy(&y21, -HALF * beta, &a21, conf);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &YTL, __nil,
            __nil, &YBR, /**/ &Y00, &y11, &Y22, /**/ Y, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
    }

    // restore subdiagonal value
    armas_x_set(A, ATL.rows, ATL.cols - 1, v0);
    return err;
}

static
int blk_trdreduce_lower(armas_x_dense_t * A, armas_x_dense_t * tauq,
                          armas_x_dense_t * Y, armas_x_dense_t * W,
                          int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A21, A22;
    armas_x_dense_t YT, YB, Y0, Y1, Y2;
    armas_x_dense_t tT, tB, tq0, tq1, tq2;
    DTYPE v0 = ZERO;
    int err = 0;

    EMPTY(A00);
    EMPTY(A11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &YT,
        &YB, /**/ Y, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, 0, ARMAS_PTOP);

    while (ABR.rows - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, __nil,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &YT,
            &Y0, &Y1, &Y2, /**/ Y, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        if (unblk_trdbuild_lower(&ABR, &tq1, &YB, W, conf)) {
            err = err == 0 ? -1 : err;
        }
        // set subdiagonal to unit
        v0 = armas_x_get(&A21, 0, A21.cols - 1);
        armas_x_set(&A21, 0, A21.cols - 1, 1.0);

        // A22 := A22 - A21*Y2.T - Y2*A21.T
        armas_x_update2_sym(ONE, &A22, -ONE, &A21, &Y2, ARMAS_LOWER, conf);

        // restore subdiagonal entry
        armas_x_set(&A21, 0, A21.cols - 1, v0);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &YT,
            &YB, /**/ &Y0, &Y1, /**/ Y, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
    }

    if (ABR.rows > 0) {
        unblk_trdreduce_lower(&ABR, &tB, W, conf);
    }
    return err;
}

/*
 * Reduce upper triangular matrix to tridiagonal.
 *
 * Elementary reflectors Q = H(n-1)...H(2)H(1) are stored on upper
 * triangular part of A. Reflector H(n-1) saved at column A(n) and
 * scalar multiplier to tau[n-1]. If parameter `tail` is true then
 * this function is used to reduce tail part of partially reduced
 * matrix and tau-vector partitioning is starting from last position. 
 */
static
int unblk_trdreduce_upper(armas_x_dense_t * A, armas_x_dense_t * tauq,
                          armas_x_dense_t * W, int tail, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a01, a11, A22;
    armas_x_dense_t tT, tB, tq0, tq1, tq2, y21;
    int toff;
    DTYPE v0, beta, tauval;

    EMPTY(ATL);
    v0 = ZERO;

    if (armas_x_size(tauq) == 0)
        return 0;

    toff = tail ? 0 : 1;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, toff, ARMAS_PBOTTOM);

    while (ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        // temp vector for this round
        armas_x_make(&y21, A00.cols, 1, A00.cols, armas_x_data(W));

        // compute householder to zero subdiagonal entries
        armas_x_compute_householder_rev(&a01, &tq1, conf);
        tauval = armas_x_get(&tq1, 0, 0);

        // set subdiagonal to unit
        v0 = armas_x_get(&a01, a01.rows - 1, 0);
        armas_x_set(&a01, a01.rows - 1, 0, 1.0);

        // y21 := tauq*A00*a01
        armas_x_mvmult_sym(ZERO, &y21, tauval, &A00, &a01, ARMAS_UPPER, conf);
        // beta := tauq*a01.T*y21
        beta = tauval * armas_x_dot(&a01, &y21, conf);
        // y21 := y21 - 0.5*beta*a01
        armas_x_axpy(&y21, -HALF * beta, &a01, conf);
        // A00 := A00 - a01*y21.T - y21*a01.T
        armas_x_mvupdate2_sym(ONE, &A00, -ONE, &a01, &y21, ARMAS_UPPER, conf);
        // restore subdiagonal
        armas_x_set(&a01, a01.rows - 1, 0, v0);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PTOP);
    }
    return 0;
}


/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
static
int unblk_trdbuild_upper(armas_x_dense_t * A, armas_x_dense_t * tauq,
                         armas_x_dense_t * Y, armas_x_dense_t * W,
                         armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
    armas_x_dense_t YTL, YBR, Y00, y01, Y02, y11, y12, Y22;
    armas_x_dense_t tT, tB, tq0, tq1, tq2, w12;
    DTYPE v0, beta, tauval, aa;
    int k, err = 0;

    v0 = ZERO;
    EMPTY(ATL);
    EMPTY(a11);
    EMPTY(YTL);
    EMPTY(Y00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x2(
        &YTL, __nil,
        __nil, &YBR, /**/ Y, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, 0, ARMAS_PBOTTOM);

    for (k = 0; k < Y->cols; k++) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, &A02,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x2to3x3(
            &YTL,
            &Y00, &y01, &Y02,
            __nil, &y11, &y12,
            __nil, __nil, &Y22, /**/ Y, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        armas_x_submatrix(&w12, Y, -1, 0, 1, Y02.cols);

        if (Y02.cols > 0) {
            // a11 := a11 - a12*y12 - y12*a12
            aa = armas_x_dot(&a12, &y12, conf);
            aa += armas_x_dot(&y12, &a12, conf);
            armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
            // a01 := a01 - A02*y12
            armas_x_mvmult(ONE, &a01, -ONE, &A02, &y12, ARMAS_NONE, conf);
            // a01 := a01 - Y02*a12
            armas_x_mvmult(ONE, &a01, -ONE, &Y02, &a12, ARMAS_NONE, conf);
            // restore subdiagonal value
            armas_x_set(&a12, 0, 0, v0);
        }
        // compute householder to zero superdiagonal entries
        armas_x_compute_householder_rev(&a01, &tq1, conf);
        tauval = armas_x_get(&tq1, 0, 0);

        // set superdiagonal to unit
        v0 = armas_x_get(&a01, a01.rows - 1, 0);
        armas_x_set(&a01, a01.rows - 1, 0, 1.0);

        // y01 := tauq*A00*a01
        armas_x_mvmult_sym(ZERO, &y01, tauval, &A00, &a01, ARMAS_UPPER, conf);
        // w12 := A02.T*a01
        armas_x_mvmult(ZERO, &w12, ONE, &A02, &a01, ARMAS_TRANS, conf);
        // y01 := y01 - tauq*Y02*(A02.T*a01)
        armas_x_mvmult(ONE, &y01, -tauval, &Y02, &w12, ARMAS_NONE, conf);
        // w12 := Y02.T*a01
        armas_x_mvmult(ZERO, &w12, ONE, &Y02, &a01, ARMAS_TRANS, conf);
        // y01 := y01 - tauq*A02*(Y02.T*a01)
        armas_x_mvmult(ONE, &y01, -tauval, &A02, &w12, ARMAS_NONE, conf);

        // beta := tauq*a01.T*y01
        beta = tauval * armas_x_dot(&a01, &y01, conf);
        // y01 := y01 - 0.5*beta*a01
        armas_x_axpy(&y01, -HALF * beta, &a01, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PTOPLEFT);
        mat_continue_3x3to2x2(
            &YTL, __nil,
            __nil, &YBR, /**/ &Y00, &y11, &Y22, /**/ Y, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PTOP);
    }

    // restore superdiagonal value
    armas_x_set(A, ATL.rows - 1, ATL.cols, v0);
    return err;
}



static
int blk_trdreduce_upper(armas_x_dense_t * A, armas_x_dense_t * tauq,
                        armas_x_dense_t * Y, armas_x_dense_t * W,
                        int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A01, A11, A22;
    armas_x_dense_t YT, YB, Y0, Y1, Y2;
    armas_x_dense_t tT, tB, tq0, tq1, tq2;
    DTYPE v0;
    int err = 0;

    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &YT,
        &YB, /**/ Y, 0, ARMAS_PBOTTOM);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tauq, 1, ARMAS_PBOTTOM);

    while (ATL.rows - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            __nil, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &YT,
            &Y0, &Y1, &Y2, /**/ Y, lb, ARMAS_PTOP);
        mat_repartition_2x1to3x1(
            &tT,
            &tq0, &tq1, &tq2, /**/ tauq, lb, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        if (unblk_trdbuild_upper(&ATL, &tq1, &YT, W, conf)) {
            err = err == 0 ? -1 : err;
        }
        // set superdiagonal to unit
        v0 = armas_x_get(&A01, A01.rows - 1, 0);
        armas_x_set(&A01, A01.rows - 1, 0, 1.0);

        // A00 := A00 - A01*Y0.T - Y0*A01.T
        armas_x_update2_sym(ONE, &A00, -ONE, &A01, &Y0, ARMAS_UPPER, conf);

        // restore superdiagonal entry
        armas_x_set(&A01, A01.rows - 1, 0, v0);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &YT,
            &YB, /**/ &Y0, &Y1, /**/ Y, ARMAS_PTOP);
        mat_continue_3x1to2x1(
            &tT,
            &tB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PTOP);
    }

    if (ATL.rows > 0) {
        unblk_trdreduce_upper(&ATL, &tT, W, TRUE, conf);
    }
    return err;
}

/**
 * \brief Reduce symmetric matrix to tridiagonal form by similiarity transformation A = QTQ^T
 *
 * \param[in,out]  A
 *      On entry, symmetric matrix with elemets stored in upper (lower) triangular
 *      part. On exit, diagonal and first super (sub) diagonals hold matrix T.
 *      The upper (lower) triangular part above (below) first super(sub)diagonal
 *      is used to store orthogonal matrix Q.
 *
 * \param[out] tauq
 *      Scalar coefficients of elementary reflectors.
 *
 * \param[in] flags
 *      ARMAS_LOWER or ARMAS_UPPER
 *
 * \param[in] confs
 *      Optional blocking configuration
 *
 * If LOWER, then the matrix Q is represented as product of elementary reflectors
 *
 *   \f$ Q = H_1 H_2...H_{n-1}. \f$
 *
 * If UPPER, then the matrix Q is represented as product 
 * 
 *   \f$ Q = H_{n-1}...H_2 H_1,  H_k = I - tau*v_k*v_k^T. \f$
 *
 * The contents of A on exit is as follow for N = 5.
 *
 *    LOWER                    UPPER
 *     ( d  .  .  .  . )         ( d  e  v3 v2 v1 )
 *     ( e  d  .  .  . )         ( .  d  e  v2 v1 )
 *     ( v1 e  d  .  . )         ( .  .  d  e  v1 )
 *     ( v1 v2 e  d  . )         ( .  .  .  d  e  )
 *     ( v1 v2 v3 e  d )         ( .  .  .  .  d  )
 */
int armas_x_trdreduce(armas_x_dense_t * A,
                      armas_x_dense_t * tauq, int flags, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_trdreduce_w(A, tauq, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_x_trdreduce_w(A, tauq, flags, wbs, conf);
    armas_wrelease(&wb);
    return stat;
}

/**
 * @brief Reduce symmetric matrix to tridiagonal form by similiarity transformation A = QTQ^T
 *
 * @param[in,out]  A
 *      On entry, symmetric matrix with elemets stored in upper (lower) triangular
 *      part. On exit, diagonal and first super (sub) diagonals hold matrix T.
 *      The upper (lower) triangular part above (below) first super(sub)diagonal
 *      is used to store orthogonal matrix Q.
 *
 * @param[out] tauq
 *      Scalar coefficients of elementary reflectors.
 *
 * @param[in] flags
 *      ARMAS_LOWER or ARMAS_UPPER
 *
 * @param[in] wb
 *     Workspace buffer needed for computation. To compute size of the required space call 
 *     the function with workspace bytes set to zero. Size of workspace is returned in 
 *     `wb.bytes` and no other computation or parameter size checking is done and function
 *     returns with success.
 *
 * @param[in] confs
 *      Optional blocking configuration
 *
 *  @retval 0  success
 *  @retval -1 error and `conf.error` set to last error
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if n(C) != m(A) for C*op(Q) or m(C) != m(A) for op(Q)*C
 *   - `ARMAS_EINVAL` C or A or tau is null pointer
 *   - `ARMAS_EWORK`  if workspace is less than required for unblocked computation
 *
 * #### Additional information
 *
 * If LOWER, then the matrix Q is represented as product of elementary reflectors
 *
 *   \f$ Q = H_1 H_2...H_{n-1}. \f$
 *
 * If UPPER, then the matrix Q is represented as product 
 * 
 *   \f$ Q = H_{n-1}...H_2 H_1,  H_k = I - tau*v_k*v_k^T. \f$
 *
 * The contents of A on exit is as follow for N = 5.
 *
 *    LOWER                    UPPER
 *     ( d  .  .  .  . )         ( d  e  v3 v2 v1 )
 *     ( e  d  .  .  . )         ( .  d  e  v2 v1 )
 *     ( v1 e  d  .  . )         ( .  .  d  e  v1 )
 *     ( v1 v2 e  d  . )         ( .  .  .  d  e  )
 *     ( v1 v2 v3 e  d )         ( .  .  .  .  d  )
 */
int armas_x_trdreduce_w(armas_x_dense_t * A,
                        armas_x_dense_t * tauq,
                        int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t Y, Wrk;
    armas_env_t *env;
    size_t wsmin, wsz;
    int lb, err = 0;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->cols > env->lb)
            wb->bytes = (A->cols * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = A->cols * sizeof(DTYPE);
        return 0;
    }

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (!armas_x_isvector(tauq) && armas_x_size(tauq) != A->cols) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    wsmin = A->cols * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // adjust blocking factor to workspace
    lb = env->lb;
    if (lb > 0 && A->cols > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < A->cols * lb) {
            lb = (wsz / A->cols) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->cols <= lb) {
        armas_x_make(&Wrk, A->rows, 1, A->rows, buf);
        if (flags & ARMAS_UPPER) {
            err = unblk_trdreduce_upper(A, tauq, &Wrk, FALSE, conf);
        } else {
            err = unblk_trdreduce_lower(A, tauq, &Wrk, conf);
        }
    } else {
        armas_x_make(&Y, A->rows, lb, A->rows, buf);
        // Make W = Y in folling. W is used in following subroutine only when reducing
        // last block to tridiagonal form and Y is not needed anymore
        // TODO: fix this later
        if (flags & ARMAS_UPPER) {
            err = blk_trdreduce_upper(A, tauq, &Y, &Y, lb, conf);
        } else {
            err = blk_trdreduce_lower(A, tauq, &Y, &Y, lb, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return err;
}
#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
