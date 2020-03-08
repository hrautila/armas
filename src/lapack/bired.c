
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Bidiagonal reduction

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bdreduce) && defined(armas_x_bdreduce_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_householder) && defined(armas_x_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

static inline
int ws_bdreduce(int M, int N, int lb)
{
    return lb <= 0 ? M + N : lb * (M + N);
}

/*
 * (1) G.Van Zee, R. van de Geijn
 *       Algorithms for Reducing a Matrix to Condensed Form
 *       2010, Flame working note #53 
 */

/*
 * Basic bidiagonal reduction for one column and row computing
 * from left to right when M >= N.
 *
 *   A  = ( 1 - tauq*u*u.T ) * A * ( 1 - taup*v*v.T )
 *
 *      = ( 1 - tauq*( 1 )*( 1 u.T )) * A * (1 - taup* ( 0 )*( 0 v.T ))
 *                   ( u )                             ( v )
 *
 *  1. Compute first Householder reflector [1 u].T,  len(u) == len(a21)
 *
 *     a12'  = (1 - tauq*( 1 )(1 u.T))( a12 )
 *     A22'              ( u )        ( A22 )
 *
 *           = ( a12 -  tauq * (a12 + u.T*A22) )
 *             ( A22 -  tauq*v*(a12 + u.T*A22) )
 *
 *           = ( a12 - tauq * y21 )    [y21 = a12 + A22.T*u]
 *             ( A22 - tauq*v*y21 )
 *
 *  2. Compute second Householder reflector [v], len(v) == len(a12)
 *
 *      A''  = ( a21; A22' )(1 - taup*( 0 )*( 0 v.T ))
 *                                    ( v )
 *
 *           = ( a21; A22' ) - taup * ( 0;  A22'*v*v.T )
 *
 *     A22   = A22' - taup*A22'*v*v.T
 *           = A22  - tauq*u*y21 - taup*(A22 - tauq*u*y21)*v*v.T
 *           = A22  - tauq*u*y21 - taup*(A22*v - tauq*u*y21*v)*v.T
 *
 *     y21   = a12 + A22.T*u
 *     z21   = A22*v - tauq*u*y21*v
 *
 * The unblocked algorithm
 * -----------------------
 *  1.  u, tauq := HOUSEV(a11, a21)
 *  2.  y21 := a12 + A22.T*u
 *  3.  a12 := a12 - tauq*u
 *  4.  v, taup := HOUSEV(a12)
 *  5.  beta := DOT(v, y21)
 *  6.  z21  := A22*v - tauq*beta*u
 *  7.  A22  := A22 - tauq*u*y21
 *  8.  A22  := A22 - taup*z21*v
 */

/*
 * Compute unblocked bidiagonal reduction for A when M >= N
 *
 * Diagonal and first super/sub diagonal are overwritten with the 
 * upper/lower bidiagonal matrix B.
 *
 * This computing (1-tauq*v*v.T)*A*(1-taup*u.u.T) from left to right. 
 */
static
int unblk_bdreduce_left(armas_x_dense_t * A, armas_x_dense_t * tauq,
                        armas_x_dense_t * taup, armas_x_dense_t * W,
                        armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2, y21;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2, z21;
    DTYPE v0, beta, tauqv, taupv;

    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // temp vector for this round
        armas_x_make(&y21, a12.cols, 1, a12.cols, armas_x_data(W));
        armas_x_make(&z21, a21.rows, 1, a21.rows,
                     &armas_x_data(W)[armas_x_size(&y21)]);

        // compute householder to zero subdiagonal entries
        armas_x_compute_householder(&a11, &a21, &tq1, conf);

        if (a12.cols > 0) {
            // y21 := a12 + A22.T*a21
            armas_x_axpby(ZERO, &y21, ONE, &a12, conf);
            armas_x_mvmult(ONE, &y21, ONE, &A22, &a21, ARMAS_TRANSA, conf);

            // a12 := a12 - tauq*y21
            tauqv = armas_x_get(&tq1, 0, 0);
            armas_x_axpy(&a12, -tauqv, &y21, conf);

            // compute householder to zero elements above 1st superdiagonal
            armas_x_compute_householder_vec(&a12, &tp1, conf);
            taupv = armas_x_get(&tp1, 0, 0);
            v0 = armas_x_get(&a12, 0, 0);
            armas_x_set(&a12, 0, 0, ONE);

            // [v == a12, u == a21]
            beta = armas_x_dot(&y21, &a12, conf);
            // z21 := tauq*beta*v == tauq*beta*a21
            armas_x_axpby(ZERO, &z21, tauqv * beta, &a21, conf);
            // z21 := A22*v - z21 == A22*a12 - z21
            armas_x_mvmult(-ONE, &z21, ONE, &A22, &a12, ARMAS_NONE, conf);
            // A22 := A22 - tauq*u*y21 == A22 - tauq*a21*y21
            armas_x_mvupdate(ONE, &A22, -tauqv, &a21, &y21, conf);
            // A22 := A22 - taup*z21*v == A22 - taup*z21*a12
            armas_x_mvupdate(ONE, &A22, -taupv, &z21, &a12, conf);

            armas_x_set(&a12, 0, 0, v0);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    }
    return 0;
}

/*
 * This is adaptation of BIRED_LAZY_UNB algorithm from (1).
 *
 * Y is workspace for building updates for first Householder.
 * And Z is space for building updates for second Householder
 * Y is n(A)-2,nb and Z is m(A)-1,nb  
 */
static
int unblk_bdbuild_left(armas_x_dense_t * A, armas_x_dense_t * tauq,
                         armas_x_dense_t * taup, armas_x_dense_t * Y,
                         armas_x_dense_t * Z, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a01, A02, a10, a11, a12, A20, a21, A22;
    armas_x_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
    armas_x_dense_t ZTL, ZBR, Z00, z10, z11, Z20, z21, Z22;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2, w00;
    DTYPE beta, tauqv, taupv, aa, v0 = ZERO;
    int k;

    EMPTY(A00);
    EMPTY(ATR);
    EMPTY(y11);
    EMPTY(z11);
    EMPTY(Z00);
    EMPTY(Y00);

    mat_partition_2x2(
        &ATL, &ATR,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &YTL, __nil,
        __nil, &YBR, /**/ Y, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &ZTL, __nil,
        __nil, &ZBR, /**/ Z, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);

    for (k = 0; k < Y->cols; k++) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &YTL,
            &Y00, __nil, __nil,
            &y10, &y11, __nil,
            &Y20, &y21, &Y22, /**/ Y, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &ZTL,
            &Z00, __nil, __nil,
            &z10, &z11, __nil,
            &Z20, &z21, &Z22, /**/ Z, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, 1, ARMAS_PBOTTOM);
        // --------------------------------------------------------------------
        armas_x_submatrix(&w00, Z, 0, Z->cols - 1, A20.cols, 1);
        // u10 == a10, U20 == A20, u21 == a21,
        // v10 == a01, V20 == A02, v21 == a12
        if (Y20.cols > 0) {
            // a11 = a11 - u10*y10 - z10*v10
            aa = armas_x_dot(&a10, &y10, conf);
            aa += armas_x_dot(&z10, &a01, conf);
            armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
            // a21 := a21 - U20*y10 - Z20*v10
            armas_x_mvmult(ONE, &a21, -ONE, &A20, &y10, ARMAS_NONE, conf);
            armas_x_mvmult(ONE, &a21, -ONE, &Z20, &a01, ARMAS_NONE, conf);
            // a12 := a12 - u10.T*Y20.T - z10.T*V20.T
            armas_x_mvmult(ONE, &a12, -ONE, &Y20, &a10, ARMAS_NONE, conf);
            armas_x_mvmult(ONE, &a12, -ONE, &A02, &z10, ARMAS_TRANS, conf);
            // restore bidiagonal entry
            armas_x_set(&a01, a01.rows - 1, 0, v0);
        }
        // compute householder to zero subdiagonal entries
        armas_x_compute_householder(&a11, &a21, &tq1, conf);
        tauqv = armas_x_get(&tq1, 0, 0);

        // y21 := a12 + A22.T*u21 - Y20*U20.T*u21 - V20*Z20.T*u21
        armas_x_axpby(ZERO, &y21, ONE, &a12, conf);
        armas_x_mvmult(ONE, &y21, ONE, &A22, &a21, ARMAS_TRANS, conf);
        // w00 := U20.T*u21 [= A20.T*a21]
        armas_x_mvmult(ZERO, &w00, ONE, &A20, &a21, ARMAS_TRANS, conf);
        // y21 := y21 - U20*w00 [U20 == A20]
        armas_x_mvmult(ONE, &y21, -ONE, &Y20, &w00, ARMAS_NONE, conf);
        // w00 := Z20.T*u21
        armas_x_mvmult(ZERO, &w00, ONE, &Z20, &a21, ARMAS_TRANS, conf);
        // y21 := y21 - V20*w00  [V20 == A02.T]
        armas_x_mvmult(ONE, &y21, -ONE, &A02, &w00, ARMAS_TRANS, conf);

        // a12 := a12 - tauq*y21
        armas_x_scale(&y21, tauqv, conf);
        armas_x_axpy(&a12, -ONE, &y21, conf);

        // compute householder to zero elements above 1st superdiagonal
        armas_x_compute_householder_vec(&a12, &tp1, conf);
        taupv = armas_x_get(&tp1, 0, 0);
        v0 = armas_x_get(&a12, 0, 0);
        armas_x_set(&a12, 0, 0, ONE);

        // z21 := taup*(A22*v - U20*Y20.T*v - Z20*V20.T*v - beta*u)
        // [v == a12, u == a21]
        beta = armas_x_dot(&y21, &a12, conf);
        // z21 := beta*u
        armas_x_axpby(ZERO, &z21, beta, &a21, conf);
        // w00 = Y20.T*v
        armas_x_mvmult(ZERO, &w00, ONE, &Y20, &a12, ARMAS_TRANS, conf);
        // z21 = z21 + U20*w00
        armas_x_mvmult(ONE, &z21, ONE, &A20, &w00, ARMAS_NONE, conf);
        // w00 := V20.T*v  (V20.T == A02)
        armas_x_mvmult(ZERO, &w00, ONE, &A02, &a12, ARMAS_NONE, conf);
        // z21 := z21 + Z20*w00
        armas_x_mvmult(ONE, &z21, ONE, &Z20, &w00, ARMAS_NONE, conf);

        // z21 := -taup*z21 + taup*A22*v
        armas_x_mvmult(-taupv, &z21, taupv, &A22, &a12, ARMAS_NONE, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &YTL, __nil,
            __nil, &YBR, /**/ &Y00, &y11, &Y22, /**/ Y, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &ZTL, __nil,
            __nil, &ZBR, /**/ &Z00, &z11, &Z22, /**/ Z, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    }
    // restore bidiagonal entry
    armas_x_set(&ATR, ATR.rows - 1, 0, v0);
    return 0;
}


static
int blk_bdreduce_left(armas_x_dense_t * A,
                      armas_x_dense_t * tauq, armas_x_dense_t * taup,
                      armas_x_dense_t * W, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A21, A22;
    armas_x_dense_t Y, YT, YB, Y0, Y1, Y2;
    armas_x_dense_t Z, ZT, ZB, Z0, Z1, Z2;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2;
    DTYPE v0;

    EMPTY(A00);
    EMPTY(A11);

    armas_x_make(&Z, A->rows, lb, A->rows, armas_x_data(W));
    armas_x_make(&Y, A->cols, lb, A->cols, &armas_x_data(W)[armas_x_size(&Z)]);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &YT,
        &YB, /**/ &Y, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &ZT,
        &ZB, /**/ &Z, 0, ARMAS_PTOP);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &YT,
            &Y0, &Y1, &Y2, /**/ &Y, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &ZT,
            &Z0, &Z1, &Z2, /**/ &Z, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        unblk_bdbuild_left(&ABR, &tq1, &tp1, &YB, &ZB, conf);

        // set superdiagonal entry to one
        v0 = armas_x_get(&A12, A12.rows - 1, 0);
        armas_x_set(&A12, A12.rows - 1, 0, ONE);

        // A22 := A22 - U2*Y2.T
        armas_x_mult(ONE, &A22, -ONE, &A21, &Y2, ARMAS_TRANSB, conf);
        // A22 := A22 - Z2*V2.T
        armas_x_mult(ONE, &A22, -ONE, &Z2, &A12, ARMAS_NONE, conf);

        // restore super-diagonal entry
        armas_x_set(&A12, A12.rows - 1, 0, v0);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &YT,
            &YB, /**/ &Y0, &Y1, /**/ &Y, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &ZT,
            &ZB, /**/ &Z0, &Z1, /**/ &Z, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        unblk_bdreduce_left(&ABR, &tqB, &tpB, W, conf);
    }
    return 0;
}

/*
 *  Computing transformation from right to left.
 *
 *  1. Compute first Householder reflector [1 u], len(u) == len(a12)
 *
 *    A'  = ( a21 A22 )(1 - taup*( 1 )*( 1 u.T ))
 *                               ( u )
 *
 *        = ( a21 A22 ) - taup*(a21 + A22*u; a21*u.T + A22*u*u.T)
 *
 *        = ( a21 - taup*(a21 + A22*u); A22 - taup*(a21 + A22*u)*u.T)
 *
 *          y21  = a21 + A22*u
 *          a21' = a21 - taup*y21
 *          A22' = A22 - taup*y21*u.T
 *
 *  2. Compute second Householder reflector [v],  len(v) == len(a21)
 *
 *    a12'' = (1 - tauq*( 0 )(0 v.T))( a12' )
 *    A22''             ( v )        ( A22' )
 *
 *          = ( a12'                    )
 *            ( A22' -  tauq*v*v.T*A22' )
 *
 *    A22   = A22' - tauq*v*v.T*A22'
 *          = A22  - taup*y21*u.T - tauq*v*v.T*(A22 - taup*y21*u.T)
 *          = A22  - taup*y21*u.T - tauq*v*v.T*A22 + tauq*v*v.T*taup*y21*u.T
 *          = A22  - taup*y21*u.T - tauq*v*(A22.T*v - taup*v.T*y21*u.T)
 *
 *  The unblocked algorithm
 *  -----------------------
 *   1.  u, taup = HOUSEV(a11, a12)
 *   2.  y21  = a21 + A22*u
 *   3.  a21' = a21 - taup*y21
 *   4.  v, tauq = HOUSEV(a21')
 *   5.  beta = DOT(y21, v)
 *   6.  z21  = A22.T*v - taup*beta*u.T
 *   7.  A22  = A22 - taup*y21*u.T
 *   8.  A22  = A22 - tauq*v*z21
 */


/*
 * Compute unblocked bidiagonal reduction for A when M < N
 *
 * Diagonal and first sub diagonal are overwritten with the lower
 * bidiagonal matrix B.
 */
static
int unblk_bdreduce_right(armas_x_dense_t * A, armas_x_dense_t * tauq,
                           armas_x_dense_t * taup, armas_x_dense_t * W,
                           armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2, y21;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2, z21;
    DTYPE v0, beta, tauqv, taupv;

    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // temp vector for this round
        armas_x_make(&y21, a21.rows, 1, a21.rows, armas_x_data(W));
        armas_x_make(&z21, a12.cols, 1, a12.cols,
                     &armas_x_data(W)[armas_x_size(&y21)]);

        // compute householder to zero superdiagonal entries
        armas_x_compute_householder(&a11, &a12, &tp1, conf);

        if (a21.rows > 0) {
            // y21 := a21 + A22.T*a12
            armas_x_axpby(ZERO, &y21, ONE, &a21, conf);
            armas_x_mvmult(ONE, &y21, ONE, &A22, &a12, ARMAS_NONE, conf);

            // a21 := a21 - taup*y21
            taupv = armas_x_get(&tp1, 0, 0);
            armas_x_axpy(&a21, -taupv, &y21, conf);

            // compute householder to zero elements below 1st subdiagonal
            armas_x_compute_householder_vec(&a21, &tq1, conf);
            tauqv = armas_x_get(&tq1, 0, 0);

            v0 = armas_x_get(&a21, 0, 0);
            armas_x_set(&a21, 0, 0, ONE);

            // [v == a21, u == a12]
            beta = armas_x_dot(&y21, &a21, conf);
            // z21 := taup*beta*a12
            armas_x_axpby(ZERO, &z21, taupv * beta, &a12, conf);
            // z21 := A22*a21 - z21
            armas_x_mvmult(-ONE, &z21, ONE, &A22, &a21, ARMAS_TRANS, conf);
            // A22 := A22 - taup*y21*a12
            armas_x_mvupdate(ONE, &A22, -taupv, &y21, &a12, conf);
            // A22 := A22 - tauq*z21*a21
            armas_x_mvupdate(ONE, &A22, -tauqv, &a21, &z21, conf);

            armas_x_set_unsafe(&a21, 0, 0, v0);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    }
    return 0;
}


/*
 * This is adaptation of BIRED_LAZY_UNB algorithm from (1).
 *
 * Z matrix accumulates updates of row transformations i.e. first
 * Householder that zeros off diagonal entries on row. Vector z21
 * is updates for current round, Z20 are already accumulated updates.
 * Vector z21 updates a12 before next transformation.
 *
 * Y matrix accumulates updates on column tranformations ie Householder
 * that zeros elements below sub-diagonal. Vector y21 is updates for current
 * round, Y20 are already accumulated updates.  Vector y21 updates
 * a21 befor next transformation.
 *
 * Z, Y matrices upper trigonal part is not needed, temporary vector
 * w00 that has maximum length of n(Y) is placed on the last column of
 * Z matrix on each iteration.
 */
static
int unblk_bdbuild_right(armas_x_dense_t * A, armas_x_dense_t * tauq,
                          armas_x_dense_t * taup, armas_x_dense_t * Y,
                          armas_x_dense_t * Z, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, a01, A02, a10, a11, a12, A20, a21, A22;
    armas_x_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
    armas_x_dense_t ZTL, ZBR, Z00, z10, z11, Z20, z21, Z22;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2, w00;
    DTYPE beta, tauqv, taupv, aa, v0 = ZERO;
    int k;

    EMPTY(A00);
    EMPTY(ABL);
    EMPTY(y11);
    EMPTY(z11);
    EMPTY(Z00);
    EMPTY(Y00);

    mat_partition_2x2(
        &ATL, __nil,
        &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &YTL, __nil,
        __nil, &YBR, /**/ Y, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &ZTL, __nil,
        __nil, &ZBR, /**/ Z, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);

    for (k = 0; k < Y->cols; k++) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &YTL,
            &Y00, __nil, __nil,
            &y10, &y11, __nil,
            &Y20, &y21, &Y22, /**/ Y, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &ZTL,
            &Z00, __nil, __nil,
            &z10, &z11, __nil,
            &Z20, &z21, &Z22, /**/ Z, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, 1, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, 1, ARMAS_PBOTTOM);
        // --------------------------------------------------------------------
        armas_x_submatrix(&w00, Z, 0, Z->cols - 1, A02.rows, 1);
        // u10 == a10, U20 == A20, u21 == a21,
        // v10 == a01, V20 == A02, v21 == a12
        if (Y20.cols > 0) {
            // a11 = a11 - u10*y10 - z10*v10
            aa = armas_x_dot(&a10, &z10, conf);
            aa += armas_x_dot(&y10, &a01, conf);
            armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
            // a12 := a12 - V20*y10 - Z20*u10
            armas_x_mvmult(ONE, &a12, -ONE, &A02, &y10, ARMAS_TRANS, conf);
            armas_x_mvmult(ONE, &a12, -ONE, &Z20, &a10, ARMAS_NONE, conf);
            // a21 := a21 - Y20*v10 - U20*z10
            armas_x_mvmult(ONE, &a21, -ONE, &Y20, &a01, ARMAS_NONE, conf);
            armas_x_mvmult(ONE, &a21, -ONE, &A20, &z10, ARMAS_NONE, conf);
            // restore bidiagonal entry
            armas_x_set(&a10, 0, a10.cols - 1, v0);
        }
        // compute householder to zero subdiagonal entries
        armas_x_compute_householder(&a11, &a12, &tp1, conf);
        taupv = armas_x_get(&tp1, 0, 0);

        // y21 := a12 + A22*v21 - Y20*U20.T*v21 - V20*Z20.T*v21
        armas_x_axpby(ZERO, &y21, ONE, &a21, conf);
        armas_x_mvmult(ONE, &y21, ONE, &A22, &a12, ARMAS_NONE, conf);
        // w00 := U20.T*v21 [= A02*a12]
        armas_x_mvmult(ZERO, &w00, ONE, &A02, &a12, ARMAS_NONE, conf);
        // y21 := y21 - U20*w00 [U20 == A20]
        armas_x_mvmult(ONE, &y21, -ONE, &Y20, &w00, ARMAS_NONE, conf);
        // w00 := Z20.T*v21
        armas_x_mvmult(ZERO, &w00, ONE, &Z20, &a12, ARMAS_TRANS, conf);
        // y21 := y21 - V20*w00
        armas_x_mvmult(ONE, &y21, -ONE, &A20, &w00, ARMAS_NONE, conf);

        // a21 := a21 - taup*y21
        armas_x_scale(&y21, taupv, conf);
        armas_x_axpy(&a21, -ONE, &y21, conf);

        // compute householder to zero elements below 1st subdiagonal
        armas_x_compute_householder_vec(&a21, &tq1, conf);
        tauqv = armas_x_get(&tq1, 0, 0);
        v0 = armas_x_get(&a21, 0, 0);
        armas_x_set_unsafe(&a21, 0, 0, ONE);

        // z21 := tauq*(A22*y - U20*Y20.T*u - Z20*V20.T*u - beta*v)
        // [v == a12, u == a21]
        beta = armas_x_dot(&y21, &a21, conf);
        // z21 := beta*v
        armas_x_axpby(ZERO, &z21, beta, &a12, conf);
        // w00 = Y20.T*u
        armas_x_mvmult(ZERO, &w00, ONE, &Y20, &a21, ARMAS_TRANS, conf);
        // z21 = z21 + V20*w00 (V20 == A02.T)
        armas_x_mvmult(ONE, &z21, ONE, &A02, &w00, ARMAS_TRANS, conf);
        // w00 := U20.T*v  (U20.T == A20.T)
        armas_x_mvmult(ZERO, &w00, ONE, &A20, &a21, ARMAS_TRANS, conf);
        // z21 := z21 + Z20*w00
        armas_x_mvmult(ONE, &z21, ONE, &Z20, &w00, ARMAS_NONE, conf);

        // z21 := -tauq*z21 + tauq*A22*v
        armas_x_mvmult(-tauqv, &z21, tauqv, &A22, &a21, ARMAS_TRANS, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &YTL, __nil,
            __nil, &YBR, /**/ &Y00, &y11, &Y22, /**/ Y, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &ZTL, __nil,
            __nil, &ZBR, /**/ &Z00, &z11, &Z22, /**/ Z, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    }
    // restore bidiagonal entry
    armas_x_set(&ABL, 0, ABL.cols - 1, v0);
    return 0;
}

static
int blk_bdreduce_right(armas_x_dense_t * A, armas_x_dense_t * tauq,
                         armas_x_dense_t * taup, armas_x_dense_t * W,
                         int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A21, A22;
    armas_x_dense_t Y, YT, YB, Y0, Y1, Y2;
    armas_x_dense_t Z, ZT, ZB, Z0, Z1, Z2;
    armas_x_dense_t tqT, tqB, tq0, tq1, tq2;
    armas_x_dense_t tpT, tpB, tp0, tp1, tp2;
    DTYPE v0;

    EMPTY(A00);
    EMPTY(A11);

    armas_x_make(&Z, A->cols, lb, A->cols, armas_x_data(W));
    armas_x_make(&Y, A->rows, lb, A->rows, &armas_x_data(W)[armas_x_size(&Z)]);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tqT,
        &tqB, /**/ tauq, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &tpT,
        &tpB, /**/ taup, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &YT,
        &YB, /**/ &Y, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &ZT,
        &ZB, /**/ &Z, 0, ARMAS_PTOP);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tqT,
            &tq0, &tq1, &tq2, /**/ tauq, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &tpT,
            &tp0, &tp1, &tp2, /**/ taup, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &YT,
            &Y0, &Y1, &Y2, /**/ &Y, lb, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &ZT,
            &Z0, &Z1, &Z2, /**/ &Z, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        unblk_bdbuild_right(&ABR, &tq1, &tp1, &YB, &ZB, conf);

        // set superdiagonal entry to one
        v0 = armas_x_get(&A21, 0, A21.cols - 1);
        armas_x_set(&A21, 0, A21.cols - 1, 1.0);

        // A22 := A22 - U2*Z2.T
        armas_x_mult(ONE, &A22, -ONE, &A21, &Z2, ARMAS_TRANSB, conf);
        // A22 := A22 - Y2*V2.T
        armas_x_mult(ONE, &A22, -ONE, &Y2, &A12, ARMAS_NONE, conf);

        // restore super-diagonal entry
        armas_x_set(&A21, 0, A21.cols - 1, v0);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tqT,
            &tqB, /**/ &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &tpT,
            &tpB, /**/ &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &YT,
            &YB, /**/ &Y0, &Y1, /**/ &Y, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &ZT,
            &ZB, /**/ &Z0, &Z1, /**/ &Z, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        unblk_bdreduce_right(&ABR, &tqB, &tpB, W, conf);
    }
    return 0;
}



/**
 * @brief Bidiagonal reduction of general matrix
 *
 * Reduce a general M-by-N matrix A to upper or lower bidiagonal form B
 * by an ortogonal transformation \f$ A = QBP^T \f$,  \f$ B = Q^TAP \f$
 *
 *
 * @param[in,out]  A
 *     On entry, the real M-by-N matrix. On exit the upper/lower
 *     bidiagonal matrix and ortogonal matrices Q and P.
 *
 * @param[out]  tauq
 *    Scalar factors for elementary reflector forming the
 *    ortogonal matrix Q.
 *
 * @param[out]  taup
 *    Scalar factors for elementary reflector forming the
 *    ortogonal matrix P.
 *
 * @param[out]  W
 *     Workspace needed for reduction.
 *
 * @param[in,out]  conf
 *     Configuration options.
 *
 *
 * #### Details
 *
 * Matrices Q and P are products of elementary reflectors \f$ H_k \f$ and \f$ G_k \f$
 *
 * If M > N:
 *   \f$  Q = H_1 H_2 ... H_N \f$  and \f$ P = G_1 G_2 ... G_{N-1} \f$
 *
 * where \f$ H_k = 1 - tauq*u*u^T \f$ and \f$ G_k = 1 - taup*v*v^T \f$
 *
 * Elementary reflector \f$ H_k \f$ are stored on columns of A below the
 * diagonal with implicit unit value on diagonal entry. Vector 'tauq` holds
 * corresponding scalar factors. Reflector \f$ G_k \f$ are stored on rows
 * of A right of first superdiagonal with implicit unit value on superdiagonal.
 * Corresponding scalar factors are stored on vector `taup`.
 * 
 * If M < N:
 *  \f$ Q = H_1 H_2 ...H_{N-1} \f$  and \f$ P = G_1 G_2 ... G_N \f$
 *
 * where \f$ H_k = 1 - tauq*u*u^T \f$ and \f$ G_k = 1 - taup*v*v^T \f$
 *
 * Elementary reflector \f$ H_k \f$ are stored on columns of A below the first
 * sub diagonal with implicit unit value on sub diagonal entry. Vector `tauq`
 * holds corresponding  scalar factors. Reflector \f$ G_k \f$ are stored on
 * rows of A right of diagonal with implicit unit value on superdiagonal.
 * Corresponding scalar factors are stored on vector `taup`.
 *
 * Contents of matrix A after reductions are as follows.
 *
 *      M = 6 and N = 5:                  M = 5 and N = 6:
 *
 *      (  d   e   v1  v1  v1 )           (  d   v1  v1  v1  v1  v1 )
 *      (  u1  d   e   v2  v2 )           (  e   d   v2  v2  v2  v2 )
 *      (  u1  u2  d   e   v3 )           (  u1  e   d   v3  v3  v3 )
 *      (  u1  u2  u3  d   e  )           (  u1  u2  e   d   v4  v4 )
 *      (  u1  u2  u3  u4  d  )           (  u1  u2  u3  e   d   v5 )
 *      (  u1  u2  u3  u4  u5 )
 *
 *  G.Van Zee, R. van de Geijn, 
 *       Algorithms for Reducing a Matrix to Condensed Form
 *       2010, Flame working note #53 
 * \ingroup lapack
 */
int armas_x_bdreduce(armas_x_dense_t * A,
                     armas_x_dense_t * tauq,
                     armas_x_dense_t * taup, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_bdreduce_w(A, tauq, taup, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_x_bdreduce_w(A, tauq, taup, wbs, conf);
    armas_wrelease(&wb);
    return err;
}


/**
 * @brief Bidiagonal reduction of general matrix
 *
 * Reduce a general M-by-N matrix A to upper or lower bidiagonal form B
 * by an ortogonal transformation \f$ A = QBP^T \f$,  \f$ B = Q^TAP \f$
 *
 *
 * @param[in,out]  A
 *     On entry, the real M-by-N matrix. On exit the upper/lower
 *     bidiagonal matrix and ortogonal matrices Q and P.
 *
 * @param[out]  tauq
 *    Scalar factors for elementary reflector forming the
 *    ortogonal matrix Q.
 *
 * @param[out]  taup
 *    Scalar factors for elementary reflector forming the
 *    ortogonal matrix P.
 *
 * @param[out]  wb
 *     Workspace needed for reduction.
 *
 * @param[in,out]  conf
 *     Configuration options.
 *
 * See armas_x_bdreduce().
 *
 * @ingroup lapack
 */
int armas_x_bdreduce_w(armas_x_dense_t * A,
                       armas_x_dense_t * tauq,
                       armas_x_dense_t * taup,
                       armas_wbuf_t * wb, armas_conf_t * conf)
{

    armas_x_dense_t W;
    armas_env_t *env;
    size_t wsmin, wsz = 0;
    int lb;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->cols > env->lb && A->rows > env->lb)
            wb->bytes = (A->cols + A->rows) * env->lb * sizeof(DTYPE);
        else
            wb->bytes = (A->cols + A->rows) * sizeof(DTYPE);
        return 0;
    }

    lb = env->lb;
    wsmin = (A->cols + A->rows) * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // adjust blocking factor for workspace
    if (lb > 0 && A->rows > lb && A->cols > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < (A->rows + A->cols) * lb) {
            lb = (wsz / (A->rows + A->cols)) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    buf = (DTYPE *) armas_wptr(wb);
    wsz = armas_wbytes(wb) / sizeof(DTYPE);
    armas_x_make(&W, wsz, 1, wsz, buf);

    wsz = armas_wpos(wb);
    if (A->rows >= A->cols) {
        if (lb > 0 && A->cols > lb) {
            blk_bdreduce_left(A, tauq, taup, &W, lb, conf);
        } else {
            unblk_bdreduce_left(A, tauq, taup, &W, conf);
        }
    } else {
        if (lb > 0 && A->cols > lb) {
            blk_bdreduce_right(A, tauq, taup, &W, lb, conf);
        } else {
            unblk_bdreduce_right(A, tauq, taup, &W, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
