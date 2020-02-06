
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_unblk_bkfactor_upper) && defined(armas_x_blk_bkfactor_upper)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "pivot.h"
#include "internal_lapack.h"

#include "sym.h"

/*
 * Finding pivot point: (see pseudo-code in ldlbkl.c)
 *
 *   d .  r  .  .  .  .  a |
 *     d  r  .  .  .  .  a |
 *        r  r  r  r  r  r |
 *           d  .  .  .  a |
 *              d  .  .  a |
 *                 d  .  a |
 *                    d  a |
 *                       d |
 *    ----------------------
 *               (ABR)
 */
static
int find_bkpivot_upper(armas_x_dense_t * A, int *nr, int *np,
                       armas_conf_t * conf)
{
    armas_x_dense_t rcol, qrow;
    DTYPE amax, rmax, qmax, qmax2;
    int r, q, lastcol;

    if (A->rows == 1) {
        *nr = -1;
        *np = 1;
        return 0;
    }
    lastcol = A->rows - 1;
    amax = ABS(armas_x_get(A, lastcol, lastcol));
    // column above diagonal on [lastcol, lastcol]
    armas_x_submatrix(&rcol, A, 0, lastcol, lastcol, 1);
    r = armas_x_iamax(&rcol, conf);
    // max off-diagonal on first column at index r
    rmax = ABS(armas_x_get(A, r, lastcol));
    if (amax >= bkALPHA * rmax) {
        // no pivoting, 1x1 diagonal
        *nr = -1;
        *np = 1;
        return 0;
    }
    // max off-diagonal on r'th row at index q
    qmax = 0.0;
    if (r > 0) {
        armas_x_submatrix(&qrow, A, 0, r, r, 1);
        q = armas_x_iamax(&qrow, conf);
        qmax = ABS(armas_x_get(A, q, r));
    }
    // elements right of diagonal
    armas_x_submatrix(&qrow, A, r, r + 1, 1, lastcol - r);
    q = armas_x_iamax(&qrow, conf);
    qmax2 = ABS(armas_x_get(&qrow, 0, q));
    if (qmax2 > qmax)
        qmax = qmax2;


    if (amax >= bkALPHA * rmax * (rmax / qmax)) {
        // no pivoting, 1x1 diagonal
        *nr = -1;
        *np = 1;
        return 0;
    }
    rmax = ABS(armas_x_get(A, r, r));
    if (rmax >= bkALPHA * qmax) {
        // 1x1 pivoting, interchange with k, r
        *nr = r;
        *np = 1;
        return 1;
    }
    // 2x2 pivoting, interchange with k+1, r
    *nr = r;
    *np = 2;
    return 2;
}

/*
 * Unblocked Bunch-Kauffman LDL factorization.
 *
 * Corresponds lapack.DSYTF2
 */
int armas_x_unblk_bkfactor_upper(armas_x_dense_t * A, armas_x_dense_t * W,
                                 armas_pivot_t * P, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABR, A00, a01, a11, A22;
    armas_x_dense_t a11inv, cwrk;
    armas_pivot_t pT, pB, p0, p1, p2;
    DTYPE t, a11val, a, b, d, scale;
    DTYPE abuf[4];
    int nc, r, np, nr, pi;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, &ATR, __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, ARMAS_PBOTTOM);

    // permanent working space for symmetric inverse of 2x2 a11
    //armas_x_submatrix(&a11inv, W, 0, W->cols-2, 2, 2);
    armas_x_make(&a11inv, 2, 2, 2, abuf);
    armas_x_set(&a11inv, 0, 1, -1.0);
    armas_x_set(&a11inv, 1, 0, -1.0);

    nc = 0;
    while (ATL.cols > 0) {

        nr = ATL.rows - 1;
        find_bkpivot_upper(&ATL, &r, &np, conf);
        if (r != -1) {
            // pivoting needed, do swaping here
            apply_bkpivot_upper(&ATL, ATL.rows - np, r, conf);
            if (np == 2) {
                /*          [r, r] | [r ,nr]
                 * a11 ==   ----------------  2-by-2 pivot
                 *          [nr,r] | [nr,nr]
                 */
                t = armas_x_get(&ATL, nr - 1, nr);
                armas_x_set(&ATL, nr - 1, nr, armas_x_get(&ATL, r, nr));
                armas_x_set(&ATL, r, nr, t);
            }
        }
        // ---------------------------------------------------------------------
        // repartition according the pivot size
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, np, ARMAS_PTOPLEFT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, np, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        if (np == 1) {
            // A00 = A00 - a01*a01.T/a11
            a11val = armas_x_get(&a11, 0, 0);
            armas_x_mvupdate_trm(
                ONE, &A00, -ONE / a11val, &a01, &a01, ARMAS_UPPER, conf);
            // a01 = a01/a11
            armas_x_scale(&a01, ONE/a11val, conf);
            // store pivot point relative to original matrix
            pi = r == -1 ? ATL.rows : r + 1;
            armas_pivot_set(&p1, 0, pi);
        } else if (np == 2) {
            /* see comments in armas_x_unblk_bkfactor_lower() */
            a = armas_x_get(&a11, 0, 0);
            b = armas_x_get(&a11, 0, 1);
            d = armas_x_get(&a11, 1, 1);
            armas_x_set(&a11inv, 0, 0, d / b);
            armas_x_set(&a11inv, 1, 1, a / b);
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale = 1.0 / ((a / b) * (d / b) - 1.0);
            scale /= b;
            // cwrk = a01
            armas_x_submatrix(&cwrk, W, 2, 0, a01.rows, np);
            armas_x_mcopy(&cwrk, &a01, 0, conf);
            // a01 := a01*a11.-1
            armas_x_mult(0.0, &a01, scale, &cwrk, &a11inv, ARMAS_NONE, conf);
            // A00 := A00 - a01*a11.-1*a01.T = A00 - a01*cwrk.T
            armas_x_update_trm(ONE, &A00, -ONE, &a01, &cwrk,
                               ARMAS_UPPER | ARMAS_TRANSB, conf);
            // store pivot points
            pi = r + 1;
            armas_pivot_set(&p1, 0, -pi);
            armas_pivot_set(&p1, 1, -pi);
        }
        // ---------------------------------------------------------------------
        nc += np;
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PTOPLEFT);
        pivot_cont_3x1to2x1(
            &pT, &pB,    /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }
    return nc;
}

/*
 * Find diagonal pivot and build incrementaly updated block.
 *
 *    d x r2 x  x  x  c1 | x x     kp1 k | w w
 *      d r2 x  x  x  c1 | x x     kp1 k | w w
 *        r2 r2 r2 r2 c1 | x x     kp1 k | w w
 *           d  x  x  c1 | x x     kp1 k | w w
 *              d  x  c1 | x x     kp1 k | w w
 *                 d  c1 | x x     kp1 k | w w
 *                    c1 | x x     kp1 k | w w
 *   --------------------------   ------------
 *               (AL)     (AR)     (WL)   (WR)
 *
 * Matrix AL contains the unfactored part of the matrix and AR the already
 * factored columns. Matrix WR is updated values of factored part ie.
 * w(i) = l(i)d(i). Matrix WL will have updated values for next column.
 * Column WL(k) contains updated AL(c1) and WL(kp1) possible pivot row AL(r2).
 *
 * On exit, for 1x1 diagonal the rightmost column of WL (k) holds the updated
 * value of AL(c1). If pivoting this required the WL(k) holds the actual pivoted
 * column/row.
 *
 * For 2x2 diagonal blocks WL(k) holds the updated AL(c1) and WL(kp1) holds
 * actual values of pivot column/row AL(r2), without the diagonal pivots.
 */
static
int build_bkpivot_upper(armas_x_dense_t * AL, armas_x_dense_t * AR,
                        armas_x_dense_t * WL, armas_x_dense_t * WR,
                        int k, int *nr, int *np, armas_conf_t * conf)
{
    armas_x_dense_t rcol, qrow, src, wk, wkp1, wkr, wrow;
    int r, q, lc, wc, lr;
    DTYPE amax, rmax, qmax, p1;

    lc = AL->cols - 1;
    wc = WL->cols - 1;
    lr = AL->rows - 1;

    // Copy AR column 0 to WR column 0 and update with WL[0:,]
    armas_x_submatrix(&src, AL, 0, lc, AL->rows, 1);
    armas_x_submatrix(&wk, WL, 0, wc, AL->rows, 1);
    armas_x_copy(&wk, &src, conf);
    if (k > 0) {
        armas_x_submatrix(&wrow, WR, lr, 0, 1, WR->cols);
        armas_x_mvmult(ONE, &wk, -ONE, AR, &wrow, ARMAS_NONE, conf);
    }
    if (AL->rows == 1) {
        *nr = -1;
        *np = 1;
        return 0;
    }
    // amax is on-diagonal element of current column
    amax = ABS(armas_x_get(WL, lr, wc));
    // find max off-diagonal on last column
    armas_x_submatrix(&rcol, WL, 0, wc, lr, 1);
    // r is row index on WR and rmax is it abs value
    r = armas_x_iamax(&rcol, conf);
    rmax = ABS(armas_x_get(&rcol, r, 0));
    if (amax >= bkALPHA * rmax) {
        // no pivoting, 1x1 diagonal
        *nr = -1;
        *np = 1;
        return 0;
    }
    // now we need to copy row r to WL[:,wc-1] (= wkp1) and update it
    armas_x_submatrix(&wkp1, WL, 0, wc - 1, AL->rows, 1);
    if (r > 0) {
        // above diagonal part of AL
        armas_x_submatrix(&qrow, AL, 0, r, r, 1);
        armas_x_submatrix(&wkr, &wkp1, 0, 0, r, 1);
        armas_x_copy(&wkr, &qrow, conf);
    }
    armas_x_submatrix(&qrow, AL, r, r, 1, AL->rows - r);
    armas_x_submatrix(&wkr, &wkp1, r, 0, AL->rows - r, 1);
    armas_x_copy(&wkr, &qrow, conf);

    if (k > 0) {
        // update wkp1 
        armas_x_submatrix(&wrow, WR, r, 0, 1, WR->cols);
        armas_x_mvmult(ONE, &wkp1, -ONE, AR, &wrow, ARMAS_NONE, conf);
    }
    // set on-diagonal entry to zero to avoid finding it
    p1 = armas_x_get(&wkp1, r, 0);
    armas_x_set(&wkp1, r, 0, 0.0);

    // max off-diagonal on r'th column/row on at index q
    q = armas_x_iamax(&wkp1, conf);
    qmax = ABS(armas_x_get(&wkp1, q, 0));
    // restore on-diagonal entry
    armas_x_set(&wkp1, r, 0, p1);

    if (amax >= bkALPHA * rmax * (rmax / qmax)) {
        // no pivoting, 1x1 diagonal
        *nr = -1;
        *np = 1;
        return 0;
    }
    rmax = ABS(armas_x_get(WL, r, wc - 1));
    if (rmax >= bkALPHA * qmax) {
        // 1x1 pivoting, interchange with k, r
        // move pivot row in column WR[:,1] to WR[:,0]
        armas_x_submatrix(&src, WL, 0, wc - 1, AL->rows, 1);
        armas_x_submatrix(&wkp1, WL, 0, wc, AL->rows, 1);
        armas_x_copy(&wkp1, &src, conf);
        armas_x_set(&wkp1, -1, 0, armas_x_get(&src, r, 0));
        armas_x_set(&wkp1, r, 0, armas_x_get(&src, -1, 0));
        *nr = r;
        *np = 1;
        return 1;
    }
    // 2x2 pivoting, interchange with k+1, r
    *nr = r;
    *np = 2;
    return 2;
}

/*
 * Unblocked, bounded Bunch-Kauffman LDL factorization for at most ncol columns.
 * At most ncol columns are factorized and trailing matrix updates are
 * restricted to ncol columns. Also original columns are accumulated to working
 * matrix, which is used by calling blocked algorithm to update the trailing
 * matrix with BLAS3 update.
 *
 * Corresponds lapack.DLASYF
 */
static
int unblk_bkbounded_upper(armas_x_dense_t * A, armas_x_dense_t * W,
                          armas_pivot_t * P, int ncol, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a01, A02, a11, a12, A22;
    armas_x_dense_t a11inv, cwrk, w00, w01, w11;
    armas_pivot_t pT, pB, p0, p1, p2;
    DTYPE t1, tr, a11val, a, b, d, scale;
    int nc, r, np, pi;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(ATL);
    EMPTY(w00);
    EMPTY(w01);

    mat_partition_2x2(&ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    pivot_2x1(&pT, &pB, /**/ P, 0, ARMAS_PBOTTOM);

    // permanent working space for symmetric inverse of 2x2 a11
    armas_x_submatrix(&a11inv, W, W->rows - 2, 0, 2, 2);
    armas_x_set(&a11inv, 1, 0, -1.0);
    armas_x_set(&a11inv, 0, 1, -1.0);

    nc = 0;
    if (ncol > A->cols)
        ncol = A->cols;

    np = 0;
    for (nc = 0; ATL.cols > 0 && nc < ncol; nc += np) {

        mat_partition_2x2(
             &w00, &w01,
            __nil, &w11, /**/ W, nc, nc, ARMAS_PBOTTOMRIGHT);

        build_bkpivot_upper(&ATL, &ATR, &w00, &w01, ncol, &r, &np, conf);
        if (np > ncol - nc) {
            // next pivot does not fit into ncol columns, restore last column
            // and return with number of factorized columns
            return nc;
        }

        if (r != -1) {
            // pivoting needed, do swaping here
            int k = ATL.rows - np;
            apply_bkpivot_upper(&ATL, k, r, conf);
            // swap right hand rows to get correct updates (undone in caller!)
            swap_rows(&ATR, k, r, conf);
            swap_rows(&w01, k, r, conf);
            if (np == 2 && r != k) {
                /*               [r,r ] | [r,-1]
                 * w00 =a11 ==   ----------------
                 *               [-1,r] | [-1,-1]
                 */
                t1 = armas_x_get(&w00, k, -1);
                tr = armas_x_get(&w00, r, -1);
                armas_x_set(&w00, k, -1, tr);
                armas_x_set(&w00, r, -1, t1);
                // interchange diagonal entries on w00[:,-2]
                t1 = armas_x_get(&w00, k, -2);
                tr = armas_x_get(&w00, r, -2);
                armas_x_set(&w00, k, -2, tr);
                armas_x_set(&w00, r, -2, t1);
            }
        }
        // ---------------------------------------------------------------------
        // repartition accoring the pivot size
        mat_repartition_2x2to3x3(
            &ATL,
            &A00,   &a01, &A02,
            __nil,  &a11, &a12,
            __nil, __nil, &A22, /**/ A, np, ARMAS_PTOPLEFT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, np, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        armas_x_submatrix(&cwrk, &w00, 0, w00.cols - np, a01.rows, np);
        if (np == 1) {
            a11val = armas_x_get(&w00, a01.rows, w00.cols - np);
            armas_x_set(&a11, 0, 0, a11val);
            // a01 = a01/a11
            armas_x_copy(&a01, &cwrk, conf);
            armas_x_scale(&a01, ONE/a11val, conf);
            // store pivot point relative to original matrix
            pi = r == -1 ? ATL.rows : r + 1;
            armas_pivot_set(&p1, 0, pi);
        } else if (np == 2) {
            /*          a | b                       d/b | -1
             *  w00 == ------  == a11 --> a11.-1 == -------- * scale
             *          . | d                        -1 | a/b
             */
            a = armas_x_get(&w00, ATL.rows - 2, -2);
            b = armas_x_get(&w00, ATL.rows - 2, -1);
            d = armas_x_get(&w00, ATL.rows - 1, -1);
            armas_x_set(&a11inv, 0, 0, d / b);
            armas_x_set(&a11inv, 1, 1, a / b);
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale = 1.0 / ((a / b) * (d / b) - 1.0);
            scale /= b;
            // cwrk = a01
            // a01 := a01*a11.-1
            armas_x_mult(0.0, &a01, scale, &cwrk, &a11inv, ARMAS_NONE, conf);
            armas_x_set(&a11, 0, 0, a);
            armas_x_set(&a11, 0, 1, b);
            armas_x_set(&a11, 1, 1, d);
            // store pivot points
            pi = r + 1;
            armas_pivot_set(&p1, 0, -pi);
            armas_pivot_set(&p1, 1, -pi);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PTOPLEFT);
        pivot_cont_3x1to2x1(
            &pT, &pB,   /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }
    return nc;
}

int armas_x_blk_bkfactor_upper(armas_x_dense_t * A, armas_x_dense_t * W,
                               armas_pivot_t * P, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, A01, A02, A11, A12, A22;
    armas_x_dense_t cwrk, s, d;
    armas_pivot_t pT, pB, p0, p1, p2;
    int nblk, k, r, r1, rlen, np, colno;

    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, ARMAS_PBOTTOM);

    while (ATL.cols - lb > 0) {
        nblk = unblk_bkbounded_upper(&ATL, W, &pT, lb, conf);
        // ---------------------------------------------------------------------
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, &A02,
            __nil, &A11, &A12,
            __nil, __nil, &A22, /**/ A, nblk, ARMAS_PTOPLEFT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, nblk, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        // here [A11 A21] as been factorized, now update A22

        // nblk last columns in W is original A01
        armas_x_submatrix(&cwrk, W, 0, W->cols - nblk, A01.rows, nblk);
        // A00 := A00 - L01*D1*L01.T == A00 - A01*W.T
        armas_x_update_trm(ONE, &A00, -ONE, &A01, &cwrk,
                           ARMAS_UPPER | ARMAS_TRANSB, conf);

        // undo partial row pivots right of diagonal from lower level
        for (k = 0; k < nblk; k++) {
            r1 = armas_pivot_get(&p1, k);
            r = r1 < 0 ? -r1 : r1;
            np = r1 < 0 ? 2 : 1;
            colno = A00.cols + k;
            if (r == colno + 1 && r1 > 0) {
                // no pivot
                continue;
            }

            rlen = ATL.cols - colno - np;
            armas_x_submatrix(&s, &ATL, colno, colno + np, 1, rlen);
            armas_x_submatrix(&d, &ATL, r - 1, colno + np, 1, rlen);
            armas_x_swap(&d, &s, conf);
            if (r1 < 0) {
                // skip the other entry in 2x2 pivots
                k++;
            }
        }

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PTOPLEFT);
        pivot_cont_3x1to2x1(
            &pT, &pB,   /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
    }

    if (ATL.cols > 0) {
        armas_x_unblk_bkfactor_upper(&ATL, W, &pT, conf);
    }
    return 0;
}

int armas_x_unblk_bksolve_upper(armas_x_dense_t * B, armas_x_dense_t * A,
                                armas_pivot_t * P, int phase,
                                armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a01, A02, a11, a12, A22;
    armas_x_dense_t BT, BB, B0, b1, B2, Bx;
    armas_x_dense_t *Aref;
    armas_pivot_t pT, pB, p0, p1, p2;
    int aStart, aDir, bStart, bDir;
    int nc, r, np, k, pr;
    DTYPE b, apb, dpb, scale, s0, s1;

    EMPTY(ATL);
    EMPTY(A00);
    EMPTY(a11);

    np = 0;
    if (phase == 2) {
        aStart = ARMAS_PTOPLEFT;
        aDir = ARMAS_PBOTTOMRIGHT;
        bStart = ARMAS_PTOP;
        bDir = ARMAS_PBOTTOM;
        nc = 1;
        Aref = &ABR;
    } else {
        aStart = ARMAS_PBOTTOMRIGHT;
        aDir = ARMAS_PTOPLEFT;
        bStart = ARMAS_PBOTTOM;
        bDir = ARMAS_PTOP;
        nc = A->rows;
        Aref = &ATL;
    }
    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, aStart);
    mat_partition_2x1(
        &BT, &BB, /**/ B, 0, bStart);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, bStart);

    while (Aref->cols > 0) {
        r = armas_pivot_get(P, nc - 1);
        np = r < 0 ? 2 : 1;
        // ---------------------------------------------------------------------
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, &A02,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, np, aDir);
        mat_repartition_2x1to3x1(
            &BT, &B0, &b1, &B2, /**/ B, np, bDir);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, np, bDir);
        // ---------------------------------------------------------------------
        pr = armas_pivot_get(&p1, 0);
        switch (phase) {
        case 1:
            if (np == 1) {
                if (pr != nc) {
                    // swap rows on top part of B
                    swap_rows(&BT, BT.rows - 1, pr - 1, conf);
                }
                // B0 = B0 - a01*b1
                armas_x_mvupdate(ONE, &B0, -ONE, &a01, &b1, conf);
                // b1 = b1/d1
                armas_x_scale(&b1, ONE/armas_x_get(&a11, 0, 0), conf);
                nc -= 1;
            } else if (np == 2) {
                if (pr != -nc) {
                    // swap rows on top part of B
                    swap_rows(&BT, BT.rows - 2, -pr - 1, conf);
                }
                b = armas_x_get(&a11, 0, 1);
                apb = armas_x_get(&a11, 0, 0) / b;
                dpb = armas_x_get(&a11, 1, 1) / b;
                // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
                scale = apb * dpb - 1.0;
                scale *= b;
                // B0 = B0 - a01*b1
                armas_x_mult(ONE, &B0, -ONE, &a01, &b1, ARMAS_NONE, conf);
                // b1 = a11.-1*b1.T
                // (2x2 block, no function for doing this in-place)
                for (k = 0; k < b1.cols; k++) {
                    s0 = armas_x_get(&b1, 0, k);
                    s1 = armas_x_get(&b1, 1, k);
                    armas_x_set(&b1, 0, k, (dpb * s0 - s1) / scale);
                    armas_x_set(&b1, 1, k, (apb * s1 - s0) / scale);
                }
                nc -= 2;
            }
            break;

        case 2:
            if (np == 1) {
                armas_x_mvmult(ONE, &b1, -ONE, &B0, &a01, ARMAS_TRANS,
                               conf);
                if (pr != nc) {
                    // swap rows on top part of B
                    mat_merge2x1(&Bx, &B0, &b1);
                    swap_rows(&Bx, Bx.rows - 1, pr - 1, conf);
                }
                nc += 1;
            } else if (np == 2) {
                armas_x_mult(ONE, &b1, -ONE, &a01, &B0, ARMAS_TRANSA, conf);
                if (pr != -nc) {
                    // swap rows on top part of B
                    mat_merge2x1(&Bx, &B0, &b1);
                    swap_rows(&Bx, Bx.rows - 2, -pr - 1, conf);
                }
                nc += 2;
            }
            break;

        default:
            break;
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            __nil, &ABR, /**/ &A00, &a11, &A22, /**/ A, aDir);
        mat_continue_3x1to2x1(
            &BT, &BB,    /**/ &B0, &b1, /**/ B, bDir);
        pivot_cont_3x1to2x1(
            &pT, &pB,    /**/ &p0, &p1, /**/ P, bDir);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
