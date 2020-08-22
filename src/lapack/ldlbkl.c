
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_unblk_bkfactor_lower) && defined(armas_x_blk_bkfactor_lower)
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
 * Finding pivot point:
 *
 * α = (1 + sqrt(17))/8
 * λ = |a(r,1)| = max{|a(2,1)|, . . . , |a(m,1)|}
 * if λ > 0
 *     if |a(1,1)| ≥ αλ
 *         use a11 as 1-by-1 pivot
 *     else
 *         σ = |a(p,r)| = max{|a(1,r)|,..., |a(r−1,r)|, |a(r+1,r)|,..., |a(m,r)|}
 *         if |a(1,1) |σ ≥ αλ^2
 *             use a(1,1) as 1-by-1 pivot
 *         else if |a(r,r)| ≥ ασ
 *             use a(r,r) as 1-by-1 pivot
 *         else
 *                  a11 | ar1
 *             use  --------  as 2-by-2 pivot
 *                  ar1 | arr
 *         end
 *     end
 * end
 *
 *    ------------------
 *    | d
 *    | a d . . .
 *    | a . d . . .
 *    | a . . d x .
 *    | a . . . d . .
 *    | r r r r r d .    r'th row
 *    | a . . . . q d
 *    | a . . . . r . d
 */
static
int find_bkpivot_lower(armas_x_dense_t * A, int *nr, int *np,
                       armas_conf_t * conf)
{
    armas_x_dense_t rcol, qrow;
    DTYPE amax, rmax, qmax, qmax2;
    int r, q;

    if (A->rows == 1) {
        *nr = 0;
        *np = 1;
        return 0;
    }
    amax = ABS(armas_x_get(A, 0, 0));
    // column below diagonal at [0,0]
    armas_x_submatrix(&rcol, A, 1, 0, A->rows - 1, 1);
    r = armas_x_iamax(&rcol, conf) + 1;
    // max off-diagonal on first column at index r
    rmax = ABS(armas_x_get(A, r, 0));
    if (amax >= bkALPHA * rmax) {
        // no pivoting, 1x1 diagonal
        *nr = 0;
        *np = 1;
        return 0;
    }
    // max off-diagonal on r'th row at index q
    armas_x_submatrix(&qrow, A, r, 0, 1, r);
    q = armas_x_iamax(&qrow, conf);
    qmax = ABS(armas_x_get(A, r, q));
    if (r < A->rows - 1) {
        // rest of the r'th row after diagonal
        armas_x_submatrix(&qrow, A, r + 1, r, A->rows - r - 1, 1);
        q = armas_x_iamax(&qrow, conf);
        qmax2 = ABS(armas_x_get(&qrow, q, 0));
        if (qmax2 > qmax)
            qmax = qmax2;
    }

    if (amax >= bkALPHA * rmax * (rmax / qmax)) {
        // no pivoting, 1x1 diagonal
        *nr = 0;
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
int armas_x_unblk_bkfactor_lower(armas_x_dense_t * A, armas_x_dense_t * W,
                                 armas_pivot_t * P, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
    armas_x_dense_t a11inv, cwrk;
    armas_pivot_t pT, pB, p0, p1, p2;
    DTYPE t, a11val, a, b, d, scale;
    DTYPE abuf[4];
    int nc, r, np, pi;

    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(&ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    pivot_2x1(&pT, &pB, /**/ P, 0, ARMAS_PTOP);

    // permanent working space for symmetric inverse of 2x2 a11
    //armas_x_submatrix(&a11inv, W, 0, W->cols-2, 2, 2);
    armas_x_make(&a11inv, 2, 2, 2, abuf);
    armas_x_set(&a11inv, 1, 0, -1.0);
    armas_x_set(&a11inv, 0, 1, -1.0);

    nc = 0;
    while (ABR.cols > 0) {

        find_bkpivot_lower(&ABR, &r, &np, conf);
        if (r != 0 && r != np - 1) {
            // pivoting needed, do swaping here
            apply_bkpivot_lower(&ABR, np - 1, r, conf);
            if (np == 2) {
                /*         [0,0] | [r,0]
                 * a11 ==  -------------  2-by-2 pivot, swapping [1,0] and [r,0]
                 *         [r,0] | [r,r]
                 */
                t = armas_x_get(&ABR, 1, 0);
                armas_x_set(&ABR, 1, 0, armas_x_get(&ABR, r, 0));
                armas_x_set(&ABR, r, 0, t);
            }
        }
        // ---------------------------------------------------------------------
        // repartition accoring the pivot size
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, np, ARMAS_PBOTTOMRIGHT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, np, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        if (np == 1) {
            // A22 = A22 - a21*a21.T/a11
            a11val = armas_x_get(&a11, 0, 0);
            armas_x_mvupdate_trm(
                ONE, &A22, -ONE / a11val, &a21, &a21, ARMAS_LOWER, conf);
            // a21 = a21/a11
            armas_x_scale(&a21, ONE/a11val, conf);
            // store pivot point relative to original matrix
            armas_pivot_set(&p1, 0, r + ATL.rows + 1);
        } else if (np == 2) {
            /* from Bunch-Kaufmann 1977:
             *  (E2 C.T) = ( I2      0      )( E  0      )( I[n-2] E.-1*C.T )
             *  (C  B  )   ( C*E.-1  I[n-2] )( 0  A[n-2] )( 0      I2       )
             *
             *  A[n-2] = B - C*E.-1*C.T
             *
             *  E.-1 is inverse of a symmetric matrix, cannot use
             *  triangular solve. We calculate inverse of 2x2 matrix.
             *  Following is inspired by lapack.SYTF2
             *
             *      a | b      1        d | -b         b         d/b | -1
             *  inv ----- =  ------  * ------  =  ----------- * --------
             *      b | d    (ad-b^2)  -b |  a    (a*d - b^2)     -1 | a/b
             */
            a = armas_x_get(&a11, 0, 0);
            b = armas_x_get(&a11, 1, 0);
            d = armas_x_get(&a11, 1, 1);
            armas_x_set(&a11inv, 0, 0, d / b);
            armas_x_set(&a11inv, 1, 1, a / b);
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale = 1.0 / ((a / b) * (d / b) - 1.0);
            scale /= b;
            // cwrk = a21
            armas_x_submatrix(&cwrk, W, 0, 0, a21.rows, a21.cols);
            armas_x_mcopy(&cwrk, &a21, 0, conf);
            // a21 := a21*a11.-1
            armas_x_mult(0.0, &a21, scale, &cwrk, &a11inv, ARMAS_NONE, conf);
            // A22 := A22 - a21*a11.-1*a21.T = A22 - a21*cwrk.T
            armas_x_update_trm(ONE, &A22, -ONE, &a21, &cwrk,
                               ARMAS_LOWER | ARMAS_TRANSB, conf);
            // store pivot points
            pi = r + ATL.rows + 1;
            armas_pivot_set(&p1, 0, -pi);
            armas_pivot_set(&p1, 1, -pi);
        }
        // ---------------------------------------------------------------------
        nc += np;
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        pivot_cont_3x1to2x1(
            &pT, &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return nc;
}

/*
 * Find diagonal pivot and build incrementaly updated block.
 *
 *  (AL)  (AR)                   (WL)  (WR)
 *  --------------------------   ----------    k'th row in W
 *  x x | c1                     w w | k kp1
 *  x x | c1 d                   w w | k kp1
 *  x x | c1 x  d                w w | k kp1
 *  x x | c1 x  x  d             w w | k kp1
 *  x x | c1 r2 r2 r2 r2         w w | k kp1
 *  x x | c1 x  x  x  r2 d       w w | k kp1
 *  x x | c1 x  x  x  r2 x d     w w | k kp1
 *
 * Matrix AR contains the unfactored part of the matrix and AL the already
 * factored columns. Matrix WL is updated values of factored part ie.
 * w(i) = l(i)d(i). Matrix WR will have updated values for next column.
 * Column WR(k) contains updated AR(c1) and WR(kp1) possible pivot row AR(r2).
 */
static
int build_bkpivot_lower(armas_x_dense_t * AL, armas_x_dense_t * AR,
                        armas_x_dense_t * WL, armas_x_dense_t * WR,
                        int k, int *nr, int *np, armas_conf_t * conf)
{
    armas_x_dense_t rcol, qrow, src, wk, wkp1, wkr, wrow;
    int r, q;
    DTYPE amax, rmax, qmax, p1;

    // Copy AR column 0 to WR column 0 and update with WL[0:,]
    armas_x_submatrix(&src, AR, 0, 0, AR->rows, 1);
    armas_x_submatrix(&wk, WR, 0, 0, AR->rows, 1);
    armas_x_copy(&wk, &src, conf);
    if (k > 0) {
        armas_x_submatrix(&wrow, WL, 0, 0, 1, WL->cols);
        armas_x_mvmult(ONE, &wk, -ONE, AL, &wrow, ARMAS_NONE, conf);
    }
    if (AR->rows == 1) {
        *nr = 0;
        *np = 1;
        return 0;
    }

    amax = ABS(armas_x_get(WR, 0, 0));
    // find max off-diagonal on first column (= WR[:,0])
    armas_x_submatrix(&rcol, WR, 1, 0, AR->rows - 1, 1);
    // r is row index on WR and rmax is it abs value
    r = armas_x_iamax(&rcol, conf) + 1;
    rmax = ABS(armas_x_get(&rcol, r - 1, 0));
    if (amax >= bkALPHA * rmax) {
        // no pivoting, 1x1 diagonal
        *nr = 0;
        *np = 1;
        return 0;
    }
    // now we need to copy row r to WR[:,1] (= wkp1) and update it
    armas_x_submatrix(&wkp1, WR, 0, 1, AR->rows, 1);

    armas_x_submatrix(&qrow, AR, r, 0, 1, r + 1);
    armas_x_submatrix(&wkr, &wkp1, 0, 0, r + 1, 1);
    armas_x_axpby(ZERO, &wkr, ONE, &qrow, conf);
    if (r < AR->rows - 1) {
        armas_x_submatrix(&qrow, AR, r, r, AR->rows - r, 1);
        armas_x_submatrix(&wkr, &wkp1, r, 0, AR->rows - r, 1);
        armas_x_copy(&wkr, &qrow, conf);
    }
    if (k > 0) {
        // update wkp1 
        armas_x_submatrix(&wrow, WL, r, 0, 1, WL->cols);
        armas_x_mvmult(ONE, &wkp1, -ONE, AL, &wrow, ARMAS_NONE, conf);
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
        *nr = 0;
        *np = 1;
        return 0;
    }
    rmax = ABS(armas_x_get(WR, r, 1));
    if (rmax >= bkALPHA * qmax) {
        // 1x1 pivoting, interchange with k, r
        // move pivot row in column WR[:,1] to WR[:,0]
        armas_x_submatrix(&src, WR, 0, 1, AR->rows, 1);
        armas_x_submatrix(&wkp1, WR, 0, 0, AR->rows, 1);
        armas_x_copy(&wkp1, &src, conf);
        armas_x_set(&wkp1, 0, 0, armas_x_get(&src, r, 0));
        armas_x_set(&wkp1, r, 0, armas_x_get(&src, 0, 0));
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
int unblk_bkbounded_lower(armas_x_dense_t * A, armas_x_dense_t * W,
                          armas_pivot_t * P, int ncol, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
    armas_x_dense_t a11inv, cwrk, w00, w10, w11;
    armas_pivot_t pT, pB, p0, p1, p2;
    DTYPE t1, tr, a11val, a, b, d, scale;
    //DTYPE abuf[4];
    int nc, r, np, pi;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(w10);

    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, ARMAS_PTOP);

    // permanent working space for symmetric inverse of 2x2 a11
    armas_x_submatrix(&a11inv, W, 0, W->cols - 2, 2, 2);
    //armas_x_make(&a11inv, 2, 2, 2, abuf);
    armas_x_set(&a11inv, 1, 0, -1.0);
    armas_x_set(&a11inv, 0, 1, -1.0);

    nc = 0;
    if (ncol > A->cols)
        ncol = A->cols;

    while (ABR.cols > 0 && nc < ncol) {

        mat_partition_2x2(
            &w00, __nil,
            &w10, &w11, /**/ W, nc, nc, ARMAS_PTOPLEFT);

        build_bkpivot_lower(&ABL, &ABR, &w10, &w11, ncol, &r, &np, conf);
        if (np > ncol - nc) {
            // next pivot does not fit into ncol columns, restore last column
            // and return with number of factorized columns
            return nc;
        }
        if (r != 0 && r != np - 1) {
            // pivoting needed, do swaping here
            apply_bkpivot_lower(&ABR, np - 1, r, conf);
            // swap left hand rows to get correct updates (this will be undone in caller!)
            swap_rows(&ABL, np - 1, r, conf);
            swap_rows(&w10, np - 1, r, conf);
            if (np == 2) {
                /*        [0,0] | [r,0]
                 * a11 == -------------  2-by-2 pivot, swapping [1,0] and [r,0]
                 *        [r,0] | [r,r]
                 */
                t1 = armas_x_get(&w11, 1, 0);
                tr = armas_x_get(&w11, r, 0);
                armas_x_set(&w11, 1, 0, tr);
                armas_x_set(&w11, r, 0, t1);
                // interchange diagonal entries on w11[:,1]
                t1 = armas_x_get(&w11, 1, 1);
                tr = armas_x_get(&w11, r, 1);
                armas_x_set(&w11, 1, 1, tr);
                armas_x_set(&w11, r, 1, t1);
            }
        }
        // ---------------------------------------------------------------------
        // repartition accoring the pivot size
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, np, ARMAS_PBOTTOMRIGHT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, np, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        if (np == 1) {
            armas_x_submatrix(&cwrk, &w11, np, 0, a21.rows, np);
            // new a11 from w11[0,0]
            a11val = armas_x_get(&w11, 0, 0);
            armas_x_set(&a11, 0, 0, a11val);
            // a21 = a21/a11
            armas_x_axpby(ZERO, &a21, ONE, &cwrk, conf);
            armas_x_scale(&a21, ONE/a11val, conf);
            // store pivot point relative to original matrix
            armas_pivot_set(&p1, 0, r + ATL.rows + 1);
        } else if (np == 2) {
            /*
             * See comment block in armas_x_unblk_bkfactor_lower().
             */
            a = armas_x_get(&w11, 0, 0);
            b = armas_x_get(&w11, 1, 0);
            d = armas_x_get(&w11, 1, 1);
            armas_x_set(&a11inv, 0, 0, d / b);
            armas_x_set(&a11inv, 1, 1, a / b);
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale = 1.0 / ((a / b) * (d / b) - 1.0);
            scale /= b;
            // cwrk = a21
            armas_x_submatrix(&cwrk, &w11, np, 0, a21.rows, np);
            // a21 := a21*a11.-1
            armas_x_mult(0.0, &a21, scale, &cwrk, &a11inv, ARMAS_NONE, conf);
            armas_x_set(&a11, 0, 0, a);
            armas_x_set(&a11, 1, 0, b);
            armas_x_set(&a11, 1, 1, d);
            // store pivot points
            pi = r + ATL.rows + 1;
            armas_pivot_set(&p1, 0, -pi);
            armas_pivot_set(&p1, 1, -pi);
        }
        // ---------------------------------------------------------------------
        nc += np;
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        pivot_cont_3x1to2x1(
            &pT, &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }
    return nc;
}


int armas_x_blk_bkfactor_lower(armas_x_dense_t * A, armas_x_dense_t * W,
                               armas_pivot_t * P, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, A10, A11, A20, A21, A22;
    armas_x_dense_t cwrk, s, d;
    armas_pivot_t pT, pB, p0, p1, p2;
    int nblk, k, r, r1, rlen, n;

    EMPTY(A00);
    EMPTY(A11);

    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    pivot_2x1(
        &pT, &pB, /**/ P, 0, ARMAS_PTOP);

    while (ABR.cols - lb > 0) {

        nblk = unblk_bkbounded_lower(&ABR, W, &pB, lb, conf);
        // ---------------------------------------------------------------------
        // repartition accoring the pivot size
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &A10, &A11, __nil,
            &A20, &A21, &A22, /**/ A, nblk, ARMAS_PBOTTOMRIGHT);
        pivot_repart_2x1to3x1(
            &pT, &p0, &p1, &p2, /**/ P, nblk, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // here [A11 A21] as been factorized, now update A22

        // nblk first columns in W is original A21; A21 is A21*D1
        armas_x_submatrix(&cwrk, W, nblk, 0, A21.rows, nblk);
        // A22 := A22 - L21*D1*L21.T == A22 - A21*W.T
        armas_x_update_trm(ONE, &A22, -ONE, &A21, &cwrk,
                           ARMAS_LOWER | ARMAS_TRANSB, conf);

        // undo partial row pivots left of diagonal from lower level
        for (k = nblk; k > 0; k--) {
            r1 = armas_pivot_get(&p1, k - 1);
            r = r1 < 0 ? -r1 : r1;
            rlen = r1 < 0 ? k - 2 : k - 1;
            if (r == k && r1 > 0) {
                // no pivot
                continue;
            }

            armas_x_submatrix(&s, &ABR, k - 1, 0, 1, rlen);
            armas_x_submatrix(&d, &ABR, r - 1, 0, 1, rlen);
            armas_x_swap(&d, &s, conf);
            if (r1 < 0) {
                // skip the other entry in 2x2 pivots
                k--;
            }
        }

        // shift pivots in this block to origin of A
        for (k = 0; k < nblk; k++) {
            n = armas_pivot_get(&p1, k);
            n = n > 0 ? n + ATL.rows : n - ATL.rows;
            armas_pivot_set(&p1, k, n);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &A11, &A22, /**/ A, ARMAS_PBOTTOMRIGHT);
        pivot_cont_3x1to2x1(
            &pT, &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
    }

    if (ABR.cols > 0) {
        armas_x_unblk_bkfactor_lower(&ABR, W, &pB, conf);
        // shift pivots in this block to origin of A
        for (k = 0; k < armas_pivot_size(&pB); k++) {
            n = armas_pivot_get(&pB, k);
            n = n > 0 ? n + ATL.rows : n - ATL.rows;
            armas_pivot_set(&pB, k, n);
        }
    }
    return 0;
}


int armas_x_unblk_bksolve_lower(armas_x_dense_t * B, armas_x_dense_t * A,
                                armas_pivot_t * P, int phase,
                                armas_conf_t * conf)
{
    armas_x_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
    armas_x_dense_t BT, BB, B0, b1, B2, Bx;
    armas_x_dense_t *Aref;
    armas_pivot_t pT, pB, p0, p1, p2;
    int aStart, aDir, bStart, bDir;
    int nc, r, np, pr, k;
    DTYPE b, apb, dpb, scale, s0, s1;

    EMPTY(ATL);
    EMPTY(A00);
    EMPTY(a11);

    np = 0;
    if (phase == 1) {
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

        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10,  &a11, __nil,
            &A20,  &a21,  &A22, /**/ A, np, aDir);
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
                    // swap rows on bottom part of B
                    swap_rows(&BB, 0, pr - BT.rows - 1, conf);
                }
                // B2 = B2 - a21*b1
                armas_x_mvupdate(ONE, &B2, -ONE, &a21, &b1, conf);
                // b1 = b1/d1
                armas_x_scale(&b1, ONE/armas_x_get(&a11, 0, 0), conf);
                nc += 1;
            } else if (np == 2) {
                if (pr != -nc) {
                    // swap rows on bottom part of B
                    swap_rows(&BB, 1, -pr - BT.rows - 1, conf);
                }
                b = armas_x_get(&a11, 1, 0);
                apb = armas_x_get(&a11, 0, 0) / b;
                dpb = armas_x_get(&a11, 1, 1) / b;
                // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
                scale = apb * dpb - 1.0;
                scale *= b;
                // B2 = B2 - a21*b1
                armas_x_mult(ONE, &B2, -ONE, &a21, &b1, ARMAS_NONE, conf);
                // b1 = a11.-1*b1.T
                // (2x2 block, no function for doing this in-place)
                for (k = 0; k < b1.cols; k++) {
                    s0 = armas_x_get(&b1, 0, k);
                    s1 = armas_x_get(&b1, 1, k);
                    armas_x_set(&b1, 0, k, (dpb * s0 - s1) / scale);
                    armas_x_set(&b1, 1, k, (apb * s1 - s0) / scale);
                }
                nc += 2;
            }
            break;

        case 2:
            if (np == 1) {
                armas_x_mvmult(ONE, &b1, -ONE, &B2, &a21, ARMAS_TRANSA,
                               conf);
                if (pr != nc) {
                    // swap rows on bottom part of B
                    mat_merge2x1(&Bx, &b1, &B2);
                    swap_rows(&Bx, 0, pr - BT.rows, conf);
                }
                nc -= 1;
            } else if (np == 2) {
                armas_x_mult(ONE, &b1, -ONE, &a21, &B2, ARMAS_TRANSA, conf);
                if (pr != -nc) {
                    // swap rows on bottom part of B
                    mat_merge2x1(&Bx, &b1, &B2);
                    swap_rows(&Bx, 1, -pr - BT.rows + 1, conf);
                }
                nc -= 2;
            }
            break;

        default:
            break;
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &a11, &A22, /**/ A, aDir);
        mat_continue_3x1to2x1(
            &BT, &BB,   /**/ &B0, &b1, /**/ B, bDir);
        pivot_cont_3x1to2x1(
            &pT, &pB,   /**/ &p0, &p1, /**/ P, bDir);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
