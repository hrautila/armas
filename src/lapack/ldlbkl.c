
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__unblk_bkfactor_lower) && defined(__blk_bkfactor_lower)
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

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * LOWER triangular; moving from top-left to bottom-right
 *
 *    -----------------------
 *    | d 
 *    | x P1 x  x  x  P2     -- current row/col 'srcix'
 *    | x S2 d  x  x  x
 *    | x S2 x  d  x  x
 *    | x S2 x  x  d  x
 *    | x P2 D2 D2 D2 P3     -- swap with row/col 'dstix'
 *    | x S3 x  x  x  D3 d
 *    | x S3 x  x  x  D3 x d
 *         (AR)
 */
static
void __apply_bkpivot_lower(__armas_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
{
  __armas_dense_t s, d;
  DTYPE p1, p3;
  // S2 -- D2
  __armas_submatrix(&s, AR, srcix+1, srcix,   dstix-srcix-1, 1);
  __armas_submatrix(&d, AR, dstix,   srcix+1, 1, dstix-srcix-1);
  __armas_swap(&s, &d, conf);
  // S3 -- D3
  __armas_submatrix(&s, AR, dstix+1, srcix,  AR->rows-dstix-1, 1);
  __armas_submatrix(&d, AR, dstix+1, dstix,  AR->rows-dstix-1, 1);
  __armas_swap(&s, &d, conf);
  // swap P1 and P3
  p1 = __armas_get(AR, srcix, srcix);
  p3 = __armas_get(AR, dstix, dstix);
  __armas_set(AR, srcix, srcix, p3);
  __armas_set(AR, dstix, dstix, p1);
}


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
int __find_bkpivot_lower(__armas_dense_t *A, int *nr, int *np, armas_conf_t *conf)
{
  __armas_dense_t rcol, qrow;
  DTYPE amax, rmax, qmax, qmax2;
  int r, q;

  if (A->rows == 1) {
    *nr = 0; *np = 1;
    return 0;
  }
  amax = __ABS(__armas_get(A, 0, 0));
  // column below diagonal at [0,0]
  __armas_submatrix(&rcol, A, 1, 0, A->rows-1, 1);
  r = __armas_iamax(&rcol, conf) + 1;
  // max off-diagonal on first column at index r
  rmax = __ABS(__armas_get(A, r, 0));
  if (amax >= bkALPHA*rmax) {
    // no pivoting, 1x1 diagonal
    *nr = 0; *np = 1;
    return 0;
  }
  // max off-diagonal on r'th row at index q
  __armas_submatrix(&qrow, A, r, 0, 1, r);
  q = __armas_iamax(&qrow, conf);
  qmax = __ABS(__armas_get(A, r, q));
  if (r < A->rows-1) {
    // rest of the r'th row after diagonal
    __armas_submatrix(&qrow, A, r+1, r, A->rows-r-1, 1);
    q = __armas_iamax(&qrow, conf);
    qmax2 = __ABS(__armas_get(&qrow, q, 0));
    if (qmax2 > qmax)
      qmax = qmax2;
  }
  
  if (amax >= bkALPHA*rmax*(rmax/qmax)) {
    // no pivoting, 1x1 diagonal
    *nr = 0; *np = 1;
    return 0;
  }
  rmax = __ABS(__armas_get(A, r, r));
  if (rmax >= bkALPHA*qmax) {
    // 1x1 pivoting, interchange with k, r
    *nr = r; *np = 1;
    return 1;
  } 
  // 2x2 pivoting, interchange with k+1, r
  *nr = r; *np = 2;
  return 2;
}

/*
 * Unblocked Bunch-Kauffman LDL factorization.
 *
 * Corresponds lapack.DSYTF2
 */
int __unblk_bkfactor_lower(__armas_dense_t *A, __armas_dense_t *W,
                           armas_pivot_t *P, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
  __armas_dense_t a11inv, cwrk;
  armas_pivot_t pT, pB, p0, p1, p2;
  DTYPE t, a11val, a, b, d, scale;
  DTYPE abuf[4];
  int nc, r, np, err, pi;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PTOP);

  // permanent working space for symmetric inverse of 2x2 a11
  //__armas_submatrix(&a11inv, W, 0, W->cols-2, 2, 2);
  __armas_make(&a11inv, 2, 2, 2, abuf);
  __armas_set(&a11inv, 1, 0, -1.0);
  __armas_set(&a11inv, 0, 1, -1.0);
  
  nc = 0;
  while (ABR.cols > 0) {

    __find_bkpivot_lower(&ABR, &r, &np, conf);
    if (r != 0 && r != np-1) {
      // pivoting needed, do swaping here
      __apply_bkpivot_lower(&ABR, np-1, r, conf);
      if (np == 2) {
        /*          [0,0] | [r,0]
         * a11 ==   -------------  2-by-2 pivot, swapping [1,0] and [r,0]
         *          [r,0] | [r,r]
         */
        t = __armas_get(&ABR, 1, 0);
        __armas_set(&ABR, 1, 0, __armas_get(&ABR, r, 0));
        __armas_set(&ABR, r, 0, t);
      }
    }
    // ---------------------------------------------------------------------------
    // repartition accoring the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00, __nil, __nil,
                           &a10, &a11,  __nil,
                           &A20, &a21,  &A22,  /**/  A, np, ARMAS_PBOTTOMRIGHT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, np, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    if (np == 1) {
      // A22 = A22 - a21*a21.T/a11
      a11val = __armas_get(&a11, 0, 0);
      __armas_mvupdate_trm(&A22, &a21, &a21, -1.0/a11val, ARMAS_LOWER, conf);
      // a21 = a21/a11
      __armas_invscale(&a21, a11val, conf);
      // store pivot point relative to original matrix
      armas_pivot_set(&p1, 0, r+ATL.rows+1);
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
      a = __armas_get(&a11, 0, 0);
      b = __armas_get(&a11, 1, 0);
      d = __armas_get(&a11, 1, 1);
      __armas_set(&a11inv, 0, 0, d/b);
      __armas_set(&a11inv, 1, 1, a/b);
      // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
      scale = 1.0 / ((a/b)*(d/b) - 1.0);
      scale /= b;
      // cwrk = a21
      __armas_submatrix(&cwrk, W, 0, 0, a21.rows, a21.cols);
      __armas_mcopy(&cwrk, &a21);
      // a21 := a21*a11.-1
      __armas_mult(&a21, &cwrk, &a11inv, scale, 0.0, ARMAS_NONE, conf);
      // A22 := A22 - a21*a11.-1*a21.T = A22 - a21*cwrk.T
      __armas_update_trm(&A22, &a21, &cwrk, -1.0, 1.0, ARMAS_LOWER|ARMAS_TRANSB, conf);
      // store pivot points
      pi = r + ATL.rows + 1;
      armas_pivot_set(&p1, 0, -pi);
      armas_pivot_set(&p1, 1, -pi);
    }
    // ---------------------------------------------------------------------------
    nc += np;
    __continue_3x3to2x2(&ATL, __nil,
                        &ABL, &ABR,  /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
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
int __build_bkpivot_lower(__armas_dense_t *AL, __armas_dense_t *AR,
                          __armas_dense_t *WL, __armas_dense_t *WR,
                          int k, int *nr, int *np, armas_conf_t *conf)
{
  __armas_dense_t rcol, qrow, src, wk, wkp1, wkr, wrow;
  int r, q;
  DTYPE amax, rmax, qmax, qmax2, p1;
  
  // Copy AR column 0 to WR column 0 and update with WL[0:,]
  __armas_submatrix(&src, AR, 0, 0, AR->rows, 1);
  __armas_submatrix(&wk,  WR, 0, 0, AR->rows, 1);
  __armas_copy(&wk, &src, conf);
  if (k > 0) {
    __armas_submatrix(&wrow, WL, 0, 0, 1, WL->cols);
    __armas_mvmult(&wk, AL, &wrow, -1.0, 1.0, ARMAS_NONE, conf);
  }
  if (AR->rows == 1) {
    *nr = 0; *np = 1;
    return 0;
  }

  amax = __ABS(__armas_get(WR, 0, 0));
  // find max off-diagonal on first column (= WR[:,0])
  __armas_submatrix(&rcol, WR, 1, 0, AR->rows-1, 1);
  // r is row index on WR and rmax is it abs value
  r = __armas_iamax(&rcol, conf) + 1;
  rmax = __ABS(__armas_get(&rcol, r-1, 0));
  if (amax >= bkALPHA*rmax) {
    // no pivoting, 1x1 diagonal
    *nr = 0; *np = 1;
    return 0;
  }

  // now we need to copy row r to WR[:,1] (= wkp1) and update it
  __armas_submatrix(&wkp1, WR, 0, 1, AR->rows, 1);

  __armas_submatrix(&qrow, AR, r, 0, 1, r+1);
  __armas_submatrix(&wkr,  &wkp1, 0, 0, r+1, 1);
  __armas_axpby(&wkr, &qrow, 1.0, 0.0, conf);
  if ( r < AR->rows - 1) {
    __armas_submatrix(&qrow, AR,    r, r, AR->rows-r, 1);
    __armas_submatrix(&wkr,  &wkp1, r, 0, AR->rows-r, 1);
    __armas_copy(&wkr, &qrow, conf);
  }
  if (k > 0) {
    // update wkp1 
    __armas_submatrix(&wrow, WL, r, 0, 1, WL->cols);
    __armas_mvmult(&wkp1, AL, &wrow, -1.0, 1.0, ARMAS_NONE, conf);
  }
  // set on-diagonal entry to zero to avoid finding it
  p1 = __armas_get(&wkp1, r, 0);
  __armas_set(&wkp1, r, 0, 0.0);
  // max off-diagonal on r'th column/row on at index q
  q = __armas_iamax(&wkp1, conf);
  qmax = __ABS(__armas_get(&wkp1, q, 0));
  // restore on-diagonal entry
  __armas_set(&wkp1, r, 0, p1);

  
  if (amax >= bkALPHA*rmax*(rmax/qmax)) {
    // no pivoting, 1x1 diagonal
    *nr = 0; *np = 1;
    return 0;
  }
  rmax = __ABS(__armas_get(WR, r, 1));
  if (rmax >= bkALPHA*qmax) {
    // 1x1 pivoting, interchange with k, r
    // move pivot row in column WR[:,1] to WR[:,0]
    __armas_submatrix(&src,  WR, 0, 1, AR->rows, 1);
    __armas_submatrix(&wkp1, WR, 0, 0, AR->rows, 1);
    __armas_copy(&wkp1, &src, conf);
    __armas_set(&wkp1, 0, 0, __armas_get(&src, r, 0));
    __armas_set(&wkp1, r, 0, __armas_get(&src, 0, 0));
    *nr = r; *np = 1;
    return 1;
  } 
  // 2x2 pivoting, interchange with k+1, r
  *nr = r; *np = 2;
  return 2;
}

/*
 * Unblocked, bounded Bunch-Kauffman LDL factorization for at most ncol columns.
 * At most ncol columns are factorized and trailing matrix updates are restricted
 * to ncol columns. Also original columns are accumulated to working matrix, which
 * is used by calling blocked algorithm to update the trailing matrix with BLAS3
 * update.
 *
 * Corresponds lapack.DLASYF
 */
static
int __unblk_bkbounded_lower(__armas_dense_t *A, __armas_dense_t *W,
                            armas_pivot_t *P, int ncol, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
  __armas_dense_t a11inv, cwrk, w00, w10, w11;
  armas_pivot_t pT, pB, p0, p1, p2;
  DTYPE t1, tr, a11val, a, b, d, scale;
  DTYPE abuf[4];
  int nc, r, np, err, pi;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PTOP);

  // permanent working space for symmetric inverse of 2x2 a11
  __armas_submatrix(&a11inv, W, 0, W->cols-2, 2, 2);
  //__armas_make(&a11inv, 2, 2, 2, abuf);
  __armas_set(&a11inv, 1, 0, -1.0);
  __armas_set(&a11inv, 0, 1, -1.0);
  
  nc = 0;
  if (ncol > A->cols)
    ncol = A->cols;
  
  while (ABR.cols > 0 && nc < ncol) {

    __partition_2x2(&w00, __nil,
                    &w10, &w11, /**/  W, nc, nc, ARMAS_PTOPLEFT);

    __build_bkpivot_lower(&ABL, &ABR, &w10, &w11, ncol, &r, &np, conf);
    if (np > ncol - nc) {
      // next pivot does not fit into ncol columns, restore last column
      // and return with number of factorized columns
      return nc;
    }
    if (r != 0 && r != np-1) {
      // pivoting needed, do swaping here
      __apply_bkpivot_lower(&ABR, np-1, r, conf);
      // swap left hand rows to get correct updates (this will be undone in caller!)
      __swap_rows(&ABL, np-1, r, conf);
      __swap_rows(&w10, np-1, r, conf);
      if (np == 2) {
        /*          [0,0] | [r,0]
         * a11 ==   -------------  2-by-2 pivot, swapping [1,0] and [r,0]
         *          [r,0] | [r,r]
         */
        t1 = __armas_get(&w11, 1, 0);
        tr = __armas_get(&w11, r, 0);
        __armas_set(&w11, 1, 0, tr);
        __armas_set(&w11, r, 0, t1);
        // interchange diagonal entries on w11[:,1] 
        t1 = __armas_get(&w11, 1, 1);
        tr = __armas_get(&w11, r, 1);
        __armas_set(&w11, 1, 1, tr);
        __armas_set(&w11, r, 1, t1);
      }
    }
    // ---------------------------------------------------------------------------
    // repartition accoring the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00, __nil, __nil,
                           &a10, &a11,  __nil,
                           &A20, &a21,  &A22,  /**/  A, np, ARMAS_PBOTTOMRIGHT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, np, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    if (np == 1) {
      __armas_submatrix(&cwrk, &w11, np, 0, a21.rows, np);
      // new a11 from w11[0,0]
      a11val = __armas_get(&w11, 0, 0);
      __armas_set(&a11, 0, 0, a11val);
      // a21 = a21/a11
      __armas_axpby(&a21, &cwrk, 1.0, 0.0, conf);
      __armas_invscale(&a21, a11val, conf);
      // store pivot point relative to original matrix
      armas_pivot_set(&p1, 0, r+ATL.rows+1);
    } else if (np == 2) {
      /* 
       * See corresponding comment block in __unblk_bkfactor_lower().
       */
      a = __armas_get(&w11, 0, 0);
      b = __armas_get(&w11, 1, 0);
      d = __armas_get(&w11, 1, 1);
      __armas_set(&a11inv, 0, 0, d/b);
      __armas_set(&a11inv, 1, 1, a/b);
      // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
      scale = 1.0 / ((a/b)*(d/b) - 1.0);
      scale /= b;
      // cwrk = a21
      __armas_submatrix(&cwrk, &w11, np, 0, a21.rows, np);
      // a21 := a21*a11.-1
      __armas_mult(&a21, &cwrk, &a11inv, scale, 0.0, ARMAS_NONE, conf);
      __armas_set(&a11, 0, 0, a);
      __armas_set(&a11, 1, 0, b);
      __armas_set(&a11, 1, 1, d);
      // store pivot points
      pi = r + ATL.rows + 1;
      armas_pivot_set(&p1, 0, -pi);
      armas_pivot_set(&p1, 1, -pi);
    }
    // ---------------------------------------------------------------------------
    nc += np;
    __continue_3x3to2x2(&ATL, __nil,
                        &ABL, &ABR,  /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
  }
  return nc;
}


int __blk_bkfactor_lower(__armas_dense_t *A, __armas_dense_t *W,
                         armas_pivot_t *P, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, A10, A11, A20, A21, A22;
  __armas_dense_t cwrk, s, d;
  armas_pivot_t pT, pB, p0, p1, p2;
  int nblk, k, r, r1, rlen, n;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PTOP);

  while (ABR.cols - lb > 0) {

    nblk = __unblk_bkbounded_lower(&ABR, W, &pB, lb, conf);
    // ---------------------------------------------------------------------------
    // repartition accoring the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00, __nil, __nil,
                           &A10, &A11,  __nil,
                           &A20, &A21,  &A22,  /**/  A, nblk, ARMAS_PBOTTOMRIGHT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, nblk, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    // here [A11 A21] as been factorized, now update A22

    // nblk first columns in W is original A21; A21 is A21*D1
    __armas_submatrix(&cwrk, W, nblk, 0, A21.rows, nblk);
    // A22 := A22 - L21*D1*L21.T == A22 - A21*W.T
    __armas_update_trm(&A22, &A21, &cwrk, -1.0, 1.0, ARMAS_LOWER|ARMAS_TRANSB, conf);

    // undo partial row pivots left of diagonal from lower level
    for (k = nblk; k > 0; k--) {
      r1   = armas_pivot_get(&p1, k-1);
      r    = r1 < 0 ? -r1   : r1;
      rlen = r1 < 0 ? k - 2 : k - 1;
      if (r == k && r1 > 0) {
        // no pivot
        continue;  
      }

      __armas_submatrix(&s, &ABR, k-1, 0, 1, rlen);
      __armas_submatrix(&d, &ABR, r-1, 0, 1, rlen);
      __armas_swap(&d, &s, conf);
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
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &A11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
  }

  if (ABR.cols > 0) {
    __unblk_bkfactor_lower(&ABR, W, &pB, conf);
    // shift pivots in this block to origin of A
    for (k = 0; k < armas_pivot_size(&pB); k++) {
      n = armas_pivot_get(&pB, k);
      n = n > 0 ? n + ATL.rows : n - ATL.rows;
      armas_pivot_set(&pB, k, n);
    }
  }
  return 0;
}


int __unblk_bksolve_lower(__armas_dense_t *B, __armas_dense_t *A,
                          armas_pivot_t *P, int phase, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a10, a11, A20, a21, A22;
  __armas_dense_t BT, BB, B0, b1, B2, Bx;
  __armas_dense_t *Aref;
  armas_pivot_t pT, pB, p0, p1, p2;
  int aStart, aDir, bStart, bDir;
  int nc, r, np, err, pi, pr, k;
  DTYPE b, apb, dpb, scale, s0, s1;

  np = 0;
  if (phase == 1) {
    aStart = ARMAS_PTOPLEFT;
    aDir   = ARMAS_PBOTTOMRIGHT;
    bStart = ARMAS_PTOP;
    bDir   = ARMAS_PBOTTOM;
    nc     = 1;
    Aref   = &ABR;
  } else {
    aStart = ARMAS_PBOTTOMRIGHT;
    aDir   = ARMAS_PTOPLEFT;
    bStart = ARMAS_PBOTTOM;
    bDir   = ARMAS_PTOP;
    nc     = A->rows;
    Aref   = &ATL;
  }
  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, aStart);
  __partition_2x1(&BT,
                  &BB,   /**/  B, 0, bStart);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, bStart);

  while (Aref->cols > 0) {
    r  = armas_pivot_get(P, nc-1);
    np = r < 0 ? 2 : 1;

    __repartition_2x2to3x3(&ATL,
                           &A00, __nil, __nil,
                           &a10, &a11,  __nil,
                           &A20, &a21,  &A22,  /**/  A, np, aDir);
    __repartition_2x1to3x1(&BT,
                           &B0, 
                           &b1, 
                           &B2,    /**/ B, np, bDir);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,  /**/  P, np, bDir);
    // ---------------------------------------------------------------------------
    pr = armas_pivot_get(&p1, 0);
    switch (phase) {
    case 1:
      if (np == 1) {
        if (pr != nc) {
          // swap rows on bottom part of B
          __swap_rows(&BB, 0, pr-BT.rows-1, conf);
        }
        // B2 = B2 - a21*b1
        __armas_mvupdate(&B2, &a21, &b1, -1.0, conf);
        // b1 = b1/d1
        __armas_invscale(&b1, __armas_get(&a11, 0, 0), conf);
        nc += 1;
      } else if (np == 2) {
        if (pr != -nc) {
          // swap rows on bottom part of B
          __swap_rows(&BB, 1, -pr-BT.rows-1, conf);
        }
        b   = __armas_get(&a11, 1, 0);
        apb = __armas_get(&a11, 0, 0) / b;
        dpb = __armas_get(&a11, 1, 1) / b;
        // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
        scale = apb*dpb - 1.0;
        scale *= b;
        // B2 = B2 - a21*b1
        __armas_mult(&B2, &a21, &b1, -1.0, 1.0, ARMAS_NONE, conf);
        // b1 = a11.-1*b1.T
        // (2x2 block, no function for doing this in-place)
        for (k = 0; k < b1.cols; k++) {
          s0 = __armas_get(&b1, 0, k);
          s1 = __armas_get(&b1, 1, k);
          __armas_set(&b1, 0, k, (dpb*s0-s1)/scale);
          __armas_set(&b1, 1, k, (apb*s1-s0)/scale);
        }
        nc += 2;
      }
      break;

    case 2:
      if (np == 1) {
        __armas_mvmult(&b1, &B2, &a21, -1.0, 1.0, ARMAS_TRANSA, conf);
        if (pr != nc) {
          // swap rows on bottom part of B
          __merge2x1(&Bx, &b1, &B2);
          __swap_rows(&Bx, 0, pr - BT.rows, conf);
        }
        nc -= 1;
      } else if (np == 2) {
        __armas_mult(&b1, &a21, &B2, -1.0, 1.0, ARMAS_TRANSA, conf);
        if (pr != -nc) {
          // swap rows on bottom part of B
          __merge2x1(&Bx, &b1, &B2);
          __swap_rows(&Bx, 1, -pr-BT.rows+1, conf);
        }
        nc -= 2;
      }
      break;

    default:
      break;
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &a11, &A22, /**/  A, aDir);
    __continue_3x1to2x1(&BT,
                        &BB,   /**/  &B0, &b1, /**/ B, bDir);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/  &p0, &p1, /**/ P, bDir);
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
