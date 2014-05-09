
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__unblk_bkfactor_upper) && defined(__blk_bkfactor_upper)
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
 * UPPER triangular; moving from bottom-right to top-left
 *
 *    d x D3 x  x  x  S3 x |
 *      d D3 x  x  x  S3 x |
 *        P3 D2 D2 D2 P2 x |  -- dstinx
 *           d  x  x  S2 x |
 *              d  x  S2 x |
 *                 d  S2 x |
 *                    P1 x |  -- srcinx
 *                       d |
 *    ----------------------
 *               (ABR)
 */
static
void __apply_bkpivot_upper(__armas_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
{
  __armas_dense_t s, d;
  DTYPE p1, p3;
  if (srcix == dstix)
    return;

  // S2 -- D2
  __armas_submatrix(&s, AR, dstix+1, srcix,   srcix-dstix-1, 1);
  __armas_submatrix(&d, AR, dstix,   dstix+1, 1, srcix-dstix-1);
  __armas_swap(&s, &d, conf);
  // S3 -- D3
  __armas_submatrix(&s, AR, 0, srcix,  dstix, 1);
  __armas_submatrix(&d, AR, 0, dstix,  dstix, 1);
  __armas_swap(&s, &d, conf);
  // swap P1 and P3
  p1 = __armas_get(AR, srcix, srcix);
  p3 = __armas_get(AR, dstix, dstix);
  __armas_set(AR, srcix, srcix, p3);
  __armas_set(AR, dstix, dstix, p1);
}


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
int __find_bkpivot_upper(__armas_dense_t *A, int *nr, int *np, armas_conf_t *conf)
{
  __armas_dense_t rcol, qrow;
  DTYPE amax, rmax, qmax, qmax2;
  int r, q, lastcol;

  if (A->rows == 1) {
    *nr = -1; *np = 1;
    return 0;
  }
  lastcol = A->rows - 1;
  amax = __ABS(__armas_get(A, lastcol, lastcol));
  // column above diagonal on [lastcol, lastcol]
  __armas_submatrix(&rcol, A, 0, lastcol, lastcol, 1);
  r = __armas_iamax(&rcol, conf);
  // max off-diagonal on first column at index r
  rmax = __ABS(__armas_get(A, r, lastcol));
  if (amax >= bkALPHA*rmax) {
    // no pivoting, 1x1 diagonal
    *nr = -1; *np = 1;
    return 0;
  }
  // max off-diagonal on r'th row at index q
  qmax = 0.0;
  if (r > 0) {
    __armas_submatrix(&qrow, A, 0, r, r, 1);
    q = __armas_iamax(&qrow, conf);
    qmax = __ABS(__armas_get(A, q, r));
  }
  // elements right of diagonal
  __armas_submatrix(&qrow, A, r, r+1, 1, lastcol-r);
  q = __armas_iamax(&qrow, conf);
  qmax2 = __ABS(__armas_get(&qrow, 0, q));
  if (qmax2 > qmax)
    qmax = qmax2;

  
  if (amax >= bkALPHA*rmax*(rmax/qmax)) {
    // no pivoting, 1x1 diagonal
    *nr = -1; *np = 1;
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
int __unblk_bkfactor_upper(__armas_dense_t *A, __armas_dense_t *W,
                           armas_pivot_t *P, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABR, A00, a01, A02, a11, a12, A22;
  __armas_dense_t a11inv, cwrk;
  armas_pivot_t pT, pB, p0, p1, p2;
  DTYPE t, a11val, a, b, d, scale;
  DTYPE abuf[4];
  int nc, r, np, nr, err, pi;

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR, /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PBOTTOM);

  // permanent working space for symmetric inverse of 2x2 a11
  //__armas_submatrix(&a11inv, W, 0, W->cols-2, 2, 2);
  __armas_make(&a11inv, 2, 2, 2, abuf);
  __armas_set(&a11inv, 0, 1, -1.0);
  __armas_set(&a11inv, 1, 0, -1.0);
  
  nc = 0;
  while (ATL.cols > 0) {

    nr = ATL.rows - 1;
    __find_bkpivot_upper(&ATL, &r, &np, conf);
    if (r != -1) {
      // pivoting needed, do swaping here
      __apply_bkpivot_upper(&ATL, ATL.rows-np, r, conf);
      if (np == 2) {
        /*          [r, r] | [r ,nr]
         * a11 ==   ----------------  2-by-2 pivot
         *          [nr,r] | [nr,nr]
         */
        t = __armas_get(&ATL, nr-1, nr);
        __armas_set(&ATL, nr-1, nr, __armas_get(&ATL, r, nr));
        __armas_set(&ATL, r, nr, t);
      }
    }
    // ---------------------------------------------------------------------------
    // repartition according the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           __nil, &a11,  __nil,
                           __nil, __nil, &A22,  /**/  A, np, ARMAS_PTOPLEFT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, np, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    if (np == 1) {
      // A00 = A00 - a01*a01.T/a11
      a11val = __armas_get(&a11, 0, 0);
      __armas_mvupdate_trm(&A00, &a01, &a01, -1.0/a11val, ARMAS_UPPER, conf);
      // a01 = a01/a11
      __armas_invscale(&a01, a11val, conf);
      // store pivot point relative to original matrix
      pi = r == -1 ? ATL.rows : r + 1;
      armas_pivot_set(&p1, 0, pi);
    }
    else if (np == 2) {
      /* see comments in __unblk_bkfactor_lower() */
      a = __armas_get(&a11, 0, 0);
      b = __armas_get(&a11, 0, 1);
      d = __armas_get(&a11, 1, 1);
      __armas_set(&a11inv, 0, 0, d/b);
      __armas_set(&a11inv, 1, 1, a/b);
      // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
      scale = 1.0 / ((a/b)*(d/b) - 1.0);
      scale /= b;
      // cwrk = a01
      __armas_submatrix(&cwrk, W, 2, 0, a01.rows, np);
      __armas_mcopy(&cwrk, &a01);
      // a01 := a01*a11.-1
      __armas_mult(&a01, &cwrk, &a11inv, scale, 0.0, ARMAS_NONE, conf);
      // A00 := A00 - a01*a11.-1*a01.T = A00 - a01*cwrk.T
      __armas_update_trm(&A00, &a01, &cwrk, -1.0, 1.0, ARMAS_UPPER|ARMAS_TRANSB, conf);
      // store pivot points
      pi = r + 1;
      armas_pivot_set(&p1, 0, -pi);
      armas_pivot_set(&p1, 1, -pi);
    }
    // ---------------------------------------------------------------------------
    nc += np;
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR,  /**/  &A00, &a11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
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
 *   --------------------------   -------------
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
int __build_bkpivot_upper(__armas_dense_t *AL, __armas_dense_t *AR,
                          __armas_dense_t *WL, __armas_dense_t *WR,
                          int k, int *nr, int *np, armas_conf_t *conf)
{
  __armas_dense_t rcol, qrow, src, wk, wkp1, wkr, wrow;
  int r, q, lc, wc, lr;
  DTYPE amax, rmax, qmax, qmax2, p1;
  
  lc = AL->cols - 1;
  wc = WL->cols - 1;
  lr = AL->rows - 1;

  // Copy AR column 0 to WR column 0 and update with WL[0:,]
  __armas_submatrix(&src, AL, 0, lc, AL->rows, 1);
  __armas_submatrix(&wk,  WL, 0, wc, AL->rows, 1);
  __armas_copy(&wk, &src, conf);
  if (k > 0) {
    __armas_submatrix(&wrow, WR, lr, 0, 1, WR->cols);
    __armas_mvmult(&wk, AR, &wrow, -1.0, 1.0, ARMAS_NONE, conf);
  }
  if (AL->rows == 1) {
    *nr = -1; *np = 1;
    return 0;
  }

  // amax is on-diagonal element of current column
  amax = __ABS(__armas_get(WL, lr, wc));
  // find max off-diagonal on last column
  __armas_submatrix(&rcol, WL, 0, wc, lr, 1);
  // r is row index on WR and rmax is it abs value
  r = __armas_iamax(&rcol, conf);
  rmax = __ABS(__armas_get(&rcol, r, 0));
  if (amax >= bkALPHA*rmax) {
    // no pivoting, 1x1 diagonal
    *nr = -1; *np = 1;
    return 0;
  }

  // now we need to copy row r to WL[:,wc-1] (= wkp1) and update it
  __armas_submatrix(&wkp1, WL, 0, wc-1, AL->rows, 1);
  if (r > 0) {
    // above diagonal part of AL
  __armas_submatrix(&qrow, AL, 0, r, r, 1);
  __armas_submatrix(&wkr,  &wkp1, 0, 0, r, 1);
  __armas_copy(&wkr, &qrow, conf);
  }
  __armas_submatrix(&qrow, AL,    r, r, 1, AL->rows-r);
  __armas_submatrix(&wkr,  &wkp1, r, 0, AL->rows-r, 1);
  __armas_copy(&wkr, &qrow, conf);

  if (k > 0) {
    // update wkp1 
    __armas_submatrix(&wrow, WR, r, 0, 1, WR->cols);
    __armas_mvmult(&wkp1, AR, &wrow, -1.0, 1.0, ARMAS_NONE, conf);
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
    *nr = -1; *np = 1;
    return 0;
  }
  rmax = __ABS(__armas_get(WL, r, wc-1));
  if (rmax >= bkALPHA*qmax) {
    // 1x1 pivoting, interchange with k, r
    // move pivot row in column WR[:,1] to WR[:,0]
    __armas_submatrix(&src,  WL, 0, wc-1, AL->rows, 1);
    __armas_submatrix(&wkp1, WL, 0, wc,   AL->rows, 1);
    __armas_copy(&wkp1, &src, conf);
    __armas_set(&wkp1, -1, 0, __armas_get(&src,  r, 0));
    __armas_set(&wkp1,  r, 0, __armas_get(&src, -1, 0));
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
int __unblk_bkbounded_upper(__armas_dense_t *A, __armas_dense_t *W,
                            armas_pivot_t *P, int ncol, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a01, A02, a11, a12, A22;
  __armas_dense_t a11inv, cwrk, w00, w01, w11;
  armas_pivot_t pT, pB, p0, p1, p2;
  DTYPE t1, tr, a11val, a, b, d, scale;
  int nc, r, np, err, pi;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PBOTTOM);

  // permanent working space for symmetric inverse of 2x2 a11
  __armas_submatrix(&a11inv, W, W->rows-2, 0, 2, 2);
  __armas_set(&a11inv, 1, 0, -1.0);
  __armas_set(&a11inv, 0, 1, -1.0);
  
  nc = 0;
  if (ncol > A->cols)
    ncol = A->cols;
  
  np = 0;
  for (nc = 0; ATL.cols > 0 && nc < ncol; nc += np) {

    __partition_2x2(&w00,  &w01,
                    __nil, &w11, /**/  W, nc, nc, ARMAS_PBOTTOMRIGHT);

    __build_bkpivot_upper(&ATL, &ATR, &w00, &w01, ncol, &r, &np, conf);
    if (np > ncol - nc) {
      // next pivot does not fit into ncol columns, restore last column
      // and return with number of factorized columns
      return nc;
    }

    if (r != -1) {
      // pivoting needed, do swaping here
      int k = ATL.rows - np;
      __apply_bkpivot_upper(&ATL, k, r, conf);
      // swap right hand rows to get correct updates (undone in caller!)
      __swap_rows(&ATR, k, r, conf);
      __swap_rows(&w01, k, r, conf);
      if (np == 2 && r != k) {
        /*               [r,r ] | [r,-1]
         * w00 =a11 ==   --------------- 
         *               [-1,r] | [-1,-1]
         */
        t1 = __armas_get(&w00, k, -1);
        tr = __armas_get(&w00, r, -1);
        __armas_set(&w00, k, -1, tr);
        __armas_set(&w00, r, -1, t1);
        // interchange diagonal entries on w00[:,-2] 
        t1 = __armas_get(&w00, k, -2);
        tr = __armas_get(&w00, r, -2);
        __armas_set(&w00, k, -2, tr);
        __armas_set(&w00, r, -2, t1);
      }
    }
    // ---------------------------------------------------------------------------
    // repartition accoring the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  &A02,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, np, ARMAS_PTOPLEFT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, np, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __armas_submatrix(&cwrk, &w00, 0, w00.cols-np, a01.rows, np);
    if (np == 1) {
      // 
      a11val = __armas_get(&w00, a01.rows, w00.cols-np);
      __armas_set(&a11, 0, 0, a11val);
      // a01 = a01/a11
      __armas_copy(&a01, &cwrk, conf);
      __armas_invscale(&a01, a11val, conf);
      // store pivot point relative to original matrix
      pi = r == -1 ? ATL.rows : r + 1;
      armas_pivot_set(&p1, 0, pi);
    } else if (np == 2) {
      /*          a | b                       d/b | -1
       *  w00 == ------  == a11 --> a11.-1 == -------- * scale
       *          . | d                        -1 | a/b
       */
      a = __armas_get(&w00, ATL.rows-2, -2);
      b = __armas_get(&w00, ATL.rows-2, -1);
      d = __armas_get(&w00, ATL.rows-1, -1);
      __armas_set(&a11inv, 0, 0, d/b);
      __armas_set(&a11inv, 1, 1, a/b);
      // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
      scale = 1.0 / ((a/b)*(d/b) - 1.0);
      scale /= b;
      // cwrk = a01
      // a01 := a01*a11.-1
      __armas_mult(&a01, &cwrk, &a11inv, scale, 0.0, ARMAS_NONE, conf);
      __armas_set(&a11, 0, 0, a);
      __armas_set(&a11, 0, 1, b);
      __armas_set(&a11, 1, 1, d);
      // store pivot points
      pi = r + 1;
      armas_pivot_set(&p1, 0, -pi);
      armas_pivot_set(&p1, 1, -pi);
    }

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &a11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
  }
  return nc;
}


int __blk_bkfactor_upper(__armas_dense_t *A, __armas_dense_t *W,
                         armas_pivot_t *P, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, A01, A02, A11, A12, A22;
  __armas_dense_t cwrk, s, d;
  armas_pivot_t pT, pB, p0, p1, p2;
  int nblk, k, r, r1, rlen, np, colno, n;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR, /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PBOTTOM);

  while (ATL.cols - lb > 0) {
    nblk = __unblk_bkbounded_upper(&ATL, W, &pT, lb, conf);
    // ---------------------------------------------------------------------------
    // repartition accoring the pivot size
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  &A02,
                           __nil, &A11,  &A12,
                           __nil, __nil, &A22,  /**/  A, nblk, ARMAS_PTOPLEFT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, nblk, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    // here [A11 A21] as been factorized, now update A22

    // nblk last columns in W is original A01
    __armas_submatrix(&cwrk, W, 0, W->cols-nblk, A01.rows, nblk);
    // A00 := A00 - L01*D1*L01.T == A00 - A01*W.T
    __armas_update_trm(&A00, &A01, &cwrk, -1.0, 1.0, ARMAS_UPPER|ARMAS_TRANSB, conf);

    // undo partial row pivots right of diagonal from lower level
    for (k = 0; k < nblk; k++) {
      r1   = armas_pivot_get(&p1, k);
      r    = r1 < 0 ? -r1   : r1;
      np   = r1 < 0 ? 2 : 1;
      colno = A00.cols + k;
      if (r == colno + 1 && r1 > 0) {
        // no pivot
        continue;  
      }

      rlen = ATL.cols - colno - np;
      __armas_submatrix(&s, &ATL, colno, colno+np, 1, rlen);
      __armas_submatrix(&d, &ATL, r-1,   colno+np, 1, rlen);
      __armas_swap(&d, &s, conf);
      if (r1 < 0) {
        // skip the other entry in 2x2 pivots
        k++;
      }
    }

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &A11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PTOP);
  }

  if (ATL.cols > 0) {
    __unblk_bkfactor_upper(&ATL, W, &pT, conf);
  }
  return 0;
}


int __unblk_bksolve_upper(__armas_dense_t *B, __armas_dense_t *A,
                          armas_pivot_t *P, int phase, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a01, A02, a11, a12, A22;
  __armas_dense_t BT, BB, B0, b1, B2, Bx;
  __armas_dense_t *Aref;
  armas_pivot_t pT, pB, p0, p1, p2;
  int aStart, aDir, bStart, bDir;
  int nc, r, np, err, pi, k, pr;
  DTYPE b, apb, dpb, scale, s0, s1;

  np = 0;
  if (phase == 2) {
    aStart = ARMAS_PTOPLEFT;
    aDir   = ARMAS_PBOTTOMRIGHT;
    bStart = ARMAS_PTOP;
    bDir   = ARMAS_PBOTTOM;
    nc     = 1;
    Aref   = &ABR;
  }
  else {
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
                           &A00,  &a01,  &A02,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, np, aDir);
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
          // swap rows on top part of B
          __swap_rows(&BT, BT.rows-1, pr-1, conf);
        }
        // B0 = B0 - a01*b1
        __armas_mvupdate(&B0, &a01, &b1, -1.0, conf);
        // b1 = b1/d1
        __armas_invscale(&b1, __armas_get(&a11, 0, 0), conf);
        nc -= 1;
      }
      else if (np == 2) {
        if (pr != -nc) {
          // swap rows on top part of B
          __swap_rows(&BT, BT.rows-2, -pr-1, conf);
        }
        b   = __armas_get(&a11, 0, 1);
        apb = __armas_get(&a11, 0, 0) / b;
        dpb = __armas_get(&a11, 1, 1) / b;
        // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
        scale = apb*dpb - 1.0;
        scale *= b;
        // B0 = B0 - a01*b1
        __armas_mult(&B0, &a01, &b1, -1.0, 1.0, ARMAS_NONE, conf);
        // b1 = a11.-1*b1.T
        // (2x2 block, no function for doing this in-place)
        for (k = 0; k < b1.cols; k++) {
          s0 = __armas_get(&b1, 0, k);
          s1 = __armas_get(&b1, 1, k);
          __armas_set(&b1, 0, k, (dpb*s0-s1)/scale);
          __armas_set(&b1, 1, k, (apb*s1-s0)/scale);
        }
        nc -= 2;
      }
      break;

    case 2:
      if (np == 1) {
        __armas_mvmult(&b1, &B0, &a01, -1.0, 1.0, ARMAS_TRANS, conf);
        if (pr != nc) {
          // swap rows on top part of B
          __merge2x1(&Bx, &B0, &b1);
          __swap_rows(&Bx, Bx.rows-1, pr-1, conf);
        }
        nc += 1;
      }
      else if (np == 2) {
        __armas_mult(&b1, &a01, &B0, -1.0, 1.0, ARMAS_TRANSA, conf);
        if (pr != -nc) {
          // swap rows on top part of B
          __merge2x1(&Bx, &B0, &b1);
          __swap_rows(&Bx, Bx.rows-2, -pr-1, conf);
        }
        nc += 2;
      }
      break;

    default:
      break;
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR,  /**/  &A00, &a11, &A22, /**/  A, aDir);
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
