
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdreduce) && defined(armas_x_trdmult)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) && defined(armas_x_blas)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "partition.h"
//! \endcond
static inline
int __ws_trdreduce(int M, int N, int lb)
{
  return lb == 0 ? M : lb*M;
}

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
int __unblk_trdreduce_lower(armas_x_dense_t *A, armas_x_dense_t *tauq,
                            armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a21, A22;
  armas_x_dense_t tT, tB, tq0, tq1, tq2, y21;
  DTYPE v0, beta, tauval;

  EMPTY(A00); EMPTY(a11);

  if (armas_x_size(tauq) == 0)
    return 0;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, 0, ARMAS_PTOP);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  __nil,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    // temp vector for this round
    armas_x_make(&y21, A22.rows, 1, A22.rows, armas_x_data(W));

    // compute householder to zero subdiagonal entries
    __compute_householder_vec(&a21, &tq1, conf);
    tauval = armas_x_get(&tq1, 0, 0);
    
    // set subdiagonal to unit
    v0 = armas_x_get(&a21, 0, 0);
    armas_x_set(&a21, 0, 0, 1.0);

    // y21 := tauq*A22*a21
    armas_x_mvmult_sym(__ZERO, &y21, tauval, &A22, &a21, ARMAS_LOWER, conf);
    // beta := tauq*a21.T*y21
    beta = tauval * armas_x_dot(&a21, &y21, conf);
    // y21 := y21 - 0.5*beta*a21
    armas_x_axpy(&y21, -__HALF*beta, &a21, conf);
    // A22 := A22 - a21*y21.T - y21*a21.T
    armas_x_mvupdate2_sym(&A22, -__ONE, &a21, &y21, ARMAS_LOWER, conf);
    // restore subdiagonal
    armas_x_set(&a21, 0, 0, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
  }
  return 0;
}

/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
static
int __unblk_trdbuild_lower(armas_x_dense_t *A, armas_x_dense_t *tauq,
                           armas_x_dense_t *Y, armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
  armas_x_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
  armas_x_dense_t tT, tB, tq0, tq1, tq2, w12;
  DTYPE beta, tauval, aa, v0 = __ZERO;
  int k, err = 0;

  EMPTY(A00); EMPTY(a11);
  EMPTY(Y00); EMPTY(y11);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&YTL,  __nil,
                  __nil, &YBR,   /**/  Y, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, 0, ARMAS_PTOP);

  for (k = 0; k < Y->cols; k++) {
    __repartition_2x2to3x3(&ATL,
                           &A00, __nil, __nil,
                           &a10, &a11,  __nil,
                           &A20, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&YTL,
                           &Y00, __nil, __nil,
                           &y10, &y11,  __nil,
                           &Y20, &y21,  &Y22,  /**/  Y, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    armas_x_submatrix(&w12, Y, 0, 0, 1, Y00.cols);
    
    if (Y00.cols > 0) {
      // a11 := a11 - a10*y10 - y10*a10
      aa  = armas_x_dot(&a10, &y10, conf);
      aa += armas_x_dot(&y10, &a10, conf);
      armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
      // a21 := a21 - A20*y10
      armas_x_mvmult(__ONE, &a21, -__ONE, &A20, &y10, ARMAS_NONE, conf);
      // a21 := a21 - Y20*a10
      armas_x_mvmult(__ONE, &a21, -__ONE, &Y20, &a10, ARMAS_NONE, conf);

      // restore subdiagonal value
      armas_x_set(&a10, 0, -1, v0);
    }

    // compute householder to zero subdiagonal entries
    __compute_householder_vec(&a21, &tq1, conf);
    tauval = armas_x_get(&tq1, 0, 0);
    
    // set subdiagonal to unit
    v0 = armas_x_get(&a21, 0, 0);
    armas_x_set(&a21, 0, 0, 1.0);

    // y21 := tauq*A22*a21
    armas_x_mvmult_sym(__ZERO, &y21, tauval, &A22, &a21, ARMAS_LOWER, conf);
    // w12 := A20.T*a21
    armas_x_mvmult(__ZERO, &w12, __ONE, &A20, &a21, ARMAS_TRANS, conf);
    // y21 := y21 - tauq*Y20*(A20.T*a21)
    armas_x_mvmult(__ONE, &y21, -tauval, &Y20, &w12, ARMAS_NONE, conf);
    // w12 := Y20.T*a21
    armas_x_mvmult(__ZERO, &w12, __ONE, &Y20, &a21, ARMAS_TRANS, conf);
    // y21 := y21 - tauq*A20*(Y20.T*a21)
    armas_x_mvmult(__ONE, &y21, -tauval, &A20, &w12, ARMAS_NONE, conf);
    
    // beta := tauq*a21.T*y21
    beta = tauval * armas_x_dot(&a21, &y21, conf);
    // y21 := y21 - 0.5*beta*a21
    armas_x_axpy(&y21, -__HALF*beta, &a21, conf);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&YTL,  __nil,
                        __nil, &YBR, /**/  &Y00, &y11, &Y22, /**/  Y, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
  }

  // restore subdiagonal value
  armas_x_set(A, ATL.rows, ATL.cols-1, v0);
  return err;
}


static
int __blk_trdreduce_lower(armas_x_dense_t *A, armas_x_dense_t *tauq,
                          armas_x_dense_t *Y, armas_x_dense_t *W,
                          int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A11, A21, A22;
  armas_x_dense_t YT, YB, Y0, Y1, Y2;
  armas_x_dense_t tT, tB, tq0, tq1, tq2;
  DTYPE v0 = __ZERO;
  int err = 0;

  EMPTY(A00); EMPTY(A11);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&YT,
                  &YB,   /**/  Y, 0, ARMAS_PTOP);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, 0, ARMAS_PTOP);

  while (ABR.rows - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  __nil,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&YT,
                           &Y0,
                           &Y1,
                           &Y2,  /**/  Y, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, lb, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------

    if (__unblk_trdbuild_lower(&ABR, &tq1, &YB, W, conf)) {
      err = err == 0 ? -1 : err;
    }
    
    // set subdiagonal to unit
    v0 = armas_x_get(&A21, 0, -1);
    armas_x_set(&A21, 0, -1, 1.0);

    // A22 := A22 - A21*Y2.T - Y2*A21.T
    armas_x_update2_sym(__ONE, &A22, -__ONE, &A21, &Y2, ARMAS_LOWER, conf);

    // restore subdiagonal entry
    armas_x_set(&A21, 0, -1, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&YT,
                        &YB, /**/  &Y0, &Y1,  /**/ Y, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
  }

  if (ABR.rows > 0) {
    __unblk_trdreduce_lower(&ABR, &tB, W, conf);
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
int __unblk_trdreduce_upper(armas_x_dense_t *A, armas_x_dense_t *tauq,
                            armas_x_dense_t *W, int tail, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a01, a11, A22;
  armas_x_dense_t tT, tB, tq0, tq1, tq2, y21;
  int toff;
  DTYPE v0, beta, tauval;

  EMPTY(ATL);
  v0 = __ZERO;

  if (armas_x_size(tauq) == 0)
    return 0;

  toff = tail ? 0 : 1;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, toff, ARMAS_PBOTTOM);

  while (ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           __nil, &a11,  __nil,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PTOP);
    // ------------------------------------------------------------------------
    // temp vector for this round
    armas_x_make(&y21, A00.cols, 1, A00.cols, armas_x_data(W));

    // compute householder to zero subdiagonal entries
    __compute_householder_rev(&a01, &tq1, conf);
    tauval = armas_x_get(&tq1, 0, 0);
    
    // set subdiagonal to unit
    v0 = armas_x_get(&a01, -1, 0);
    armas_x_set(&a01, -1, 0, 1.0);

    // y21 := tauq*A00*a01
    armas_x_mvmult_sym(__ZERO, &y21, tauval, &A00, &a01, ARMAS_UPPER, conf);
    // beta := tauq*a01.T*y21
    beta = tauval * armas_x_dot(&a01, &y21, conf);
    // y21 := y21 - 0.5*beta*a01
    armas_x_axpy(&y21, -__HALF*beta, &a01, conf);
    // A00 := A00 - a01*y21.T - y21*a01.T
    armas_x_mvupdate2_sym(&A00, -__ONE, &a01, &y21, ARMAS_UPPER, conf);
    // restore subdiagonal
    armas_x_set(&a01, -1, 0, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PTOP);
  }
  return 0;
}


/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
static
int __unblk_trdbuild_upper(armas_x_dense_t *A, armas_x_dense_t *tauq,
                           armas_x_dense_t *Y, armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
  armas_x_dense_t YTL, YBR, Y00, y01, Y02, y11, y12, Y22;
  armas_x_dense_t tT, tB, tq0, tq1, tq2, w12;
  DTYPE v0, beta, tauval, aa;
  int k, err = 0;

  v0 = __ZERO;
  EMPTY(ATL); EMPTY(a11);
  EMPTY(YTL); EMPTY(Y00);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x2(&YTL,  __nil,
                  __nil, &YBR,   /**/  Y, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, 0, ARMAS_PBOTTOM);

  for (k = 0; k < Y->cols; k++) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  &A02,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x2to3x3(&YTL,
                           &Y00,  &y01,  &Y02,
                           __nil, &y11,  &y12,
                           __nil, __nil, &Y22,  /**/  Y, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PTOP);
    // ------------------------------------------------------------------------
    armas_x_submatrix(&w12, Y, -1, 0, 1, Y02.cols);
    
    if (Y02.cols > 0) {
      // a11 := a11 - a12*y12 - y12*a12
      aa  = armas_x_dot(&a12, &y12, conf);
      aa += armas_x_dot(&y12, &a12, conf);
      armas_x_set(&a11, 0, 0, armas_x_get(&a11, 0, 0) - aa);
      // a01 := a01 - A02*y12
      armas_x_mvmult(__ONE, &a01, -__ONE, &A02, &y12, ARMAS_NONE, conf);
      // a01 := a01 - Y02*a12
      armas_x_mvmult(__ONE, &a01, -__ONE, &Y02, &a12, ARMAS_NONE, conf);

      // restore subdiagonal value
      armas_x_set(&a12, 0, 0, v0);
    }

    // compute householder to zero superdiagonal entries
    __compute_householder_rev(&a01, &tq1, conf);
    tauval = armas_x_get(&tq1, 0, 0);
    
    // set superdiagonal to unit
    v0 = armas_x_get(&a01, -1, 0);
    armas_x_set(&a01, -1, 0, 1.0);

    // y01 := tauq*A00*a01
    armas_x_mvmult_sym(__ZERO, &y01, tauval, &A00, &a01, ARMAS_UPPER, conf);
    // w12 := A02.T*a01
    armas_x_mvmult(__ZERO, &w12, __ONE, &A02, &a01, ARMAS_TRANS, conf);
    // y01 := y01 - tauq*Y02*(A02.T*a01)
    armas_x_mvmult(__ONE, &y01, -tauval, &Y02, &w12, ARMAS_NONE, conf);
    // w12 := Y02.T*a01
    armas_x_mvmult(__ZERO, &w12, __ONE, &Y02, &a01, ARMAS_TRANS, conf);
    // y01 := y01 - tauq*A02*(Y02.T*a01)
    armas_x_mvmult(__ONE, &y01, -tauval, &A02, &w12, ARMAS_NONE, conf);
    
    // beta := tauq*a01.T*y01
    beta = tauval * armas_x_dot(&a01, &y01, conf);
    // y01 := y01 - 0.5*beta*a01
    armas_x_axpy(&y01, -__HALF*beta, &a01, conf);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __continue_3x3to2x2(&YTL,  __nil,
                        __nil, &YBR, /**/  &Y00, &y11, &Y22, /**/  Y, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PTOP);
  }

  // restore superdiagonal value
  armas_x_set(A, ATL.rows-1, ATL.cols, v0);
  return err;
}



static
int __blk_trdreduce_upper(armas_x_dense_t *A, armas_x_dense_t *tauq,
                          armas_x_dense_t *Y, armas_x_dense_t *W,
                          int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A01, A11, A22;
  armas_x_dense_t YT, YB, Y0, Y1, Y2;
  armas_x_dense_t tT, tB, tq0, tq1, tq2;
  DTYPE v0;
  int err = 0;

  EMPTY(ATL);
  
  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&YT,
                  &YB,   /**/  Y, 0, ARMAS_PBOTTOM);
  __partition_2x1(&tT,
                  &tB,   /**/  tauq, 1, ARMAS_PBOTTOM);

  while (ATL.rows - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           __nil, &A11,  __nil,
                           __nil, __nil, &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&YT,
                           &Y0,
                           &Y1,
                           &Y2,  /**/  Y, lb, ARMAS_PTOP);
    __repartition_2x1to3x1(&tT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, lb, ARMAS_PTOP);
    // ------------------------------------------------------------------------
    if (__unblk_trdbuild_upper(&ATL, &tq1, &YT, W, conf)) {
      err = err == 0 ? -1 : err;
    }
    
    // set superdiagonal to unit
    v0 = armas_x_get(&A01, -1, 0);
    armas_x_set(&A01, -1, 0, 1.0);

    // A00 := A00 - A01*Y0.T - Y0*A01.T
    armas_x_update2_sym(__ONE, &A00, -__ONE, &A01, &Y0, ARMAS_UPPER, conf);

    // restore superdiagonal entry
    armas_x_set(&A01, -1, 0, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22, /**/  A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&YT,
                        &YB, /**/  &Y0, &Y1,  /**/ Y, ARMAS_PTOP);
    __continue_3x1to2x1(&tT,
                        &tB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PTOP);
  }

  if (ATL.rows > 0) {
    __unblk_trdreduce_upper(&ATL, &tT, W, TRUE, conf);
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
 * \param[in] W
 *      Workspace
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
int armas_x_trdreduce(armas_x_dense_t *A, armas_x_dense_t *tauq,
                      armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t Y;
  int wsmin, wsneed, lb, err = 0;
  if (!conf)
    conf = armas_conf_default();

  // default to lower triangular if uplo not defined
  if (!(flags & (ARMAS_LOWER|ARMAS_UPPER)))
    flags |= ARMAS_LOWER;

  if (A->rows != A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }    
  wsmin = __ws_trdreduce(A->rows, A->cols, 0);
  if (armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor to workspace
  lb = conf->lb;
  wsneed = __ws_trdreduce(A->rows, A->cols, conf->lb);
  if (armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_trdreduce);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    if (flags & ARMAS_LOWER) {
      err = __unblk_trdreduce_lower(A, tauq, W, conf);
    } else {
      err = __unblk_trdreduce_upper(A, tauq, W, FALSE, conf);
    }
  } else {
    armas_x_make(&Y, A->rows, lb, A->rows, armas_x_data(W));
    if (flags & ARMAS_LOWER) {
      err = __blk_trdreduce_lower(A, tauq, &Y, W, lb, conf);
    } else {
      err = __blk_trdreduce_upper(A, tauq, &Y, W, lb, conf);
    }
  }
  return err;
}

//! \brief Workspace size for trdreduce().
int armas_x_trdreduce_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_trdreduce(A->rows, A->cols, conf->lb);
}

/**
 * \brief Multiply matrix C with orthogonal matrix Q.
 *
 * \param[in,out] C
 *    On entry matrix C. On exit product of C and orthogonal matrix Q.
 * \param[in] A
 *    Orthogonal matrix C as elementary reflectors saved in upper (lower) triangular
 *    part of A. See trdreduce().
 * \param[in] tau
 *    Scalar coeffients of elementary reflectors.
 * \param[out] W
 *    Workspace
 * \param[in] flags
 *    Indicator flags, combination of *ARMAS_LOWER*, *ARMAS_UPPER*, *ARMAS_LEFT*,
 *    *ARMAS_RIGHT* and *ARMAS_TRANS*.
 * \param[in] conf
 *    Blocking configuration
 *
 * \retval 0 Sucess
 * \retval -1 Error
 */
int armas_x_trdmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t Ch, Qh, tauh;
  int err = 0;
  
  // default to multiplication from left is nothing defined
  if (!(flags & (ARMAS_LEFT|ARMAS_RIGHT)))
    flags |= ARMAS_LEFT;
  // default to lower triangular if uplo not defined
  if (!(flags & (ARMAS_LOWER|ARMAS_UPPER)))
    flags |= ARMAS_LOWER;

  if (flags & ARMAS_LOWER) {
    if (flags & ARMAS_LEFT) {
      armas_x_submatrix(&Ch, C, 1, 0, C->rows-1, C->cols);
    } else {
      armas_x_submatrix(&Ch, C, 0, 1, C->rows, C->cols-1);
    }
    armas_x_submatrix(&Qh, A, 1, 0, A->rows-1, A->rows-1);
    armas_x_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
    err = armas_x_qrmult(&Ch, &Qh, &tauh, W, flags, conf);
  } else {
    if (flags & ARMAS_LEFT) {
      armas_x_submatrix(&Ch, C, 0, 0, C->rows-1, C->cols);
    } else {
      armas_x_submatrix(&Ch, C, 0, 0, C->rows, C->cols-1);
    }
    armas_x_submatrix(&Qh, A, 0, 1, A->rows-1, A->rows-1);
    armas_x_submatrix(&tauh, tau, 0, 0,A->rows-1, 1);
    err = armas_x_qlmult(&Ch, &Qh, &tauh, W, flags, conf);
  }
  return err;
}

//! \brief Workspace size for trdmult().
int armas_x_trdmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
  if (flags & ARMAS_UPPER)
    return armas_x_qlmult_work(A, flags, conf);
  return armas_x_qrmult_work(A, flags, conf);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

