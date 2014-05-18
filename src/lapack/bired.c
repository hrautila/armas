
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_bdreduce) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) && defined(__armas_blas)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "partition.h"

static inline
int __ws_bdreduce(int M, int N, int lb)
{
  return lb <= 0 ? M+N : lb*(M+N);
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
int __unblk_bdreduce_left(__armas_dense_t *A, __armas_dense_t *tauq,
                          __armas_dense_t *taup, __armas_dense_t *W, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2, y21;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, z21;
  DTYPE v0, beta, tauqv, taupv;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    // temp vector for this round
    __armas_make(&y21, a12.cols, 1, a12.cols, __armas_data(W));
    __armas_make(&z21, a21.rows, 1, a21.rows, &__armas_data(W)[__armas_size(&y21)]);

    // compute householder to zero subdiagonal entries
    __compute_householder(&a11, &a21, &tq1, conf);
    tauqv = __armas_get(&tq1, 0, 0);
    
    // y21 := a12 + A22.T*a21
    __armas_axpby(&y21, &a12, 1.0, 0.0, conf);
    __armas_mvmult(&y21, &A22, &a21, 1.0, 1.0, ARMAS_TRANSA, conf);

    // a12 := a12 - tauq*y21
    __armas_axpy(&a12, &y21, -tauqv, conf);

    // compute householder to zero elements above 1st superdiagonal
    __compute_householder_vec(&a12, &tp1, conf);
    taupv = __armas_get(&tp1, 0, 0);
    v0    = __armas_get(&a12, 0, 0);
    __armas_set(&a12, 0, 0, 1.0);

    // [v == a12, u == a21]
    beta = __armas_dot(&y21, &a12, conf);
    // z21 := tauq*beta*v == tauq*beta*a21
    __armas_axpby(&z21, &a21, tauqv*beta, 0.0, conf);
    // z21 := A22*v - z21 == A22*a12 - z21
    __armas_mvmult(&z21, &A22, &a12, 1.0, -1.0, ARMAS_NONE, conf);
    // A22 := A22 - tauq*u*y21 == A22 - tauq*a21*y21
    __armas_mvupdate(&A22, &a21, &y21, -tauqv, conf);
    // A22 := A22 - taup*z21*v == A22 - taup*z21*a12
    __armas_mvupdate(&A22, &z21, &a12, -taupv, conf);

    __armas_set(&a12, 0, 0, v0);
    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1,  /**/ taup, ARMAS_PBOTTOM);
  }
}

/*
 * This is adaptation of BIRED_LAZY_UNB algorithm from (1).
 *
 * Y is workspace for building updates for first Householder.
 * And Z is space for building updates for second Householder
 * Y is n(A)-2,nb and Z is m(A)-1,nb  
 */
static
int __unblk_bdbuild_left(__armas_dense_t *A, __armas_dense_t *tauq,
                          __armas_dense_t *taup, __armas_dense_t *Y,
                          __armas_dense_t *Z, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABR, A00, a01, A02, a10, a11, a12, A20, a21, A22;
  __armas_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
  __armas_dense_t ZTL, ZBR, Z00, z10, z11, Z20, z21, Z22;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, w00;
  DTYPE v0, beta, tauqv, taupv, aa;
  int k;

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&YTL,  __nil,
                  __nil, &YBR,   /**/  Y, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&ZTL,  __nil,
                  __nil, &ZBR,   /**/  Z, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);

  for (k = 0; k < Y->cols; k++) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  &A02,
                           &a10,  &a11,  &a12,
                           &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&YTL,
                           &Y00,  __nil, __nil,
                           &y10,  &y11,  __nil,
                           &Y20 , &y21,  &Y22,  /**/  Y, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&ZTL,
                           &Z00,  __nil, __nil,
                           &z10,  &z11,  __nil,
                           &Z20 , &z21,  &Z22,  /**/  Z, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    __armas_submatrix(&w00, Z, 0, Z->cols-1, A20.cols, 1);
    // u10 == a10, U20 == A20, u21 == a21,
    // v10 == a01, V20 == A02, v21 == a12
    if (Y20.cols > 0) {
      // a11 = a11 - u10*y10 - z10*v10
      aa  = __armas_dot(&a10, &y10, conf);
      aa += __armas_dot(&z10, &a01, conf);
      __armas_set(&a11, 0, 0, __armas_get(&a11, 0, 0) - aa);
      // a21 := a21 - U20*y10 - Z20*v10
      __armas_mvmult(&a21, &A20, &y10, -1.0, 1.0, ARMAS_NONE, conf);
      __armas_mvmult(&a21, &Z20, &a01, -1.0, 1.0, ARMAS_NONE, conf);
      // a12 := a12 - u10.T*Y20.T - z10.T*V20.T
      __armas_mvmult(&a12, &Y20, &a10, -1.0, 1.0, ARMAS_NONE, conf);
      __armas_mvmult(&a12, &A02, &z10, -1.0, 1.0, ARMAS_TRANS, conf);
      // restore bidiagonal entry
      __armas_set(&a01, -1, 0, v0);
    }
    // compute householder to zero subdiagonal entries
    __compute_householder(&a11, &a21, &tq1, conf);
    tauqv = __armas_get(&tq1, 0, 0);

    // y21 := a12 + A22.T*u21 - Y20*U20.T*u21 - V20*Z20.T*u21
    __armas_axpby(&y21, &a12, 1.0, 0.0, conf);
    __armas_mvmult(&y21, &A22, &a21, 1.0, 1.0, ARMAS_TRANS, conf);
    // w00 := U20.T*u21 [= A20.T*a21]
    __armas_mvmult(&w00, &A20, &a21, 1.0, 0.0, ARMAS_TRANS, conf);
    // y21 := y21 - U20*w00 [U20 == A20]
    __armas_mvmult(&y21, &Y20, &w00, -1.0, 1.0, ARMAS_NONE, conf);
    // w00 := Z20.T*u21
    __armas_mvmult(&w00, &Z20, &a21, 1.0, 0.0, ARMAS_TRANS, conf);
    // y21 := y21 - V20*w00  [V20 == A02.T]
    __armas_mvmult(&y21, &A02, &w00, -1.0, 1.0, ARMAS_TRANS, conf);

    // a12 := a12 - tauq*y21
    __armas_scale(&y21, tauqv, conf);
    __armas_axpy(&a12, &y21, -1.0, conf);

    // compute householder to zero elements above 1st superdiagonal
    __compute_householder_vec(&a12, &tp1, conf);
    taupv = __armas_get(&tp1, 0, 0);
    v0    = __armas_get(&a12, 0, 0);
    __armas_set(&a12, 0, 0, 1.0);

    // z21 := taup*(A22*v - U20*Y20.T*v - Z20*V20.T*v - beta*u)
    // [v == a12, u == a21]
    beta = __armas_dot(&y21, &a12, conf);
    // z21 := beta*u
    __armas_axpby(&z21, &a21, beta, 0.0, conf);
    // w00 = Y20.T*v
    __armas_mvmult(&w00, &Y20, &a12, 1.0, 0.0, ARMAS_TRANS, conf);
    // z21 = z21 + U20*w00
    __armas_mvmult(&z21, &A20, &w00, 1.0, 1.0, ARMAS_NONE, conf);
    // w00 := V20.T*v  (V20.T == A02)
    __armas_mvmult(&w00, &A02, &a12, 1.0, 0.0, ARMAS_NONE, conf);
    // z21 := z21 + Z20*w00
    __armas_mvmult(&z21, &Z20, &w00, 1.0, 1.0, ARMAS_NONE, conf);

    // z21 := -taup*z21 + taup*A22*v
    __armas_mvmult(&z21, &A22, &a12, taupv, -taupv, ARMAS_NONE, conf);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&YTL,  __nil,
                        __nil, &YBR, /**/  &Y00, &y11, &Y22, /**/  Y, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&ZTL,  __nil,
                        __nil, &ZBR, /**/  &Z00, &z11, &Z22, /**/  Z, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1,  /**/ taup, ARMAS_PBOTTOM);
  }
  // restore bidiagonal entry
  __armas_set(&ATR, -1, 0, v0);
  return 0;
}


static
int __blk_bdreduce_left(__armas_dense_t *A,
                        __armas_dense_t *tauq, __armas_dense_t *taup,
                        __armas_dense_t *W, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A20, A21, A22;
  __armas_dense_t Y, YT, YB, Y0, Y1, Y2;
  __armas_dense_t Z, ZT, ZB, Z0, Z1, Z2;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, w00;
  DTYPE v0;
  int k;

  __armas_make(&Z, A->rows, lb, A->rows, __armas_data(W));
  __armas_make(&Y, A->cols, lb, A->cols, &__armas_data(W)[__armas_size(&Z)]);
  
  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);
  __partition_2x1(&YT,
                  &YB,    /**/  &Y, 0, ARMAS_PTOP);
  __partition_2x1(&ZT,
                  &ZB,    /**/  &Z, 0, ARMAS_PTOP);

  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&YT,
                           &Y0,
                           &Y1,
                           &Y2,   /**/  &Y, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&ZT,
                           &Z0,
                           &Z1,
                           &Z2,   /**/  &Z, lb, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    __unblk_bdbuild_left(&ABR, &tq1, &tp1, &YB, &ZB, conf);

    // set superdiagonal entry to one
    v0  = __armas_get(&A12, A12.rows-1, 0);
    __armas_set(&A12, A12.rows-1, 0, 1.0);

    // A22 := A22 - U2*Y2.T
    __armas_mult(&A22, &A21, &Y2, -1.0, 1.0, ARMAS_TRANSB, conf);
    // A22 := A22 - Z2*V2.T
    __armas_mult(&A22, &Z2, &A12,  -1.0, 1.0, ARMAS_NONE, conf);

    // restore super-diagonal entry
    __armas_set(&A12, A12.rows-1, 0, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&YT,
                        &YB,  /**/  &Y0, &Y1,   /**/ &Y, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&ZT,
                        &ZB,  /**/  &Z0, &Z1,   /**/ &Z, ARMAS_PBOTTOM);
  }

  if (ABR.cols > 0) {
    __unblk_bdreduce_left(&ABR, &tqB, &tpB, W, conf);
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
int __unblk_bdreduce_right(__armas_dense_t *A, __armas_dense_t *tauq,
                           __armas_dense_t *taup, __armas_dense_t *W, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2, y21;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, z21;
  DTYPE v0, beta, tauqv, taupv, aa;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    // temp vector for this round
    __armas_make(&y21, a21.rows, 1, a21.rows, __armas_data(W));
    __armas_make(&z21, a12.cols, 1, a12.cols, &__armas_data(W)[__armas_size(&y21)]);

    // compute householder to zero subdiagonal entries
    __compute_householder(&a11, &a12, &tp1, conf);
    taupv = __armas_get(&tp1, 0, 0);
    
    // y21 := a12 + A22.T*a12
    __armas_axpby(&y21, &a21, 1.0, 0.0, conf);
    __armas_mvmult(&y21, &A22, &a12, 1.0, 1.0, ARMAS_NONE, conf);

    // a21 := a21 - taup*y21
    __armas_axpy(&a21, &y21, -taupv, conf);

    // compute householder to zero elements above 1st superdiagonal
    __compute_householder_vec(&a21, &tq1, conf);
    tauqv = __armas_get(&tq1, 0, 0);

    v0    = __armas_get(&a21, 0, 0);
    __armas_set(&a21, 0, 0, 1.0);

    // [v == a21, u == a12]
    beta = __armas_dot(&y21, &a21, conf);
    // z21 := taup*beta*a12
    __armas_axpby(&z21, &a12, taupv*beta, 0.0, conf);
    // z21 := A22*a21 - z21
    __armas_mvmult(&z21, &A22, &a21, 1.0, -1.0, ARMAS_TRANS, conf);
    // A22 := A22 - taup*y21*a12
    __armas_mvupdate(&A22, &y21, &a12, -taupv, conf);
    // A22 := A22 - tauq*z21*a21
    __armas_mvupdate(&A22, &a21, &z21, -tauqv, conf);

    __armas_set(&a21, 0, 0, v0);
    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1,  /**/ taup, ARMAS_PBOTTOM);
  }
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
int __unblk_bdbuild_right(__armas_dense_t *A, __armas_dense_t *tauq,
                          __armas_dense_t *taup, __armas_dense_t *Y,
                          __armas_dense_t *Z, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, A00, a01, A02, a10, a11, a12, A20, a21, A22;
  __armas_dense_t YTL, YBR, Y00, y10, y11, Y20, y21, Y22;
  __armas_dense_t ZTL, ZBR, Z00, z10, z11, Z20, z21, Z22;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, w00;
  DTYPE v0, beta, tauqv, taupv, aa;
  int k;

  __partition_2x2(&ATL,  __nil,
                  &ABL,  &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&YTL,  __nil,
                  __nil, &YBR,   /**/  Y, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&ZTL,  __nil,
                  __nil, &ZBR,   /**/  Z, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);

  for (k = 0; k < Y->cols; k++) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  &A02,
                           &a10,  &a11,  &a12,
                           &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&YTL,
                           &Y00,  __nil, __nil,
                           &y10,  &y11,  __nil,
                           &Y20 , &y21,  &Y22,  /**/  Y, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&ZTL,
                           &Z00,  __nil, __nil,
                           &z10,  &z11,  __nil,
                           &Z20 , &z21,  &Z22,  /**/  Z, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, 1, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, 1, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    __armas_submatrix(&w00, Z, 0, Z->cols-1, A02.rows, 1);
    // u10 == a10, U20 == A20, u21 == a21,
    // v10 == a01, V20 == A02, v21 == a12
    if (Y20.cols > 0) {
      // a11 = a11 - u10*y10 - z10*v10
      aa  = __armas_dot(&a10, &z10, conf);
      aa += __armas_dot(&y10, &a01, conf);
      __armas_set(&a11, 0, 0, __armas_get(&a11, 0, 0) - aa);
      // a12 := a12 - V20*y10 - Z20*u10
      __armas_mvmult(&a12, &A02, &y10, -1.0, 1.0, ARMAS_TRANS, conf);
      __armas_mvmult(&a12, &Z20, &a10, -1.0, 1.0, ARMAS_NONE, conf);
      // a21 := a21 - Y20*v10 - U20*z10
      __armas_mvmult(&a21, &Y20, &a01, -1.0, 1.0, ARMAS_NONE, conf);
      __armas_mvmult(&a21, &A20, &z10, -1.0, 1.0, ARMAS_NONE, conf);
      // restore bidiagonal entry
      __armas_set(&a10, 0, -1, v0);
    }
    // compute householder to zero subdiagonal entries
    __compute_householder(&a11, &a12, &tp1, conf);
    taupv = __armas_get(&tp1, 0, 0);

    // y21 := a12 + A22*v21 - Y20*U20.T*v21 - V20*Z20.T*v21
    __armas_axpby(&y21, &a21, 1.0, 0.0, conf);
    __armas_mvmult(&y21, &A22, &a12, 1.0, 1.0, ARMAS_NONE, conf);
    // w00 := U20.T*v21 [= A02*a12]
    __armas_mvmult(&w00, &A02, &a12, 1.0, 0.0, ARMAS_NONE, conf);
    // y21 := y21 - U20*w00 [U20 == A20]
    __armas_mvmult(&y21, &Y20, &w00, -1.0, 1.0, ARMAS_NONE, conf);
    // w00 := Z20.T*v21
    __armas_mvmult(&w00, &Z20, &a12, 1.0, 0.0, ARMAS_TRANS, conf);
    // y21 := y21 - V20*w00  
    __armas_mvmult(&y21, &A20, &w00, -1.0, 1.0, ARMAS_NONE, conf);

    // a21 := a21 - taup*y21
    __armas_scale(&y21, taupv, conf);
    __armas_axpy(&a21, &y21, -1.0, conf);

    // compute householder to zero elements below 1st subdiagonal
    __compute_householder_vec(&a21, &tq1, conf);
    tauqv = __armas_get(&tq1, 0, 0);
    v0    = __armas_get(&a21, 0, 0);
    __armas_set(&a21, 0, 0, 1.0);

    // z21 := tauq*(A22*y - U20*Y20.T*u - Z20*V20.T*u - beta*v)
    // [v == a12, u == a21]
    beta = __armas_dot(&y21, &a21, conf);
    // z21 := beta*v
    __armas_axpby(&z21, &a12, beta, 0.0, conf);
    // w00 = Y20.T*u
    __armas_mvmult(&w00, &Y20, &a21, 1.0, 0.0, ARMAS_TRANS, conf);
    // z21 = z21 + V20*w00 (V20 == A02.T)
    __armas_mvmult(&z21, &A02, &w00, 1.0, 1.0, ARMAS_TRANS, conf);
    // w00 := U20.T*v  (U20.T == A20.T)
    __armas_mvmult(&w00, &A20, &a21, 1.0, 0.0, ARMAS_TRANS, conf);
    // z21 := z21 + Z20*w00
    __armas_mvmult(&z21, &Z20, &w00, 1.0, 1.0, ARMAS_NONE, conf);

    // z21 := -tauq*z21 + tauq*A22*v
    __armas_mvmult(&z21, &A22, &a21, tauqv, -tauqv, ARMAS_TRANS, conf);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil, 
                        &ABL,  &ABR, /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&YTL,  __nil,
                        __nil, &YBR, /**/  &Y00, &y11, &Y22, /**/  Y, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&ZTL,  __nil,
                        __nil, &ZBR, /**/  &Z00, &z11, &Z22, /**/  Z, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1,  /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1,  /**/ taup, ARMAS_PBOTTOM);
  }
  // restore bidiagonal entry
  __armas_set(&ABL, 0, -1, v0);
  return 0;
}

static
int __blk_bdreduce_right(__armas_dense_t *A, __armas_dense_t *tauq,
                         __armas_dense_t *taup, __armas_dense_t *W,
                         int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A20, A21, A22;
  __armas_dense_t Y, YT, YB, Y0, Y1, Y2;
  __armas_dense_t Z, ZT, ZB, Z0, Z1, Z2;
  __armas_dense_t tqT, tqB, tq0, tq1, tq2;
  __armas_dense_t tpT, tpB, tp0, tp1, tp2, w00;
  DTYPE v0;

  __armas_make(&Z, A->cols, lb, A->cols, __armas_data(W));
  __armas_make(&Y, A->rows, lb, A->rows, &__armas_data(W)[__armas_size(&Z)]);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tqT,
                  &tqB,   /**/  tauq, 0, ARMAS_PTOP);
  __partition_2x1(&tpT,
                  &tpB,   /**/  taup, 0, ARMAS_PTOP);
  __partition_2x1(&YT,
                  &YB,    /**/  &Y, 0, ARMAS_PTOP);
  __partition_2x1(&ZT,
                  &ZB,    /**/  &Z, 0, ARMAS_PTOP);

  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tqT,
                           &tq0,
                           &tq1,
                           &tq2,  /**/  tauq, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&tpT,
                           &tp0,
                           &tp1,
                           &tp2,  /**/  taup, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&YT,
                           &Y0,
                           &Y1,
                           &Y2,   /**/  &Y, lb, ARMAS_PBOTTOM);
    __repartition_2x1to3x1(&ZT,
                           &Z0,
                           &Z1,
                           &Z2,   /**/  &Z, lb, ARMAS_PBOTTOM);
    // ------------------------------------------------------------------------
    __unblk_bdbuild_right(&ABR, &tq1, &tp1, &YB, &ZB, conf);

    // set superdiagonal entry to one
    v0  = __armas_get(&A21, 0, A21.cols-1);
    __armas_set(&A21, 0, A21.cols-1, 1.0);

    // A22 := A22 - U2*Z2.T
    __armas_mult(&A22, &A21, &Z2, -1.0, 1.0, ARMAS_TRANSB, conf);
    // A22 := A22 - Y2*V2.T
    __armas_mult(&A22, &Y2, &A12,  -1.0, 1.0, ARMAS_NONE, conf);

    // restore super-diagonal entry
    __armas_set(&A21, 0, A21.cols-1, v0);

    // ------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tqT,
                        &tqB, /**/  &tq0, &tq1, /**/ tauq, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&tpT,
                        &tpB, /**/  &tp0, &tp1, /**/ taup, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&YT,
                        &YB,  /**/  &Y0, &Y1,   /**/ &Y, ARMAS_PBOTTOM);
    __continue_3x1to2x1(&ZT,
                        &ZB,  /**/  &Z0, &Z1,   /**/ &Z, ARMAS_PBOTTOM);
  }

  if (ABR.cols > 0) {
    __unblk_bdreduce_right(&ABR, &tqB, &tpB, W, conf);
  }
  return 0;
}



/*
 * Reduce a general M-by-N matrix A to upper or lower bidiagonal form B
 * by an ortogonal transformation A = Q*B*P.T,  B = Q.T*A*P
 *
 *
 * Arguments
 *   A     On entry, the real M-by-N matrix. On exit the upper/lower
 *         bidiagonal matrix and ortogonal matrices Q and P.
 *
 *   tauq  Scalar factors for elementary reflector forming the
 *         ortogonal matrix Q.
 *
 *   taup  Scalar factors for elementary reflector forming the
 *         ortogonal matrix P.
 *
 *   W     Workspace needed for reduction.
 *
 *   conf  Current blocking configuration. Optional.
 *
 *
 * Details
 *
 * Matrices Q and P are products of elementary reflectors H(k) and G(k)
 *
 * If M > N:
 *     Q = H(1)*H(2)*...*H(N)   and P = G(1)*G(2)*...*G(N-1)
 *
 * where H(k) = 1 - tauq*u*u.T and G(k) = 1 - taup*v*v.T
 *
 * Elementary reflector H(k) are stored on columns of A below the diagonal with
 * implicit unit value on diagonal entry. Vector TAUQ holds corresponding scalar
 * factors. Reflector G(k) are stored on rows of A right of first superdiagonal
 * with implicit unit value on superdiagonal. Corresponding scalar factors are
 * stored on vector TAUP.
 * 
 * If M < N:
 *   Q = H(1)*H(2)*...*H(N-1)   and P = G(1)*G(2)*...*G(N)
 *
 * where H(k) = 1 - tauq*u*u.T and G(k) = 1 - taup*v*v.T
 *
 * Elementary reflector H(k) are stored on columns of A below the first sub diagonal 
 * with implicit unit value on sub diagonal entry. Vector TAUQ holds corresponding 
 * scalar factors. Reflector G(k) are sotre on rows of A right of diagonal with
 * implicit unit value on superdiagonal. Corresponding scalar factors are stored
 * on vector TAUP.
 *
 * Contents of matrix A after reductions are as follows.
 *
 *    M = 6 and N = 5:                  M = 5 and N = 6:
 *
 *    (  d   e   v1  v1  v1 )           (  d   v1  v1  v1  v1  v1 )
 *    (  u1  d   e   v2  v2 )           (  e   d   v2  v2  v2  v2 )
 *    (  u1  u2  d   e   v3 )           (  u1  e   d   v3  v3  v3 )
 *    (  u1  u2  u3  d   e  )           (  u1  u2  e   d   v4  v4 )
 *    (  u1  u2  u3  u4  d  )           (  u1  u2  u3  e   d   v5 )
 *    (  u1  u2  u3  u4  u5 )
 */
int __armas_bdreduce(__armas_dense_t *A, __armas_dense_t *tauq,
                     __armas_dense_t *taup, __armas_dense_t *W,
                     armas_conf_t *conf)
{
  int wsmin, wsneed, lb;

  if (!conf)
    conf = armas_conf_default();
  
  lb = conf->lb;
  wsmin = __ws_bdreduce(A->rows, A->cols, 0);
  if (__armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_bdreduce(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, wsneed, __ws_bdreduce);
    lb = min(lb, conf->lb);
  }


  if (A->rows >= A->cols) {
    if (lb > 0 && A->cols > lb) {
      __blk_bdreduce_left(A, tauq, taup, W, lb, conf);
    } else {
      __unblk_bdreduce_left(A, tauq, taup, W, conf);
    }
  } else {
    if (lb > 0 && A->cols > lb) {
      __blk_bdreduce_right(A, tauq, taup, W, lb, conf);
    } else {
      __unblk_bdreduce_right(A, tauq, taup, W, conf);
    }
  }
  return 0;
}

/*
 */
int __armas_bdreduce_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_bdreduce(A->rows, A->cols, conf->lb);
}

/*
 * Multiply and replace C with product of C and Q or P where Q and P are real orthogonal
 * matrices defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)   and   P = G(1) G(2). . . G(k)
 *
 * as returned by bdreduce().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     Bidiagonal reduction as returned by bdreduce() where the lower trapezoidal
 *        part, on and below first subdiagonal, holds the product Q. The upper
 *        trapezoidal part holds the product P.
 *
 *  tau   The scalar factors of the elementary reflectors. If flag MULTQ is set then holds
 *        scalar factors for Q. If flag MULTP is set then holds scalar factors for P.
 *        Expected to be column vector is size min(M(A), N(A)).
 *
 *  bits  Indicators, valid bits LEFT, RIGHT, TRANS, MULTQ, MULTP
 *
 *
 * Additional information
 *
 *   If M(A) >= N(A) then order N(Q) of orthogonal matrix Q is M(A) and the order N(P)
 *   of the orthogonal matrix P is N(A). If M(A) < N(A) then order N(Q) of Q is N(A) and
 *   order N(P) of P is M(A).
 *
 *        flags              result
 *        ------------------------------------------
 *        MULTQ,LEFT         C = Q*C     n(A) == m(C)
 *        MULTQ,RIGHT        C = C*Q     n(C) == m(A)
 *        MULTQ,TRANS,LEFT   C = Q.T*C   n(A) == m(C)
 *        MULTQ,TRANS,RIGHT  C = C*Q.T   n(C) == m(A)
 *        MULTP,LEFT         C = P*C     n(A) == m(C)
 *        MULTP,RIGHT        C = C*P     n(C) == m(A)
 *        MULTP,TRANS,LEFT   C = P.T*C   n(A) == m(C)
 *        MULTP,TRANS,RIGHT  C = C*P.T   n(C) == m(A)
 *
 */
int __armas_bdmult(__armas_dense_t *C, __armas_dense_t *A,
                   __armas_dense_t *tau, __armas_dense_t *W,
                   int flags, armas_conf_t *conf)
{
  __armas_dense_t Qh, Ch, Ph, tauh;
  int err, ok;

  // to be done!
  switch (flags & (ARMAS_MULTQ|ARMAS_MULTP|ARMAS_LEFT|ARMAS_RIGHT)) {
  case ARMAS_MULTQ|ARMAS_LEFT:
  case ARMAS_MULTQ:
    break;
  case ARMAS_MULTQ|ARMAS_RIGHT:
    break;
  case ARMAS_MULTP|ARMAS_LEFT:
  case ARMAS_MULTP:
    break;
  case ARMAS_MULTP|ARMAS_RIGHT:
    break;
  default:
    break;
  }

  if (flags & ARMAS_MULTP) {
    flags = flags & ARMAS_TRANS ? flags^ARMAS_TRANS : flags|ARMAS_TRANS;
  } 

  if (A->rows >= A->cols) {
    // M >= N
    switch (flags & (ARMAS_MULTQ|ARMAS_MULTP)) {
    case ARMAS_MULTQ:
      __armas_submatrix(&tauh, tau, 0, 0, A->cols, 1);
      err = __armas_qrmult(C, A, &tauh, W, flags, conf);
      break;
    case ARMAS_MULTP:
      __armas_submatrix(&Ph, A, 0, 1, A->cols-1, A->cols-1);
      __armas_submatrix(&tauh, tau, 0, 0, A->cols-1, 1);
      if (flags & ARMAS_RIGHT) {
        __armas_submatrix(&Ch, C, 0, 1, C->rows, C->cols-1);
      } else {
        __armas_submatrix(&Ch, C, 1, 0, C->rows-1, C->cols);
      }
      err = __armas_lqmult(&Ch, &Ph, &tauh, W, flags, conf);
      break;
    default:
      conf->error = ARMAS_EINVAL;
      return -1;
    }
  } else {
    // M < N
    switch (flags & (ARMAS_MULTQ|ARMAS_MULTP)) {
    case ARMAS_MULTQ:
      __armas_submatrix(&Qh, A, 1, 0, A->rows-1, A->rows-1);
      __armas_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
      if (flags & ARMAS_RIGHT) {
        __armas_submatrix(&Ch, C, 0, 1, C->rows, C->cols-1);
      } else {
        __armas_submatrix(&Ch, C, 1, 0, C->rows-1, C->cols);
      }
      err = __armas_qrmult(&Ch, &Qh, &tauh, W, flags, conf);
      break;
    case ARMAS_MULTP:
      __armas_submatrix(&tauh, tau, 0, 0, A->rows, 1);
      err = __armas_lqmult(C, A, &tauh, W, flags, conf);
      break;
    default:
      conf->error = ARMAS_EINVAL;
      return -1;
    }
  }
  return err;
}

/*
 */
int __armas_bdmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
  int nq, np;
  nq = __armas_qrmult_work(A, flags, conf);
  np = __armas_lqmult_work(A, flags, conf);
  return np > nq ? np : nq;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

