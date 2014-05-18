
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_hessreduce)  && defined(__armas_hessmult)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) && defined(__update_qr_left) && defined(__armas_qrmult)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

static inline
int __ws_hess_reduce(int M, int N, int lb)
{
  return lb == 0 ? M : lb*(M + lb);
}

/*
 * (1) Quintana-Orti, van de Geijn:
 *     Improving the Performance of Reduction to Hessenberg form, 2006
 *
 * (2) Van Zee, van de Geijn, Quintana-Orti:
 *     Algorithms for Reducing a Matrix to Condensed Form, 2010, FLAME working notes #53
 *     
 *                              I(k)|  0  
 * Householder reflector H(k) = --------- 
 *                               0  | H(k)
 *
 * update: H(k)*A*H(k) =
 *
 *           I | 0 | 0    A00 | a01 | A02   I | 0 | 0      A00 | a01 |  A02*H
 *           ---------    ---------------   ---------      -------------------
 *           0 | 1 | 0 *  a10 | a11 | a12 * 0 | 1 | 0  ==  a10 | a11 |  a12*H
 *           ---------    ---------------   ---------     --------------------
 *           0 | 0 | H    A20 | a21 | A22   0 | 0 | H      A20 | a21 | H*A22*H
 *
 * For blocked version elementary reflectors are combined to block reflector
 * H = I - Y*T*Y.T  Elementary reflectors are computed to block [A11; A21].T with
 * unblocked algorithm and the block A01 needs to be updated afterwards.
 * (Not during the reflector computation as happens in unblocked version)
 *
 * Approximate flops needed: (10/3)*N^3
 */



/*
 * Computes upper Hessenberg reduction of N-by-N matrix A using unblocked
 * algorithm as described in (1).
 *
 * Hessenberg reduction: A = Q.T*B*Q, Q unitary, B upper Hessenberg
 *  Q = H(0)*H(1)*...*H(k) where H(k) is k'th Householder reflector.
 *
 * Compatible with lapack.DGEHD2.
 */
static
int __unblk_hess_gqvdg(__armas_dense_t *A, __armas_dense_t *tau,
                       __armas_dense_t *W, int row, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  __armas_dense_t AL, AR, A0, a1, A2;
  __armas_dense_t tT, tB, t0, t1, t2, w12, v1;
  DTYPE tauval, beta;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, row, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&AL, &AR,      /**/  A, 0, ARMAS_PLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  __armas_submatrix(&v1, W, 0, 0, A->rows, 1);

  while (ABR.rows > 1 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_1x2to1x3(&AL,
                           &A0,  &a1,  &A2,     /**/  A, 1, ARMAS_PRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    __compute_householder_vec(&a21, &t1, conf);
    tauval = __armas_get(&t1, 0, 0);
    beta   = __armas_get(&a21, 0, 0);
    __armas_set(&a21, 0, 0, 1.0);

    // v1 := A2*a21
    __armas_mvmult(&v1, &A2, &a21, 1.0, 0.0, ARMAS_NONE, conf);
    
    // A2 := A2 - tau*v1*a21  (A2 = A2*H(k))
    __armas_mvupdate(&A2, &v1, &a21, -tauval, conf);
    
    __armas_submatrix(&w12, W, 0, 0, A22.cols, 1);
    // w12 := a21.T*A22 = A22.T*a21
    __armas_mvmult(&w12, &A22, &a21, 1.0, 0.0, ARMAS_TRANS, conf);
    // A22 := A22 - tau*a21*w12  (A22 = H(k)*A22)
    __armas_mvupdate(&A22, &a21, &w12, -tauval, conf);

    // restore a21[0]
    __armas_set(&a21, 0, 0, beta);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&AL,   &AR,  /**/  &A0,  &a1,  A, ARMAS_PRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}

/*
 * Update vector with compact WY Householder block
 *   (I - Y*T*Y.T)*v  = v - Y*T*Y.T*v 
 *
 * LEFT:
 *    1 | 0 * v0 = v0     = v0
 *    0 | Q   v1   Q*v1   = v1 - Y*T*Y.T*v1
 *
 *    1 | 0   * v0 = v0     = v0
 *    0 | Q.T   v1   Q.T*v1 = v1 - Y*T.T*Y.T*v1
 */
static
int __update_vec_left_wy(__armas_dense_t *v, __armas_dense_t *Y1, __armas_dense_t *Y2,
                         __armas_dense_t *T, __armas_dense_t *W, int bits,
                         armas_conf_t *conf)
{
  __armas_dense_t v1, v2, w0;
  
  __armas_submatrix(&v1, v, 1, 0, Y1->cols, 1);
  __armas_submatrix(&v2, v, 1+Y1->cols, 0, Y2->rows, 1);
  __armas_submatrix(&w0, W, 0, 0, Y1->rows, 1);
  
  // w0 = Y1.T*v1 + Y2.T*v2
  __armas_axpby(&w0, &v1, 1.0, 0.0, conf);
  __armas_mvmult_trm(&w0, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANS, conf);
  __armas_mvmult(&w0, Y2, &v2, 1.0, 1.0, ARMAS_TRANS, conf);
  
  // w0 = opt(T)*w0
  __armas_mvmult_trm(&w0, T, 1.0, bits|ARMAS_UPPER, conf);
  
  // v2 = v2 - Y2*w0
  __armas_mvmult(&v2, Y2, &w0, -1.0, 1.0, ARMAS_NONE, conf);
  
  // v1 = v1 - Y1*w0
  __armas_mvmult_trm(&w0, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT, conf);
  __armas_axpy(&v1, &w0, -1.0, conf);
  return 0;
}


/*
 * 
 *  Building reduction block for blocked algorithm as described in (1).
 *
 *  A. update next column
 *    a10        [(U00)     (U00)  ]   [(a10)    (V00)            ]
 *    a11 :=  I -[(u10)*T00*(u10).T] * [(a11)  - (v01) * T00 * a10]
 *    a12        [(U20)     (U20)  ]   [(a12)    (V02)            ]
 *
 *  B. compute Householder reflector for updated column
 *    a21, t11 := Householder(a21)
 *
 *  C. update intermediate reductions
 *    v10      A02*a21
 *    v11  :=  a12*a21
 *    v12      A22*a21
 *
 *  D. update block reflector
 *    t01 :=  A20*a21
 *    t11 :=  t11
 */
static
int __unblk_build_hess_gqvdg(__armas_dense_t *A, __armas_dense_t *T,
                             __armas_dense_t *V, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
  __armas_dense_t AL, AR, A0, a1, A2;
  __armas_dense_t VL, VR, V0, v1, V2, Y0;
  __armas_dense_t TTL, TBR, T00, t01, t11, T22;
  DTYPE beta, tauval;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&TTL,  __nil,
                  __nil, &TBR,   /**/  T, 0, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&AL, &AR,      /**/  A, 0, ARMAS_PLEFT);
  __partition_1x2(&VL, &VR,      /**/  V, 0, ARMAS_PLEFT);
                 
  while (VR.cols > 0 ) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &a10,  &a11,  __nil,
                           &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&TTL,
                           &T00,   &t01, __nil,
                           __nil,  &t11, __nil,
                           __nil, __nil, &T22,  /**/  T, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_1x2to1x3(&AL,
                           &A0,  &a1,  &A2,     /**/  A, 1, ARMAS_PRIGHT);
    __repartition_1x2to1x3(&VL,
                           &V0,  &v1,  &V2,     /**/  V, 1, ARMAS_PRIGHT);
    // ---------------------------------------------------------------------
    if (V0.cols > 0) {
      // y10 := T00*a10 (t01 is workspace)
      __armas_axpby(&t01, &a10, 1.0, 0.0, conf);
      __armas_mvmult_trm(&t01, &T00, 1.0, ARMAS_UPPER, conf);
      
      // a1 = a1 - V0*T00*a10
      __armas_mvmult(&a1, &V0, &t01, -1.0, 1.0, ARMAS_NONE, conf);
      
      // update a1 = (I - Y*T*Y.T).T*a1
      __armas_submatrix(&Y0, A, 1, 0, A00.cols, A00.cols);
      __update_vec_left_wy(&a1, &Y0, &A20, &T00, &t01, ARMAS_TRANS, conf);
      // restore last entry of a10, 
      __armas_set(&a10, 0, -1, beta);
    }
    // compute householder
    __compute_householder_vec(&a21, &t11, conf);
    beta = __armas_get(&a21, 0, 0);
    __armas_set(&a21, 0, 0, 1.0);
    
    // v1 = A2*a21
    __armas_mvmult(&v1, &A2, &a21, 1.0, 0.0, ARMAS_NONE, conf);
    // update T
    tauval = __armas_get(&t11, 0, 0);
    if (tauval != 0) {
      // t01 := -tauval*A20.T*a21
      __armas_mvmult(&t01, &A20, &a21, -tauval, 0.0, ARMAS_TRANS, conf);
      // t01 := T00*t01
      __armas_mvmult_trm(&t01, &T00, 1.0, ARMAS_UPPER, conf);
    }

    // ---------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&TTL,  __nil,
                        __nil, &TBR, /**/  &T00, &t11, &T22,   T, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&AL,   &AR,  /**/  &A0,  &a1,   A, ARMAS_PRIGHT);
    __continue_1x3to1x2(&VL,   &VR,  /**/  &V0,  &v1,   V, ARMAS_PRIGHT);
  }

  __armas_set(A, V->cols, V->cols-1, beta);
  return 0;
}

// compute: (I - Y*T*Y.T).T*C
static
int __update_hess_left_wy(__armas_dense_t *C, __armas_dense_t *Y1, __armas_dense_t *Y2,
                          __armas_dense_t *T, __armas_dense_t *V, armas_conf_t *conf)
{
  __armas_dense_t C1, C2;
  if (__armas_size(C) == 0) {
    return 0;
  }
  __armas_submatrix(&C1, C, 1, 0, Y1->rows, C->cols);
  __armas_submatrix(&C2, C, 1+Y1->cols, 0, Y2->rows, C->cols);
  
  __update_qr_left(&C1, &C2, Y1, Y2, T, V, TRUE, conf);
  return 0;
}

// compute: C*(I - Y*T*Y.T)
static
int __update_hess_right_wy(__armas_dense_t *C, __armas_dense_t *Y1, __armas_dense_t *Y2,
                          __armas_dense_t *T, __armas_dense_t *V, armas_conf_t *conf)
{
  __armas_dense_t C1, C2;
  if (__armas_size(C) == 0) {
    return 0;
  }
  __armas_submatrix(&C1, C, 0, 1, C->rows, Y1->cols);
  __armas_submatrix(&C2, C, 0, 1+Y1->cols, C->rows, Y2->rows);
  
  __update_qr_right(&C1, &C2, Y1, Y2, T, V, FALSE, conf);
  return 0;
}

/*
 * Blocked version of Hessenberg reduction algorithm as presented in (1). This
 * version uses compact-WY transformation.
 *
 * Some notes:
 *
 * Elementary reflectors stored in [A11; A21].T are not on diagonal of A11. Update of
 * a block aligned with A11; A21 is as follow
 *
 * 1. Update from left Q(k)*C:
 *                                         c0   0                            c0
 * (I - Y*T*Y.T).T*C = C - Y*(C.T*Y)*T.T = C1 - Y1 * (C1.T.Y1+C2.T*Y2)*T.T = C1-Y1*W
 *                                         C2   Y2                           C2-Y2*W
 *
 * where W = (C1.T*Y1+C2.T*Y2)*T.T and first row of C is not affected by update
 * 
 * 2. Update from right C*Q(k):
 *                                       0
 * C - C*Y*T*Y.T = c0;C1;C2 - c0;C1;C2 * Y1 *T*(0;Y1;Y2) = c0; C1-W*Y1; C2-W*Y2
 *                                       Y2
 * where  W = (C1*Y1 + C2*Y2)*T and first column of C is not affected
 *
 */
static
int __blk_hess_gqvdg(__armas_dense_t *A, __armas_dense_t *tau,
                     __armas_dense_t *T, __armas_dense_t *V, int lb,
                     armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABR, A00, A11, A12, A21, A22, A2;
  __armas_dense_t VT, VB, Y1, Y2, W0;
  __armas_dense_t tT, tB, t0, t1, t2, td;
  DTYPE beta;

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  __armas_diag(&td, T, 0);
  while (ABR.rows > lb+1 && ABR.cols > lb) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, ARMAS_PBOTTOM);
    __partition_2x1(&VT,
                    &VB,   /**/ V, ATL.rows, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __unblk_build_hess_gqvdg(&ABR, T, &VB, conf);
    // t1 = diag(T)
    __armas_axpby(&t1, &td, 1.0, 0.0, conf);

    __armas_submatrix(&Y1, &ABR, 1, 0, A11.cols, A11.cols);
    __armas_submatrix(&Y2, &ABR, 1+A11.cols, 0, A21.rows-1, A11.cols);
    
    // [A01, A02] == ATR := ATR*(I - Y*T*Y.T);
    __update_hess_right_wy(&ATR, &Y1, &Y2, T, &VT, conf);

    __merge2x1(&A2, &A12, &A22);
    
    // A2 := A2 - VB*T*A21.T
    beta = __armas_get(&A21, 0, -1);
    __armas_set(&A21, 0, -1, 1.0);
    __armas_mult_trm(&VB, T, 1.0, ARMAS_UPPER|ARMAS_RIGHT, conf);
    __armas_mult(&A2, &VB, &A21, -1.0, 1.0, ARMAS_TRANSB, conf);
    __armas_set(&A21, 0, -1, beta);
    
    // A2 := (I - Y*T*Y.T).T * A2
    __armas_submatrix(&W0, V, 0, 0, A2.cols, Y2.cols);
    __update_hess_left_wy(&A2, &Y1, &Y2, T, &W0, conf);

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }

  if (ABR.rows > 1) {
    __merge2x1(&A2, &ATR, &ABR);
    __armas_make(&W0, A->rows, 1, A->rows, __armas_data(V));
    __unblk_hess_gqvdg(&A2, &tB, &W0, ATR.rows, conf);
  }
  return 0;
}

/*
 * Reduce general matrix A to upper Hessenberg form H by similiarity
 * transformation H = Q.T*A*Q.
 *
 * Arguments:
 *  A    On entry, the general matrix A. On exit, the elements on and
 *       above the first subdiagonal contain the reduced matrix H.
 *       The elements below the first subdiagonal with the vector tau
 *       represent the ortogonal matrix A as product of elementary reflectors.
 *
 *  tau  On exit, the scalar factors of the elementary reflectors.
 *
 *  W    Workspace, as defined by WorksizeHess()
 *
 *  conf The blocking configration. 
 * 
 * ReduceHess is compatible with lapack.DGEHRD.
 */
int __armas_hessreduce(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                       armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_hess_reduce(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_hess_reduce(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, wsneed, __ws_hess_reduce);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    __unblk_hess_gqvdg(A, tau, W, 0, conf);
  } else {
    __armas_dense_t T, V;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    __armas_make(&V, A->rows, lb, A->rows, &__armas_data(W)[__armas_size(&T)]);
    
    __blk_hess_gqvdg(A, tau, &T, &V, lb, conf);
  }
  return 0;
}

int __armas_hessreduce_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_hess_reduce(A->rows, A->cols, conf->lb);
}

/*
 * Multiply and replace C with product of C and Q where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     Hessenberg reduction as returned by __hess_reduce() where the lower trapezoidal
 *        part, on and below first subdiagonal, holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors. A column vector.
 *
 *  W     Workspace matrix,  required size is returned by WorksizeMultHess().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block sized. If it is zero
 *        unblocked invocation is assumed.
 *
 *        flags        result
 *        -------------------------------------
 *        LEFT         C = Q*C     n(A) == m(C)
 *        RIGHT        C = C*Q     n(C) == m(A)
 *        TRANS|LEFT   C = Q.T*C   n(A) == m(C)
 *        TRANS|RIGHT  C = C*Q.T   n(C) == m(A)
 *
 */
int __armas_hessmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                      __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t Qh, Ch, tauh;
  if (!conf)
    conf = armas_conf_default();

  __armas_submatrix(&Qh, A, 1, 0, A->rows-1, A->cols-1);
  __armas_submatrix(&tauh, tau, 0, 0, __armas_size(tau)-1, 1);
  if (flags & ARMAS_RIGHT) {
    __armas_submatrix(&Ch, C, 0, 1, C->rows, C->cols-1);
  } else {
    __armas_submatrix(&Ch, C, 1, 0, C->rows-1, C->cols);
  }

  return __armas_qrmult(&Ch, &Qh, &tauh, W, flags, conf);
}

int __armas_hessmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  return __armas_qrmult_work(A, flags, conf);
}
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

 
