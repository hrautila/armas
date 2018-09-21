
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Hessenberg reduction
#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_hessreduce)  && defined(armas_x_hessmult)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) && defined(__update_qr_left) && defined(armas_x_qrmult)
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

#ifndef ARMAS_NIL
#define ARMAS_NIL (armas_x_dense_t *)0
#endif

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
int __unblk_hess_gqvdg(armas_x_dense_t *A, armas_x_dense_t *tau,
                       armas_x_dense_t *W, int row, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  armas_x_dense_t AL, AR, A0, a1, A2;
  armas_x_dense_t tT, tB, t0, t1, t2, w12, v1;
  DTYPE tauval, beta;

  EMPTY(A00); EMPTY(a11);
  EMPTY(AL); EMPTY(A0); EMPTY(a1);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, row, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&AL, &AR,      /**/  A, 0, ARMAS_PLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  armas_x_submatrix(&v1, W, 0, 0, A->rows, 1);

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
    tauval = armas_x_get(&t1, 0, 0);
    beta   = armas_x_get(&a21, 0, 0);
    armas_x_set(&a21, 0, 0, 1.0);

    // v1 := A2*a21
    armas_x_mvmult(__ZERO, &v1, __ONE, &A2, &a21, ARMAS_NONE, conf);
    
    // A2 := A2 - tau*v1*a21  (A2 = A2*H(k))
    armas_x_mvupdate(&A2, -tauval, &v1, &a21, conf);
    
    armas_x_submatrix(&w12, W, 0, 0, A22.cols, 1);
    // w12 := a21.T*A22 = A22.T*a21
    armas_x_mvmult(__ZERO, &w12, __ONE, &A22, &a21, ARMAS_TRANS, conf);
    // A22 := A22 - tau*a21*w12  (A22 = H(k)*A22)
    armas_x_mvupdate(&A22, -tauval, &a21, &w12, conf);

    // restore a21[0]
    armas_x_set(&a21, 0, 0, beta);
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
int __update_vec_left_wy(armas_x_dense_t *v, armas_x_dense_t *Y1, armas_x_dense_t *Y2,
                         armas_x_dense_t *T, armas_x_dense_t *W, int bits,
                         armas_conf_t *conf)
{
  armas_x_dense_t v1, v2, w0;
  
  armas_x_submatrix(&v1, v, 1, 0, Y1->cols, 1);
  armas_x_submatrix(&v2, v, 1+Y1->cols, 0, Y2->rows, 1);
  armas_x_submatrix(&w0, W, 0, 0, Y1->rows, 1);
  
  // w0 = Y1.T*v1 + Y2.T*v2
  armas_x_axpby(__ZERO, &w0, __ONE, &v1, conf);
  armas_x_mvmult_trm(&w0, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANS, conf);
  armas_x_mvmult(__ONE, &w0, __ONE, Y2, &v2, ARMAS_TRANS, conf);
  
  // w0 = opt(T)*w0
  armas_x_mvmult_trm(&w0, __ONE, T, bits|ARMAS_UPPER, conf);
  
  // v2 = v2 - Y2*w0
  armas_x_mvmult(__ONE, &v2, -__ONE, Y2, &w0, ARMAS_NONE, conf);
  
  // v1 = v1 - Y1*w0
  armas_x_mvmult_trm(&w0, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT, conf);
  armas_x_axpy(&v1, -__ONE, &w0, conf);
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
int __unblk_build_hess_gqvdg(armas_x_dense_t *A, armas_x_dense_t *T,
                             armas_x_dense_t *V, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
  armas_x_dense_t AL, AR, A0, a1, A2;
  armas_x_dense_t VL, VR, V0, v1, V2, Y0;
  armas_x_dense_t TTL, TBR, T00, t01, t11, T22;
  DTYPE beta, tauval;

  EMPTY(a11); EMPTY(A00);
  EMPTY(VR); EMPTY(VL); EMPTY(AL); 
  EMPTY(A0);
  
  beta = __ZERO;

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
      armas_x_axpby(__ZERO, &t01, __ONE, &a10, conf);
      armas_x_mvmult_trm(&t01, __ONE, &T00, ARMAS_UPPER, conf);
      
      // a1 = a1 - V0*T00*a10
      armas_x_mvmult(__ONE, &a1, -__ONE, &V0, &t01, ARMAS_NONE, conf);
      
      // update a1 = (I - Y*T*Y.T).T*a1
      armas_x_submatrix(&Y0, A, 1, 0, A00.cols, A00.cols);
      __update_vec_left_wy(&a1, &Y0, &A20, &T00, &t01, ARMAS_TRANS, conf);
      // restore last entry of a10, 
      armas_x_set(&a10, 0, -1, beta);
    }
    // compute householder
    __compute_householder_vec(&a21, &t11, conf);
    beta = armas_x_get(&a21, 0, 0);
    armas_x_set(&a21, 0, 0, 1.0);
    
    // v1 = A2*a21
    armas_x_mvmult(__ZERO, &v1, __ONE, &A2, &a21, ARMAS_NONE, conf);
    // update T
    tauval = armas_x_get(&t11, 0, 0);
    if (tauval != 0) {
      // t01 := -tauval*A20.T*a21
      armas_x_mvmult(__ZERO, &t01, -tauval, &A20, &a21, ARMAS_TRANS, conf);
      // t01 := T00*t01
      armas_x_mvmult_trm(&t01, __ONE, &T00, ARMAS_UPPER, conf);
    }

    // ---------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&TTL,  __nil,
                        __nil, &TBR, /**/  &T00, &t11, &T22,   T, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&AL,   &AR,  /**/  &A0,  &a1,   A, ARMAS_PRIGHT);
    __continue_1x3to1x2(&VL,   &VR,  /**/  &V0,  &v1,   V, ARMAS_PRIGHT);
  }

  armas_x_set(A, V->cols, V->cols-1, beta);
  return 0;
}

// compute: (I - Y*T*Y.T).T*C
static
int __update_hess_left_wy(armas_x_dense_t *C, armas_x_dense_t *Y1, armas_x_dense_t *Y2,
                          armas_x_dense_t *T, armas_x_dense_t *V, armas_conf_t *conf)
{
  armas_x_dense_t C1, C2;
  if (armas_x_size(C) == 0) {
    return 0;
  }
  armas_x_submatrix(&C1, C, 1, 0, Y1->rows, C->cols);
  armas_x_submatrix(&C2, C, 1+Y1->cols, 0, Y2->rows, C->cols);
  
  __update_qr_left(&C1, &C2, Y1, Y2, T, V, TRUE, conf);
  return 0;
}

// compute: C*(I - Y*T*Y.T)
static
int __update_hess_right_wy(armas_x_dense_t *C, armas_x_dense_t *Y1, armas_x_dense_t *Y2,
                          armas_x_dense_t *T, armas_x_dense_t *V, armas_conf_t *conf)
{
  armas_x_dense_t C1, C2;
  if (armas_x_size(C) == 0) {
    return 0;
  }
  armas_x_submatrix(&C1, C, 0, 1, C->rows, Y1->cols);
  armas_x_submatrix(&C2, C, 0, 1+Y1->cols, C->rows, Y2->rows);
  
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
int __blk_hess_gqvdg(armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *T, armas_x_dense_t *V, int lb,
                     armas_conf_t *conf)
{
  armas_x_dense_t ATL, ATR, ABR, A00, A11, A12, A21, A22, A2;
  armas_x_dense_t VT, VB, Y1, Y2, W0;
  armas_x_dense_t tT, tB, t0, t1, t2, td;
  DTYPE beta;

  EMPTY(A12); EMPTY(A00); EMPTY(A11); EMPTY(A22);
  EMPTY(ATR);

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  armas_x_diag(&td, T, 0);
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
    armas_x_axpby(__ZERO, &t1, __ONE, &td, conf);

    armas_x_submatrix(&Y1, &ABR, 1, 0, A11.cols, A11.cols);
    armas_x_submatrix(&Y2, &ABR, 1+A11.cols, 0, A21.rows-1, A11.cols);
    
    // [A01, A02] == ATR := ATR*(I - Y*T*Y.T);
    __update_hess_right_wy(&ATR, &Y1, &Y2, T, &VT, conf);

    __merge2x1(&A2, &A12, &A22);
    
    // A2 := A2 - VB*T*A21.T
    beta = armas_x_get(&A21, 0, -1);
    armas_x_set(&A21, 0, -1, 1.0);
    armas_x_mult_trm(&VB, __ONE, T, ARMAS_UPPER|ARMAS_RIGHT, conf);
    armas_x_mult(__ONE, &A2, -__ONE, &VB, &A21, ARMAS_TRANSB, conf);
    armas_x_set(&A21, 0, -1, beta);
    
    // A2 := (I - Y*T*Y.T).T * A2
    armas_x_submatrix(&W0, V, 0, 0, A2.cols, Y2.cols);
    __update_hess_left_wy(&A2, &Y1, &Y2, T, &W0, conf);

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }

  if (ABR.rows > 1) {
    __merge2x1(&A2, &ATR, &ABR);
    armas_x_make(&W0, A->rows, 1, A->rows, armas_x_data(V));
    __unblk_hess_gqvdg(&A2, &tB, &W0, ATR.rows, conf);
  }
  return 0;
}

/**
 * \brief Hessenberg reduction of general matrix
 *
 * Reduce general matrix A to upper Hessenberg form H by similiarity
 * transformation \f$ H = Q^T A Q \f$.
 *
 * \param[in,out] A
 *    On entry, the general matrix A. On exit, the elements on and
 *    above the first subdiagonal contain the reduced matrix H.
 *    The elements below the first subdiagonal with the vector tau
 *    represent the ortogonal matrix A as product of elementary reflectors.
 *
 * \param[out] tau  
 *    On exit, the scalar factors of the elementary reflectors.
 *
 * \param[in,out] conf
 *    The blocking configration. 
 * 
 * \retval 0 Succes
 * \retval -1 Failure, conf.error set to error code
 *
 * Compatible with lapack.DGEHRD.
 * \ingroup lapack
 */
int armas_x_hessreduce(armas_x_dense_t *A,
                       armas_x_dense_t *tau,
                       armas_x_dense_t *W,
                       armas_conf_t *conf)
{
  int err;
  armas_wbuf_t wb = ARMAS_WBNULL;

  if (!conf)
    conf = armas_conf_default();

  if (armas_x_hessreduce_w(A, tau, &wb, conf) < 0)
    return -1;

  if (!armas_walloc(&wb, wb.bytes)) {
    conf->error = ARMAS_EMEMORY;
    return -1;
  }

  err = armas_x_hessreduce_w(A, tau, &wb, conf);
  armas_wrelease(&wb);
  return err;
}

//! \brief Workspace size for Hessenberg reduction
//! \ingroup lapack
int armas_x_hessreduce_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_hess_reduce(A->rows, A->cols, conf->lb);
}

/**
 * \brief Hessenberg reduction of general matrix
 *
 * Reduce general matrix A to upper Hessenberg form H by similiarity
 * transformation \f$ H = Q^T A Q \f$.
 *
 * \param[in,out] A
 *    On entry, the general matrix A. On exit, the elements on and
 *    above the first subdiagonal contain the reduced matrix H.
 *    The elements below the first subdiagonal with the vector tau
 *    represent the ortogonal matrix A as product of elementary reflectors.
 *
 * \param[out] tau  
 *    On exit, the scalar factors of the elementary reflectors.
 *
 * \param wb
 *    Work buffer.
 *
 * \param[in,out] conf
 *    The blocking configration. 
 * 
 * \retval 0 Succes
 * \retval -1 Failure, conf.error set to error code
 *
 * Compatible with lapack.DGEHRD.
 * \ingroup lapack
 */
int armas_x_hessreduce_w(armas_x_dense_t *A,
                         armas_x_dense_t *tau,
                         armas_wbuf_t *wb,
                         armas_conf_t *conf)
{
  armas_x_dense_t T, V;
  size_t wsmin, wsz;
  int lb;
  DTYPE *buf;
  
  if (!conf)
    conf = armas_conf_default();

  if (!A) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }

  if (wb && wb->bytes == 0) {
    if (conf->lb > 0 && A->rows > conf->lb)
      wb->bytes = (A->rows + conf->lb) * conf->lb * sizeof(DTYPE);
    else
      wb->bytes = A->rows * sizeof(DTYPE);
    return 0;
  }
  
  if (A->rows != A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (! armas_x_isvector(tau) || armas_x_size(tau) != A->rows-1) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }
  
  lb = conf->lb;
  wsmin = A->rows * sizeof(DTYPE);
  if (! wb || (wsz = armas_wbytes(wb)) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }

  // adjust blocking factor for workspace
  if (lb > 0 && A->rows > lb) {
    wsz /= sizeof(DTYPE);
    if (wsz < (A->rows + lb)*lb) {
      // solve: lb^2 + m(A)*lb - wsz = 0
      lb  = ((int)(__SQRT((DTYPE)(A->rows*A->rows + 4*wsz))) - A->rows) / 2;
      lb &= ~0x3;
    }
  }
  
  wsz = armas_wpos(wb);
  buf = (DTYPE *)armas_wptr(wb);
  
  
  if (lb == 0 || A->cols <= lb) {
    armas_x_make(&V, A->rows, 1, A->rows, buf);
    __unblk_hess_gqvdg(A, tau, &V, 0, conf);
  } else {
    // block reflector; temporary space, [n(A), lb] matrix
    armas_x_make(&T, lb, lb, lb, buf);
    armas_x_make(&V, A->rows, lb, A->rows, &buf[armas_x_size(&T)]);
    
    __blk_hess_gqvdg(A, tau, &T, &V, lb, conf);
  }
  armas_wsetpos(wb, wsz);
  return 0;
}

/**
 * \brief Multiply with the orthogonal matrix Q of Hessenberg reduction
 *
 * Multiply and replace C with product of C and Q where Q is a real orthogonal matrix
 * defined as the product of K = n(A) elementary reflectors.
 *
 *    \f$ Q = H_1  H_2 . . . H_K \f$ 
 *
 * \param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then N-by-M matrix
 *     On exit C is overwritten by \f$ Q C \f$ or \f$ Q^T C \f$. If bit *ARMAS_RIGHT* is 
 *     set then C is overwritten by \f$ CQ \f$ or \f$ C Q^T \f$
 *
 * \param[in] A
 *      Hessenberg reduction as returned by hessreduce() where the lower trapezoidal
 *      part, on and below first subdiagonal, holds the elementary reflectors.
 *
 * \param[in] tau
 *     The scalar factors of the elementary reflectors. A column vector.
 *
 * \param[out] W
 *     Workspace matrix,  required size is returned by hessmult_work()
 *
 * \param[in] flags 
 *    Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT* and *ARMAS_TRANS*
 *       
 * \param[in,out] conf  
 *    Blocking configuration. Field conf.lb defines block size. If it is zero
 *    unblocked invocation is assumed.
 *
 * \cond
 *        flags        result
 *        -------------------------------------
 *        LEFT         C = Q*C     n(A) == m(C)
 *        RIGHT        C = C*Q     n(C) == m(A)
 *        TRANS|LEFT   C = Q.T*C   n(A) == m(C)
 *        TRANS|RIGHT  C = C*Q.T   n(C) == m(A)
 * \endcond
 *
 * \ingroup lapack
 */
int armas_x_hessmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                      armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  int err;
  armas_wbuf_t wb = ARMAS_WBNULL;

  if (!conf)
    conf = armas_conf_default();

  if (armas_x_hessmult_w(C, A, tau, flags, &wb, conf) < 0)
    return -1;

  if (!armas_walloc(&wb, wb.bytes)) {
    conf->error = ARMAS_EMEMORY;
    return -1;
  }
  err = armas_x_hessmult_w(C, A, tau, flags, &wb, conf);
  armas_wrelease(&wb);
  return err;
}

int armas_x_hessmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  return armas_x_qrmult_work(A, flags, conf);
}

int armas_x_hessmult_w(armas_x_dense_t *C,
                       const armas_x_dense_t *A,
                       const armas_x_dense_t *tau,
                       int flags,
                       armas_wbuf_t *wb,
                       armas_conf_t *conf)
{
  armas_x_dense_t Qh, Ch, tauh;
  if (!conf)
    conf = armas_conf_default();

  if (!C) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }
  
  armas_x_submatrix(&Qh, A, 1, 0, A->rows-1, A->cols-1);
  armas_x_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
  if (flags & ARMAS_RIGHT) {
    armas_x_submatrix(&Ch, C, 0, 1, C->rows, C->cols-1);
  } else {
    armas_x_submatrix(&Ch, C, 1, 0, C->rows-1, C->cols);
  }

  if (wb && wb->bytes == 0) {
    return armas_x_qrmult_w(&Ch, ARMAS_NIL, ARMAS_NIL, flags, wb, conf);
  }
  return armas_x_qrmult_w(&Ch, &Qh, &tauh, flags, wb, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

 
