
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! RQ factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_rqfactor) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) 
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

#define IFERROR(exp) do { \
  int _e = (exp); \
  if (_e) { printf("error at: %s:%d\n", __FILE__, __LINE__); } \
  } while (0);

/*
 * RQ factorization of matrix A.
 *
 *  $$ A = R*Q == \left(0 R right\) \left( Q_1 \over Q_2 \right)  = R*Q_2 $$
 *
 *  where $$ A \in R^{m x n}, R \in R^{m x m}, Q_1 \in R^{m x n} and Q_2 \in R^{{n-m} x n}$$
 *
 * $Q_1$
 */
static inline
int __ws_rqfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*M : M;
}

/*
 * Unblocked factorization.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**T
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(n-k+i+1:n) = 0 and v(n-k+i) = 1; v(1:n-k+i-1) is stored on exit in
 *  A(m-k+i,1:n-k+i-1), and tau in TAU(i).
 *
 *  m >= n
 *   ( v1 v1 v1 r  r  r  r )
 *   ( v2 v2 v2 v2 r  r  r )
 *   ( v3 v3 v3 v3 v3 r  r )
 *   ( v4 v4 v4 v4 v4 v4 r )
 */
static
int __unblk_rqfactor(armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a01, a10, A22;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;

  EMPTY(ATL); 

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PBOTTOM);
                 
  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           &a10,  &a11,  __nil,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __compute_householder(&a11, &a10, &t1, conf);

    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a01), 1);

    __apply_householder2x1(&t1, &a10, &a01, &A00, &w12, ARMAS_RIGHT, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}


/*
 * Blocked factorization.
 */
static
int __blk_rqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *Twork,
                   armas_x_dense_t *W, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A01, A10, A11, A22, AL;
  armas_x_dense_t tT, tB, t0, t1, t2, w1, Wrk;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PBOTTOM);
                 
  while (ATL.rows - lb > 0 && ATL.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           &A10,  &A11,  __nil,
                           __nil, __nil, &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, A11.cols, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    // decompose current panel AL = ( A10 A11 )
    armas_x_submatrix(&w1, W, 0, 0, A11.rows, 1);
    __merge1x2(&AL, &A10, &A11);
    __unblk_rqfactor(&AL, &t1, &w1, conf);

    // build block reflector
    armas_x_mscale(Twork, 0.0, ARMAS_ANY);
    __unblk_rq_reflector(Twork, &AL, &t1, conf);

    // update ( A00 A01 )
    armas_x_submatrix(&Wrk, W, 0, 0, A01.rows, A01.cols);
    __update_rq_right(&A01, &A00, &A11, &A10, Twork, &Wrk, FALSE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }

  // last block with unblocked
  if (ATL.rows > 0 && ATL.cols > 0) {
    armas_x_submatrix(&w1, W, 0, 0, ATL.rows, 1);
    __unblk_rqfactor(&ATL, &tT, &w1, conf);
  }

  return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
 *
 * where  C = ( C1 )   Y = ( Y2 Y1 )
 *            ( C2 )       
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is nb*P, T is nb*nb,  W is K*nb
 */
int __update_rq_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                     armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                     int transpose, armas_conf_t *conf)
{
  // W = C1.T
  armas_x_scale_plus(__ZERO, W, __ONE, C1, ARMAS_TRANSB, conf);
  // W = C1.T*Y1.T = W*Y1.T
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2.T*Y2.T
  armas_x_mult(__ONE, W, __ONE, C2, Y2, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // here: W = C.T*Y

  int bits = ARMAS_LOWER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  armas_x_mult_trm(W, __ONE, T, bits, conf);
  // here: W == C.T*Y*T or C.T*Y*T.T

  // C2 = C2 - Y2*W.T
  armas_x_mult(__ONE, C2, -__ONE, Y2, W, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // W = Y1*W.T ==> W.T = W*Y1
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
  // C1 = C1 - W.T
  armas_x_scale_plus(__ONE, C1, -__ONE, W, ARMAS_TRANSB, conf);
  // here: C = (I - Y*T*Y.T)*C or C = (I - Y*T.Y.T).T*C
  return 0;
}


/*
 * compute:
 *      C*Q.T = C*(I -Y*T*Y.T).T ==  C - C*Y*T.T*Y.T
 * or
 *      C*Q   = (I -Y*T*Y.T)*C   ==  C - C*Y*T*Y.T
 *
 * where  C = ( C1 C2 )   Y = ( Y1 Y2 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb trilu, Y2 is nb*P, T is nb*nb, W = K*nb
*/
int __update_rq_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                      armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                      int transpose, armas_conf_t *conf)
{
  // W = C1
  armas_x_scale_plus(__ZERO, W, __ONE, C1, ARMAS_NONE, conf);
  // W = C1*Y1 = W*Y1
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2*Y2.T
  armas_x_mult(__ONE, W, __ONE, C2, Y2, ARMAS_TRANSB, conf);
  // here: W = C*Y

  int bits = ARMAS_LOWER|ARMAS_RIGHT;
  if (transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  armas_x_mult_trm(W, __ONE, T, bits, conf);
  // here: W == C*Y*T or C*Y*T.T

  // C2 = C2 - W*Y2
  armas_x_mult(__ONE, C2, -__ONE, W, Y2, ARMAS_NONE, conf);
  // C1 = C1 - W*Y1
  //  W = W*Y1.T
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
  // C1 = C1 - W
  armas_x_scale_plus(__ONE, C1, -__ONE, W, ARMAS_NONE, conf);
  // here: C = C*(I - Y*T*Y.T)*C or C = C*(I - Y*T.Y.T).T
  return 0;
}

/*
 * Build block reflector T from elementary reflectors stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  0 |   z = -tau*T*Y.T*v
 *     | z  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 *
 */
int __unblk_rq_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
                         armas_conf_t *conf)
{
  double tauval;
  armas_x_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
  armas_x_dense_t TTL, TBR, T00, t11, t21, T22;
  armas_x_dense_t tT, tB, t0, t1, t2;

  EMPTY(ATL); EMPTY(A00);
  EMPTY(TTL); EMPTY(T00); EMPTY(t11);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x2(&TTL,  __nil,
                  __nil, &TBR,   /**/  T, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PBOTTOM);
                 
  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &a10,  &a11,  __nil,
                           &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x2to3x3(&TTL,
                           &T00,  __nil, __nil,
                           __nil, &t11,  __nil,
                           __nil, &t21,  &T22,  /**/  T, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    tauval = armas_x_get(&t1, 0, 0);
    if (tauval != 0.0) {
      armas_x_set(&t11, 0, 0, tauval);
      // t21 := -tauval*(a21 + &A20*a10)
      armas_x_axpby(__ZERO, &t21, __ONE, &a21, conf);
      armas_x_mvmult(-tauval, &t21, -tauval, &A20, &a10, ARMAS_NONE, conf);
      // t01 := T22*t21
      armas_x_mvmult_trm(&t21, __ONE, &T22, ARMAS_LOWER, conf);
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x3to2x2(&TTL,  __nil,
                        __nil, &TBR, /**/  &T00, &t11, &T22,   T, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}

/*
 * Build block reflector from RQ factorized matrix.
 *
 * Elementary reflector stored in matrix A rowwise as descriped below. Result
 * block reflector matrix is lower triangular with tau-vector on diagonal.
 *
 *    ( v1 v1 v1 1  .  . )  ( t1 )    ( t1 .  .  )
 *    ( v2 v2 v2 v2 1  . )  ( t2 )    ( t  t2 .  )
 *    ( v3 v3 v3 v3 v3 1 )  ( t3 )    ( t  t  t3 )
 */
int armas_x_rqreflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
                        armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (T->cols < A->rows || T->rows < A->rows) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  __unblk_rq_reflector(T, A, tau, conf);
  return 0;
}

/**
 * \brief Compute RQ factorization of a M-by-N matrix A
 *
 * \param[in,out] A
 *    On entry, the M-by-N matrix A, M <= N. On exit, upper triangular matrix R
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * \param[out] tau  
 *    On exit, the scalar factors of the elemenentary reflectors.
 *
 * \param[out]  W    
 *    Workspace, M-by-nb matrix used for work space in blocked invocations. 
 *
 * \param[in,out] conf 
 *    The blocking configuration. If nil then default blocking configuration
 *    is used. Member conf.lb defines blocking size of blocked algorithms.
 *    If it is zero then unblocked algorithm is used.
 *
 * \retval  0 Success
 * \retval -1 Error.
 *
 * #### Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *   \f$ Q = H_0 H-1,...,H_{K-1} \f$ , where \f$ K = \min M N \f$
 *
 * Elementary reflector H(k) is stored on first N-M+k elements of row k of A.
 * with implicit unit value on element N-M+k entry. The vector TAU holds scalar
 * factors of the elementary reflectors.
 *
 * Contents of matrix A after factorization is as follow:
 *
 *      ( v0 v0 r  r  r  r )  M=4, N=6
 *      ( v1 v1 v1 r  r  r )  r  is element of R
 *      ( v2 v2 v2 v2 r  r )  vk is element of H(k)
 *      ( v3 v3 v3 v3 v3 r )  
 *
 * rqfactor() is compatible with lapack.DGERQF
 */
int armas_x_rqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                     armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  // must have: M <= N
  if (A->rows > A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  lb = conf->lb;
  wsmin = __ws_rqfactor(A->rows, A->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_rqfactor(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_rqfactor);
    lb = min(lb, conf->lb);
  }
  if (lb == 0 || A->rows <= lb) {
    __unblk_rqfactor(A, tau, W, conf);
  } else {
    armas_x_dense_t T, Wrk;
    // block reflector at start of workspace
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    armas_x_make(&Wrk, A->rows-lb, lb, A->rows-lb, &armas_x_data(W)[armas_x_size(&T)]);

    __blk_rqfactor(A, tau, &T, &Wrk, lb, conf);
  }
  return 0;
}

/**
 * \brief Calculate worksize for RQ factorization
 */
int armas_x_rqfactor_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_rqfactor(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

