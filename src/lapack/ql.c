
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! QL factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qlfactor) 
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

#ifndef IFERROR
#define IFERROR(exp) do { \
  int _e = (exp); \
  if (_e) { printf("error at: %s:%d\n", __FILE__, __LINE__); } \
  } while (0);
#endif
  


/*
 *   QL factorization.
 *
 *    A = Q*L = ( Q1 Q2 )*( 0 )  = Q2*L  
 *                        ( L )
 *
 *  For M-by-N matrix A the orthogonal matrix Q is product of N elementary
 *  reflectors H(k) = I - tau*v*v.T
 *
 *   H(i)*A = ( H(i)  0 ) * ( A0 ) = ( H(i)*A0 )
 *            (   0   I ) * ( A1 )   (   A1    )
 *
 *   H(i)*A0 = ( I - tau * ( v ) * ( v.T 1 ) ) * ( A0 )
 *             (           ( 1 )             )   ( a1 )
 *
 *           = ( A0 ) - tau * ( v*v.T  v ) ( A0 )
 *             ( a1 )         ( v.T    1 ) ( a1 )
 *
 *           = ( A0 - tau* (v*v.T*A0 + v*a1) )
 *             ( a1 - tau* (v.T*A0 + a1)     )
 *
 *           = ( A0 - tau*v*w )  where w = v.T*A0 + a1 = A0.T*v + a1
 *             ( a1 - tau*w   )
 */

static inline
int __ws_qlfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*N : N;
}

/*
 * Unblocked factorization.
 */
static
int __unblk_qlfactor(armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a01, a10, a11, A22;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;

  EMPTY(ATL);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PBOTTOM);
                 
  int k = 0;
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
    k++;
    __compute_householder(&a11, &a01, &t1, conf);

    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a10), 1);

    __apply_householder2x1(&t1, &a01, &a10, &A00, &w12, ARMAS_LEFT, conf);
    if (k == 4) {
      //printf("unblk.A k=%d\n", k); armas_x_printf(stdout, "%9.2e", A);
    }
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
int __blk_qlfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *T,
                   armas_x_dense_t *W, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A01, A10, A11, A22, AT;
  armas_x_dense_t tT, tB, t0, t1, t2, w12, Wrk;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PBOTTOM);
                 
  while (ATL.rows-lb > 0 && ATL.cols-lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           &A10,  &A11,  __nil,
                           __nil, __nil, &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    // current panel ( A01 )
    //               ( A11 )
    armas_x_submatrix(&w12, W, 0, 0, A11.cols, 1);
    __merge2x1(&AT, &A01, &A11);
    __unblk_qlfactor(&AT, &t1, &w12, conf);

    // build reflector T
    armas_x_mscale(T, 0.0, ARMAS_ANY);
    __unblk_ql_reflector(T, &AT, &t1, conf);

    // update with (I - Y*T*Y.T).T
    armas_x_submatrix(&Wrk, W, 0, 0, A10.cols, A10.rows);
    __update_ql_left(&A10, &A00, &A11, &A01, T, &Wrk, TRUE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }

  // last block with unblocked
  if (ATL.rows > 0 && ATL.cols > 0) {
    armas_x_submatrix(&w12, W, 0, 0, ATL.cols, 1);
    __unblk_qlfactor(&ATL, &t0, &w12, conf);
  }

  return 0;
}


/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
 *
 * where  C = ( C2 )   Y = ( Y2 )
 *            ( C1 )       ( Y1 )
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is P*nb, T is nb*nb,  W is K*nb
 */
int __update_ql_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                     armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                     int transpose, armas_conf_t *conf)
{
  // W = C1.T
  IFERROR(armas_x_scale_plus(W, C1, 0.0, 1.0, ARMAS_TRANSB, conf));
  // W = C1.T*Y1 = W*Y1
  IFERROR(armas_x_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf));
  // W = W + C2.T*Y2
  IFERROR(armas_x_mult(W, C2, Y2, 1.0, 1.0, ARMAS_TRANSA, conf));
  // here: W = C.T*Y

  int bits = ARMAS_LOWER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  IFERROR(armas_x_mult_trm(W, T, 1.0, bits, conf));
  // here: W == C.T*Y*T or C.T*Y*T.T

  // C2 = C2 - Y2*W.T
  IFERROR(armas_x_mult(C2, Y2, W, -1.0, 1.0, ARMAS_TRANSB, conf));
  // W = Y1*W.T ==> W.T = W*Y1.T
  IFERROR(armas_x_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf));
  // C1 = C1 - W.T
  IFERROR(armas_x_scale_plus(C1, W, 1.0, -1.0, ARMAS_TRANSB, conf));
  // here: C = (I - Y*T*Y.T)*C or C = (I - Y*T.Y.T).T*C
  return 0;
}

/*
 * compute:
 *      C*Q.T = C*(I -Y*T*Y.T).T ==  C - C*Y*T.T*Y.T
 * or
 *      C*Q   = (I -Y*T*Y.T)*C   ==  C - C*Y*T*Y.T
 *
 * where  C = ( C2 C1 )   Y = ( Y2 )
 *                            ( Y1 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb triuu, Y2 is P*nb, T is nb*nb, W = K*nb
*/
int __update_ql_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                      armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                      int transpose, armas_conf_t *conf)
{
  // W = C1
  IFERROR(armas_x_scale_plus(W, C1, 0.0, 1.0, ARMAS_NONE, conf));
  // W = C1*Y1 = W*Y1
  IFERROR(armas_x_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf));
  // W = W + C2*Y2
  IFERROR(armas_x_mult(W, C2, Y2, 1.0, 1.0, ARMAS_NONE, conf));
  // here: W = C*Y

  int bits = ARMAS_LOWER|ARMAS_RIGHT;
  if (transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  IFERROR(armas_x_mult_trm(W, T, 1.0, bits, conf));
  // here: W == C*Y*T or C*Y*T.T

  // C2 = C2 - W*Y2.T
  IFERROR(armas_x_mult(C2, W, Y2, -1.0, 1.0, ARMAS_TRANSB, conf));
  // C1 = C1 - W*Y1*T
  //  W = W*Y1.T
  IFERROR(armas_x_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf));
  // C1 = C1 - W
  IFERROR(armas_x_scale_plus(C1, W, 1.0, -1.0, ARMAS_NONE, conf));
  // here: C = C*(I - Y*T*Y.T)*C or C = C*(I - Y*T.Y.T).T
  return 0;
}

/*
 * Build block reflector T from HH reflector stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  0 |   z = -tau*T*Y.T*v
 *     | z  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
int __unblk_ql_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
                         armas_conf_t *conf)
{
  double tauval;
  armas_x_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
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
                           &A00,  &a01,  &A02,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
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
      // t21 := -tauval*(a12.T + &A02.T*a01)
      armas_x_axpby(&t21, &a12, 1.0, 0.0, conf);
      armas_x_mvmult(&t21, &A02, &a01, -tauval, -tauval, ARMAS_TRANSA, conf);
      // t21 := T22*t21
      armas_x_mvmult_trm(&t21, &T22, 1.0, ARMAS_LOWER, conf);
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
 */
int armas_x_qlreflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
                        armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (T->cols < A->cols || T->rows < A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  __unblk_ql_reflector(T, A, tau, conf);
  return 0;
}

/**
 * \brief Compute QL factorization of a M-by-N matrix A
 *
 * \param[in,out] A    
 *    On entry, the M-by-N matrix A, M >= N. On exit, lower triangular matrix L
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * \param[out] tau  
 *   On exit, the scalar factors of the elemenentary reflectors.
 *
 * \param[out] W    
 *   Workspace, N-by-nb matrix used for work space in blocked invocations. 
 *
 * \param[in,out] conf 
 *    The blocking configuration. If nil then default blocking configuration
 *    is used. Member conf.LB defines blocking size of blocked algorithms.
 *    If it is zero then unblocked algorithm is used.
 *
 * \retval  0 Success
 * \retval -1 Error, conf.error holds error code
 *
 * #### Additional information
 *
 *  Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *    \f$ Q = H_{k-1}...H_1 H_0, where K = min(M,N) \f$
 *
 *  Elementary reflector H(k) is stored on column k of A above the diagonal with
 *  implicit unit value on diagonal entry. The vector TAU holds scalar factors
 *  of the elementary reflectors.
 *
 *  Contents of matrix A after factorization is as follow:
 *
 *      ( v0 v1 v2 v3 )   for M=6, N=4
 *      ( v0 v1 v2 v3 )   l is element of L
 *      ( l  v1 v2 v3 )   vk is element of H(k)
 *      ( l  l  v2 v3 )
 *      ( l  l  l  v3 )
 *      ( l  l  l  l  )
 *
 *  qlfactor() is compatible with lapack.DGEQLF
 */
int armas_x_qlfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                     armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  // must have: M >= N
  if (A->rows < A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  lb = conf->lb;
  wsmin = __ws_qlfactor(A->rows, A->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_qlfactor(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_qlfactor);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    __unblk_qlfactor(A, tau, W, conf);
  } else {
    armas_x_dense_t T, Wrk;
    // block reflector at start of workspace
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    armas_x_make(&Wrk, A->cols-lb, lb, A->cols-lb, &armas_x_data(W)[armas_x_size(&T)]);

    __blk_qlfactor(A, tau, &T, &Wrk, lb, conf);
  }
  return 0;
}

/**
 * \brief Calculate work space for QL factorization
 *
 * Calculate required workspace to decompose matrix A with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int armas_x_qlfactor_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qlfactor(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

