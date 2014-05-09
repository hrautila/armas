
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_lqfactor) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) 
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
int __ws_lqfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*M : M;
}

/*
 * Unblocked factorization.
 */
static
int __unblk_lqfactor(__armas_dense_t *A, __armas_dense_t *tau,
                     __armas_dense_t *W, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  __armas_dense_t tT, tB, t0, t1, t2, w12;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    __compute_householder(&a11, &a12, &t1, conf);

    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a21), 1);

    __apply_householder2x1(&t1, &a12, &a21, &A22, &w12, ARMAS_RIGHT, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}


/*
 * Blocked factorization.
 */
static
int __blk_lqfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *Twork,
                   __armas_dense_t *W, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AR;
  __armas_dense_t tT, tB, t0, t1, t2, w1, Wrk, row;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    // decompose current panel AT = ( A11 A 12 )
    __armas_submatrix(&w1, W, 0, 0, A11.rows, 1);
    __merge1x2(&AR, &A11, &A12);
    __unblk_lqfactor(&AR, &t1, &w1, conf);

    // build block reflector
    __armas_mscale(Twork, 0.0, ARMAS_ANY);
    __unblk_lq_reflector(Twork, &AR, &t1, conf);

    // update ( A21 A22 )
    __armas_submatrix(&Wrk, W, 0, 0, A21.rows, A21.cols);
    __update_lq_right(&A21, &A22, &A11, &A12, Twork, &Wrk, TRUE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }

  // last block with unblocked
  if (ABR.rows > 0 && ABR.cols > 0) {
    __armas_submatrix(&w1, W, 0, 0, ABR.rows, 1);
    __unblk_lqfactor(&ABR, &t2, &w1, conf);
  }

  return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
 *
 * where  C = ( C1 )   Y = ( Y1 Y2 )
 *            ( C2 )       
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is nb*P, T is nb*nb,  W is K*nb
 */
int __update_lq_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
                     __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *W,
                     int transpose, armas_conf_t *conf)
{
  int err;
  // W = C1.T
  __armas_scale_plus(W, C1, 0.0, 1.0, ARMAS_TRANSB, conf);
  // W = C1.T*Y1.T = W*Y1.T
  __armas_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2.T*Y2.T
  __armas_mult(W, C2, Y2, 1.0, 1.0, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // here: W = C.T*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  __armas_mult_trm(W, T, 1.0, bits, conf);
  // here: W == C.T*Y*T or C.T*Y*T.T

  // C2 = C2 - Y2*W.T
  __armas_mult(C2, Y2, W, -1.0, 1.0, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // W = Y1*W.T ==> W.T = W*Y1
  __armas_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
  // C1 = C1 - W.T
  __armas_scale_plus(C1, W, 1.0, -1.0, ARMAS_TRANSB, conf);
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
int __update_lq_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
                      __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *W,
                      int transpose, armas_conf_t *conf)
{
  // W = C1
  __armas_scale_plus(W, C1, 0.0, 1.0, ARMAS_NONE, conf);
  // W = C1*Y1.T = W*Y1.T
  __armas_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2*Y2.T
  __armas_mult(W, C2, Y2, 1.0, 1.0, ARMAS_TRANSB, conf);
  // here: W = C*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  __armas_mult_trm(W, T, 1.0, bits, conf);
  // here: W == C*Y*T or C*Y*T.T

  // C2 = C2 - W*Y2
  __armas_mult(C2, W, Y2, -1.0, 1.0, ARMAS_NONE, conf);
  // C1 = C1 - W*Y1
  //  W = W*Y1.T
  __armas_mult_trm(W, Y1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
  // C1 = C1 - W
  __armas_scale_plus(C1, W, 1.0, -1.0, ARMAS_NONE, conf);
  // here: C = C*(I - Y*T*Y.T)*C or C = C*(I - Y*T.Y.T).T
  return 0;
}

/*
 * Build block reflector T from HH reflector stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  z |   z = -tau*T*Y.T*v
 *     | 0  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
int __unblk_lq_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
                         armas_conf_t *conf)
{
  double tauval;
  __armas_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
  __armas_dense_t TTL, TBR, T00, t01, T02, t11, t12, T22;
  __armas_dense_t tT, tB, t0, t1, t2, w1;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&TTL,  __nil,
                  __nil, &TBR,   /**/  T, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, 0, ARMAS_PTOP);
                 
  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  &A02,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&ATL,
                           &T00,  &t01,  &T02,
                           __nil, &t11,  &t12,
                           __nil, __nil, &T22,  /**/  T, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    tauval = __armas_get(&t1, 0, 0);
    if (tauval != 0.0) {
      __armas_set(&t11, 0, 0, tauval);
      // t01 := -tauval*(a01.T + &A02.T*a21)
      __armas_axpby(&t01, &a01, 1.0, 0.0, conf);
      __armas_mvmult(&t01, &A02, &a12, -tauval, -tauval, ARMAS_NONE, conf);
      // t01 := T00*t01
      __armas_mvmult_trm(&t01, &T00, 1.0, ARMAS_UPPER, conf);
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&TTL,  __nil,
                        __nil, &TBR, /**/  &T00, &t11, &T22,   T, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}

int __armas_lqreflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
                        armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (T->cols < A->cols || T->rows < A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  __unblk_lq_reflector(T, A, tau, conf);
  return 0;
}

/**
 * Compute LQ factorization of a M-by-N matrix A: A = L*Q 
 *
 * Arguments:
 *  A   On entry, the M-by-N matrix A. On exit, the elements on and below
 *      the diagonal contain the min(M,N)-by-N lower trapezoidal matrix L.
 *      The elements above the diagonal with the column vector TAU, represent
 *      the ortogonal matrix Q as product of elementary reflectors.
 *
 * tau  On exit, the scalar factors of the elemenentary reflectors.
 *
 * W    Workspace, M-by-nb matrix used for work space in blocked invocations. 
 *
 * conf The blocking configuration. If nil then default blocking configuration
 *      is used. Member conf.lb defines blocking size of blocked algorithms.
 *      If it is zero then unblocked algorithm is used.
 *
 * Returns:
 *      Error indicator.
 *
 * Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *   Q = H(K)H(K-1),...,H(1), where K = min(M,N)
 *
 * Elementary reflector H(k) is stored on row k of A right of the diagonal with
 * implicit unit value on diagonal entry. The vector TAU holds scalar factors of
 * the elementary reflectors.
 *
 * Contents of matrix A after factorization is as follow:
 *
 *    ( l  v1 v1 v1 v1 v1 )  for M=4, N=6
 *    ( l  l  v2 v2 v2 v2 )
 *    ( l  l  l  v3 v3 v3 )
 *    ( l  l  l  l  v4 v4 )
 *
 * where l is element of L, vk is element of H(k).
 *
 * DecomposeLQ is compatible with lapack.DGELQF
 */
int __armas_lqfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                     armas_conf_t *conf)
{
  int wsmin, lb;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_lqfactor(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  if (lb == 0 || A->cols <= lb) {
    __unblk_lqfactor(A, tau, W, conf);
  } else {
    __armas_dense_t T, Wrk;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    __armas_make(&Wrk, A->rows-lb, lb, A->rows-lb, &__armas_data(W)[__armas_size(&T)]);

    __blk_lqfactor(A, tau, &T, &Wrk, lb, conf);
  }
}

/*
 * Calculate required workspace to decompose matrix A with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int __armas_lqfactor_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_lqfactor(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
