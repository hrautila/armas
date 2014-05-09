
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_qrfactor) 
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
int __ws_qrfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*N : N;
}

/*
 * Unblocked factorization.
 */
static
int __unblk_qrfactor(__armas_dense_t *A, __armas_dense_t *tau,
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
    __compute_householder(&a11, &a21, &t1, conf);

    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a12), 1);

    __apply_householder2x1(&t1, &a21, &a12, &A22, &w12, ARMAS_LEFT, conf);
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
int __blk_qrfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *Twork,
                   __armas_dense_t *W, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AL;
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
    // decompose current panel AL = ( A11 )
    //                              ( A21 )
    __armas_submatrix(&w1, W, 0, 0, A11.rows, 1);
    __merge2x1(&AL, &A11, &A21);
    __unblk_qrfactor(&AL, &t1, &w1, conf);

    // build block reflector
    __armas_mscale(Twork, 0.0, ARMAS_NULL);
    __unblk_qr_reflector(Twork, &AL, &t1, conf);

    // update ( A12 A22 ).T
    __armas_submatrix(&Wrk, W, 0, 0, A12.cols, A12.rows);
    __update_qr_left(&A12, &A22, &A11, &A21, Twork, &Wrk, TRUE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }

  // last block with unblocked
  if (ABR.rows > 0 && ABR.cols > 0) {
    __armas_submatrix(&w1, W, 0, 0, ABR.cols, 1);
    __unblk_qrfactor(&ABR, &t2, &w1, conf);
  }

  return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
 *
 * where  C = ( C1 )   Y = ( Y1 )
 *            ( C2 )       ( Y2 )
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb,  W is K*nb
 */
int __update_qr_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
                     __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *W,
                     int transpose, armas_conf_t *conf)
{
  int err;
  // W = C1.T
  ONERROR(__armas_scale_plus(W, C1, 0.0, 1.0, ARMAS_TRANSB, conf));
  // W = C1.T*Y1 = W*Y1
  ONERROR(__armas_mult_trm(W, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf));
  // W = W + C2.T*Y2
  ONERROR(__armas_mult(W, C2, Y2, 1.0, 1.0, ARMAS_TRANSA, conf));
  // here: W = C.T*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  ONERROR(__armas_mult_trm(W, T, 1.0, bits, conf));
  // here: W == C.T*Y*T or C.T*Y*T.T

  // C2 = C2 - Y2*W.T
  ONERROR(__armas_mult(C2, Y2, W, -1.0, 1.0, ARMAS_TRANSB, conf));
  // W = Y1*W.T ==> W.T = W*Y1.T
  ONERROR(__armas_mult_trm(W, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf));
  // C1 = C1 - W.T
  ONERROR(__armas_scale_plus(C1, W, 1.0, -1.0, ARMAS_TRANSB, conf));
  // here: C = (I - Y*T*Y.T)*C or C = (I - Y*T.Y.T).T*C
  return 0;
}


/*
 * compute:
 *      C*Q.T = C*(I -Y*T*Y.T).T ==  C - C*Y*T.T*Y.T
 * or
 *      C*Q   = (I -Y*T*Y.T)*C   ==  C - C*Y*T*Y.T
 *
 * where  C = ( C1 C2 )   Y = ( Y1 )
 *                            ( Y2 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb, W = K*nb
*/
int __update_qr_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
                      __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *W,
                      int transpose, armas_conf_t *conf)
{
  int err;
  // W = C1
  ONERROR(__armas_scale_plus(W, C1, 0.0, 1.0, ARMAS_NONE, conf));
  // W = C1*Y1 = W*Y1
  ONERROR(__armas_mult_trm(W, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf));
  // W = W + C2*Y2
  ONERROR(__armas_mult(W, C2, Y2, 1.0, 1.0, ARMAS_NONE, conf));
  // here: W = C*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  ONERROR(__armas_mult_trm(W, T, 1.0, bits, conf));
  // here: W == C*Y*T or C*Y*T.T

  // C2 = C2 - W*Y2.T
  ONERROR(__armas_mult(C2, W, Y2, -1.0, 1.0, ARMAS_TRANSB, conf));
  // C1 = C1 - W*Y1*T
  //  W = W*Y1.T
  ONERROR(__armas_mult_trm(W, Y1, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf));
  // C1 = C1 - W
  ONERROR(__armas_scale_plus(C1, W, 1.0, -1.0, ARMAS_NONE, conf));
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
int __unblk_qr_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
                         armas_conf_t *conf)
{
  double tauval;
  __armas_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
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
                           &A00,  __nil, __nil,
                           &a10,  &a11,  __nil,
                           &A20,  &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&ATL,
                           &T00,  &t01,  &T02,
                           __nil, &t11,  &t12,
                           __nil, __nil,  &T22,  /**/  T, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    tauval = __armas_get(&t1, 0, 0);
    if (tauval != 0.0) {
      __armas_set(&t11, 0, 0, tauval);
      // t01 := -tauval*(a10.T + &A20.T*a21)
      __armas_axpby(&t01, &a10, 1.0, 0.0, conf);
      __armas_mvmult(&t01, &A20, &a21, -tauval, -tauval, ARMAS_TRANSA, conf);
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

int __armas_qrreflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
                        armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (T->cols < A->cols || T->rows < A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  __unblk_qr_reflector(T, A, tau, conf);
  return 0;
}

/**
 * @brief Compute QR factorization of a M-by-N matrix A = Q * R.
 *
 * Arguments:
 *  A   On entry, the M-by-N matrix A. On exit, the elements on and above
 *      the diagonal contain the min(M,N)-by-N upper trapezoidal matrix R.
 *      The elements below the diagonal with the column vector 'tau', represent
 *      the ortogonal matrix Q as product of elementary reflectors.
 *
 * tau  On exit, the scalar factors of the elementary reflectors.
 *
 * W    Workspace, N-by-nb matrix used for work space in blocked invocations. 
 *
 * conf The blocking configuration. If nil then default blocking configuration
 *      is used. Member conf.LB defines blocking size of blocked algorithms.
 *      If it is zero then unblocked algorithm is used.
 *
 * @returns:
 *      0 if succesfull, -1 otherwise
 *
 * Additional information
 *
 *  Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *    Q = H(1)H(2),...,H(K), where K = min(M,N)
 *
 *  Elementary reflector H(k) is stored on column k of A below the diagonal with
 *  implicit unit value on diagonal entry. The vector TAU holds scalar factors
 *  of the elementary reflectors.
 *
 *  Contents of matrix A after factorization is as follow:
 *
 *    ( r  r  r  r  )   for M=6, N=4
 *    ( v1 r  r  r  )
 *    ( v1 v2 r  r  )
 *    ( v1 v2 v3 r  )
 *    ( v1 v2 v3 v4 )
 *    ( v1 v2 v3 v4 )
 *
 *  where r is element of R, vk is element of H(k).
 *
 *  Compatible with lapack.xGEQRF
 */
int __armas_qrfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                     armas_conf_t *conf)
{
  int wsmin, lb;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_qrfactor(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  if (lb == 0 || A->cols <= lb) {
    __unblk_qrfactor(A, tau, W, conf);
  } else {
    __armas_dense_t T, Wrk;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    __armas_make(&Wrk, A->cols-lb, lb, A->cols-lb, &__armas_data(W)[__armas_size(&T)]);
    
    __blk_qrfactor(A, tau, &T, &Wrk, lb, conf);
  }
}

/*
 * Calculate required workspace to decompose matrix A with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int __armas_qrfactor_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qrfactor(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
