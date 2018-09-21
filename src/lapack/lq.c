
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! LQ factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_lqfactor) && defined(armas_x_lqfactor_w) 
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

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

/*
 * Unblocked factorization.
 */
static
int __unblk_lqfactor(armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *W, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;

  EMPTY(A00);

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

    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a21), 1);

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
int __blk_lqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *Twork,
                   armas_x_dense_t *W, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A11, A12, A21, A22, AR;
  armas_x_dense_t tT, tB, t0, t1, t2, w1, Wrk;

  EMPTY(A00);

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
    armas_x_submatrix(&w1, W, 0, 0, A11.rows, 1);
    __merge1x2(&AR, &A11, &A12);
    __unblk_lqfactor(&AR, &t1, &w1, conf);

    // build block reflector
    armas_x_mscale(Twork, 0.0, ARMAS_ANY);
    __unblk_lq_reflector(Twork, &AR, &t1, conf);

    // update ( A21 A22 )
    armas_x_submatrix(&Wrk, W, 0, 0, A21.rows, A21.cols);
    __update_lq_right(&A21, &A22, &A11, &A12, Twork, &Wrk, TRUE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }

  // last block with unblocked
  if (ABR.rows > 0 && ABR.cols > 0) {
    armas_x_submatrix(&w1, W, 0, 0, ABR.rows, 1);
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
int __update_lq_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                     armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                     int transpose, armas_conf_t *conf)
{
  // W = C1.T
  armas_x_scale_plus(__ZERO, W, __ONE, C1, ARMAS_TRANSB, conf);
  // W = C1.T*Y1.T = W*Y1.T
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2.T*Y2.T
  armas_x_mult(__ONE, W, __ONE, C2, Y2, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // here: W = C.T*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  armas_x_mult_trm(W, __ONE, T, bits, conf);
  // here: W == C.T*Y*T or C.T*Y*T.T

  // C2 = C2 - Y2*W.T
  armas_x_mult(__ONE, C2, -__ONE, Y2, W, ARMAS_TRANSA|ARMAS_TRANSB, conf);
  // W = Y1*W.T ==> W.T = W*Y1
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
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
int __update_lq_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
                      armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *W,
                      int transpose, armas_conf_t *conf)
{
  // W = C1
  armas_x_scale_plus(__ZERO, W, __ONE, C1, ARMAS_NONE, conf);
  // W = C1*Y1.T = W*Y1.T
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
  // W = W + C2*Y2.T
  armas_x_mult(__ONE, W, __ONE, C2, Y2, ARMAS_TRANSB, conf);
  // here: W = C*Y

  int bits = ARMAS_UPPER|ARMAS_RIGHT;
  if (! transpose)
    bits |= ARMAS_TRANSA;
  // W = W*T or W.T*T
  armas_x_mult_trm(W, __ONE, T, bits, conf);
  // here: W == C*Y*T or C*Y*T.T

  // C2 = C2 - W*Y2
  armas_x_mult(__ONE, C2, -__ONE, W, Y2, ARMAS_NONE, conf);
  // C1 = C1 - W*Y1
  //  W = W*Y1.T
  armas_x_mult_trm(W, __ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
  // C1 = C1 - W
  armas_x_scale_plus(__ONE, C1, -__ONE, W, ARMAS_NONE, conf);
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
int __unblk_lq_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
                         armas_conf_t *conf)
{
  double tauval;
  armas_x_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
  armas_x_dense_t TTL, TBR, T00, t01, T02, t11, t12, T22;
  armas_x_dense_t tT, tB, t0, t1, t2;

  EMPTY(A00); EMPTY(a11);
  EMPTY(t11);

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
    __repartition_2x2to3x3(&TTL,
                           &T00,  &t01,  &T02,
                           __nil, &t11,  &t12,
                           __nil, __nil, &T22,  /**/  T, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    tauval = armas_x_get(&t1, 0, 0);
    if (tauval != 0.0) {
      armas_x_set(&t11, 0, 0, tauval);
      // t01 := -tauval*(a01.T + &A02.T*a21)
      armas_x_axpby(__ZERO, &t01, __ONE, &a01, conf);
      armas_x_mvmult(-tauval, &t01, -tauval, &A02, &a12, ARMAS_NONE, conf);
      // t01 := T00*t01
      armas_x_mvmult_trm(&t01, __ONE, &T00, ARMAS_UPPER, conf);
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

int armas_x_lqreflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
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
 * \brief Compute LQ factorization of a M-by-N matrix A
 *
 * \param[in,out] A
 *    On entry, the M-by-N matrix A, M <= N. On exit, lower triangular matrix L
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * \param[out] tau
 *    On exit, the scalar factors of the elemenentary reflectors.
 *
 * \param[out] W
 *    Workspace, M-by-nb matrix used for work space in blocked invocations. 
 *
 * \param[in,out] conf
 *     The blocking configuration. If nil then default blocking configuration
 *     is used. Member conf.lb defines blocking size of blocked algorithms.
 *     If it is zero then unblocked algorithm is used.
 *
 * \retval  0 Success
 * \retval -1 Failure, `conf.error` holds error code
 *
 * #### Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *     \f$ Q = H_{k-1}H_{k-2}...H_0,  K = \min(M,N) \f$
 *
 * Elementary reflector H(k) is stored on row k of A right of the diagonal with
 * implicit unit value on diagonal entry. The vector TAU holds scalar factors of
 * the elementary reflectors.
 *
 * Contents of matrix A after factorization is as follow:
 *
 *      ( l  v0 v0 v0 v0 v0 )  for M=4, N=6
 *      ( l  l  v1 v1 v1 v1 )  l   is element of L
 *      ( l  l  l  v2 v2 v2 )  vk  is element of L
 *      ( l  l  l  l  v3 v3 )
 *
 * lqfactor() is compatible with lapack.DGELQF
 * \ingroup lapack
 */
int armas_x_lqfactor(armas_x_dense_t *A,
                     armas_x_dense_t *tau,
                     armas_x_dense_t *W,
                     armas_conf_t *cf)
{
  if (!cf)
    cf = armas_conf_default();

  armas_wbuf_t wb = ARMAS_WBNULL;
  if (armas_x_lqfactor_w(A, tau, &wb, cf) < 0)
    return -1;

  if (!armas_walloc(&wb, wb.bytes)) {
    cf->error = ARMAS_EMEMORY;
    return -1;
  }
  int stat = armas_x_lqfactor_w(A, tau, &wb, cf);
  armas_wrelease(&wb);
  return stat;
}

static inline
int __ws_lqfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*M : M;
}

/**
 * \brief Workspace size for LQ factorization
 * \ingroup lapack
 */
int armas_x_lqfactor_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_lqfactor(A->rows, A->cols, conf->lb);
}

/**
 * @brief Compute LQ factorization of a M-by-N matrix A
 *
 * @param[in,out] A
 *    On entry, the M-by-N matrix A, M <= N. On exit, lower triangular matrix L
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * @param[out] tau
 *    On exit, the scalar factors of the elemenentary reflectors.
 *
 * @param[out] wb
 *    Workspace buffer needed for factorization. If workspace is too small for blocked
 *    factorization then actual blocking factor is adjusted to fit into provided space.
 *    To compute size of the required space call the function with workspace bytes set to zero. 
 *    Size of workspace is returned in  `wb.bytes` and no other computation or parameter size 
 *    checking is done and function returns with success.
 *
 * @param[in,out] conf
 *     The blocking configuration. If nil then default blocking configuration
 *     is used. Member conf.lb defines blocking size of blocked algorithms.
 *     If it is zero then unblocked algorithm is used.
 *
 * @retval  0 Success
 * @retval -1 Failure, `conf.error` holds error code
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if N < M 
 *   - `ARMAS_EINVAL` tau is not column vector or size(tau) < M
 *   - `ARMAS_EWORK`  if workspace is less than M elements
 *
 * #### Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *     \f$ Q = H_{k-1}H_{k-2}...H_0,  K = \min(M,N) \f$
 *
 * Elementary reflector H(k) is stored on row k of A right of the diagonal with
 * implicit unit value on diagonal entry. The vector TAU holds scalar factors of
 * the elementary reflectors.
 *
 * Contents of matrix A after factorization is as follow:
 *
 *      ( l  v0 v0 v0 v0 v0 )  for M=4, N=6
 *      ( l  l  v1 v1 v1 v1 )  l   is element of L
 *      ( l  l  l  v2 v2 v2 )  vk  is element of H(k)
 *      ( l  l  l  l  v3 v3 )
 *
 * lqfactor() is compatible with lapack.DGELQF
 * \ingroup lapack
 */
int armas_x_lqfactor_w(armas_x_dense_t *A,
                       armas_x_dense_t *tau,
                       armas_wbuf_t *wb,
                       armas_conf_t *conf)
{
  armas_x_dense_t T, Wrk;
  size_t wsmin, wsz = 0;
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
      wb->bytes = (A->rows * conf->lb) * sizeof(DTYPE);
    else
      wb->bytes = A->rows * sizeof(DTYPE);
    return 0;
  }
  
  if (! armas_x_isvector(tau) || A->rows > armas_x_size(tau)) {
    conf->error = ARMAS_EINVAL;
    return -1;   
  }
  // must have: M <= N
  if (A->rows > A->cols) {
    conf->error = ARMAS_ESIZE;
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
    if (wsz < A->rows * lb) {
      lb = (wsz / A->rows) & ~0x3;
      if (lb < ARMAS_BLOCKING_MIN)
        lb = 0;
    }
  }
  wsz = armas_wpos(wb);
  buf = (DTYPE *)armas_wptr(wb);
  
  if (lb == 0 || A->rows <= lb) {
    armas_x_make(&Wrk, A->rows, 1, A->rows, buf);
    __unblk_lqfactor(A, tau, &Wrk, conf);
  } else {
    // block reflector [lb, lb]; temporary space [N(A)-lb,lb] matrix
    armas_x_make(&T, lb, lb, lb, buf);
    armas_x_make(&Wrk, A->rows-lb, lb, A->rows-lb, &buf[armas_x_size(&T)]);

    __blk_lqfactor(A, tau, &T, &Wrk, lb, conf);
  }
  armas_wsetpos(wb, wsz);
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

