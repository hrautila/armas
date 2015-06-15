
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_qrtfactor) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) && defined(__update_qr_left) && defined(__update_qr_right)
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
int __ws_qrtfactor(int M, int N, int lb)
{
  return lb > 0 ? lb*(N-lb) : N;
}

/*
 * Unblocked factorization.
 */
static
int __unblk_qrtfactor(__armas_dense_t *A, __armas_dense_t *T,
                     __armas_dense_t *W, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  __armas_dense_t TTL, TTR, TBR, T00, t01, t11, T22, w12;
  DTYPE tauval;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,  /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_2x2(&TTL,  &TTR,
                  __nil, &TBR,  /**/  T, 0, 0, ARMAS_PTOPLEFT);
                 
  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x2to3x3(&TTL,
                           &T00,  &t01,  __nil,
                           __nil, &t11,  __nil,
                           __nil, __nil, &T22,  /**/  T, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    __compute_householder(&a11, &a21, &t11, conf);

    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a12), 1);

    __apply_householder2x1(&t11, &a21, &a12, &A22, &w12, ARMAS_LEFT, conf);
    tauval = __armas_get_unsafe(&t11, 0, 0);
    if ( tauval != __ZERO) {
      // t01 := -tauval*(a10.T + A20.T*a21)
      __armas_axpby(&t01, &a12, 1.0, 0.0, conf);
      __armas_mvmult(&t01, &A22, &a21, -tauval, -tauval, ARMAS_TRANSA, conf);
      // t01 := T00*t01
      __armas_mvmult_trm(&t01, &T00, 1.0, ARMAS_UPPER, conf);
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x3to2x2(&TTL,  &TTR,
                        __nil, &TBR, /**/  &T00, &t11, &T22,   T, ARMAS_PBOTTOMRIGHT);
  }
  return 0;
}


/*
 * Blocked factorization.
 */
static
int __blk_qrtfactor(__armas_dense_t *A, __armas_dense_t *T, __armas_dense_t *W, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AL;
  __armas_dense_t TL, TR, TBR, T00, T01, T02, T11, T12, T22, w12;
  __armas_dense_t tT, tB, t0, t1, t2, w1, Wrk, row;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,  /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&TL,  &TR,    /**/  T, 0,    ARMAS_LEFT);
                 
  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22, /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_1x2to1x3(&TL,
                           &T00,  &T01, &T02,  /**/  T, lb, ARMAS_PRIGHT);
    // ---------------------------------------------------------------------------
    // decompose current panel AL = ( A11 )
    //                              ( A21 )
    __armas_submatrix(&w1, W, 0, 0, A11.rows, 1);
    __merge2x1(&AL, &A11, &A21);
    __unblk_qrtfactor(&AL, &T01, &w1, conf);

    // update ( A12 A22 ).T
    __armas_submatrix(&Wrk, W, 0, 0, A12.cols, A12.rows);
    __update_qr_left(&A12, &A22, &A11, &A21, &T01, &Wrk, TRUE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,  A, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&TL,  &TR,   /**/  &T00, &T01, /*&T02,*/  T, ARMAS_PRIGHT);
  }

  // last block with unblocked
  if (ABR.rows > 0 && ABR.cols > 0) {
    __armas_submatrix(&T01, &TR, 0, 0, ABR.cols, ABR.cols);
    __unblk_qrtfactor(&ABR, &T01, W, conf);
  }

  return 0;
}



/**
 * @brief Compute QR factorization of a M-by-N matrix A = Q * R.
 *
 * Arguments:
 *  A    On entry, the M-by-N matrix A, M >= N. On exit, upper triangular matrix R
 *       and the orthogonal matrix Q as product of elementary reflectors.
 *
 *  T    On exit, block reflectors
 *
 *  W    Workspace, N-by-lb matrix used for work space in blocked invocations. 
 *
 *  conf The blocking configuration. If nil then default blocking configuration
 *       is used. Member conf.LB defines blocking size of blocked algorithms.
 *       If it is zero then unblocked algorithm is used.
 *
 * @returns:
 *      0 if succesfull, -1 otherwise
 *
 * Additional information
 *
 *  Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *    Q = H(0)H(1),...,H(K-1), where K = min(M,N)
 *
 *  Elementary reflector H(k) is stored on column k of A below the diagonal with
 *  implicit unit value on diagonal entry. The matrix T holds Householder block
 *  reflectors.
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
 *  Compatible with lapack.xGEQRT
 */
int __armas_qrtfactor(__armas_dense_t *A, __armas_dense_t *T, __armas_dense_t *W,
                      armas_conf_t *conf)
{
  __armas_dense_t sT;
  int wsmin, wsneed, lb;
  if (!conf)
    conf = armas_conf_default();

  // must have: M >= N
  if (A->rows < A->cols || T->cols < A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  
  // set blocking factor to number of rows in T
  lb = T->rows;
  wsmin = __ws_qrtfactor(A->rows, A->cols, lb);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }

  if (lb == 0 || A->cols <= lb) {
    __armas_submatrix(&sT, T, 0, 0, A->cols, A->cols);
    __unblk_qrtfactor(A, &sT, W, conf);
  } else {
    __blk_qrtfactor(A, T, W, lb, conf);
  }
  return 0;
}

/*
 * Calculate required workspace to decompose matrix A with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int __armas_qrtfactor_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qrtfactor(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

