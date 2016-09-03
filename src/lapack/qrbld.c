
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate the orthogonal matrix of QR factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qrbuild) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1) 
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
int __ws_qrbuild(int M, int N, int lb)
{
  return lb > 0 ? lb*N : N;
}


/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(1)H(2)...H(k)  , 0 < k <= N, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(1)*(H(2)...*(H(k)*I)) ie. from right to left.
 *
 * If k < N columns k+1:N are cleared and diagonal entries [k+1:N,k+1:N] are
 * set to unit. Then the matrix Q is generated by left multiplying elements right
 * of i'th elementary reflector H(i).
 * 
 * Compatible to lapack.xORG2R subroutine.
 */
static
int __unblk_qrbuild(armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ATR, ABR, A00, a01, a11, a12, a21, A22;
  armas_x_dense_t tT, tB, t0, t1, t2, w12, D;
  DTYPE tauval;

  EMPTY(ATL); EMPTY(A00); EMPTY(a11);

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR,   /**/  A, mk, nk, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, nk, ARMAS_PBOTTOM);
                 
  // zero the right side
  if (nk > 0 && mayclear) {
    armas_x_mscale(&ATR, 0.0, ARMAS_ANY);
    armas_x_mscale(&ABR, 0.0, ARMAS_ANY);
    armas_x_diag(&D, &ABR, 0);
    armas_x_add(&D, 1.0, conf);
  }

  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a12), 1);
    __apply_householder2x1(&t1, &a21, &a12, &A22, &w12, ARMAS_LEFT, conf);
    
    tauval = armas_x_get(&t1, 0, 0);
    armas_x_scale(&a21, -tauval, conf);
    armas_x_set(&a11, 0, 0, 1.0 - tauval);

    // zero
    armas_x_scale(&a01, 0.0, conf);

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}



/*
 * Blocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * If the number K of elementary reflectors is not multiple of the blocking
 * factor lb, then unblocked code is used first to generate the lower right corner
 * of the matrix Q. 
 *
 * Compatible with lapack.DORGQR subroutine.
 */
static
int __blk_qrbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *T,
                  armas_x_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ATR, ABR, A00, A01, A11, A12, A21, A22, AL;
  armas_x_dense_t tT, tB, t0, t1, t2, D, Wrk;
  int nk, mk, uk;

  nk = A->cols - K;
  mk = A->rows - K;
  uk = K % lb;

  EMPTY(ATL); EMPTY(A00); 

  __partition_2x2(&ATL,  &ATR,
                  __nil, &ABR,   /**/  A, mk+uk, nk+uk, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, nk+uk, ARMAS_PBOTTOM);
                 
  // zero the right side
  if (nk+uk > 0 ) {
    armas_x_mscale(&ATR, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // blocking factor not multiple of K, do the first uk
      // columns with unblocked code
      __unblk_qrbuild(&ABR, &tB, W, ABR.rows-uk, ABR.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      armas_x_mscale(&ABR, 0.0, ARMAS_ANY);
      armas_x_diag(&D, &ABR, 0);
      armas_x_add(&D, 1.0, conf);
    }
  }

  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb,  ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __merge2x1(&AL, &A11, &A21);
    
    // block reflector
    __unblk_qr_reflector(T, &AL, &t1, conf);
    
    // update rightside i.e A12, A22
    armas_x_submatrix(&Wrk, W, 0, 0, A12.cols, A12.rows);
    __update_qr_left(&A12, &A22, &A11, &A21, T, &Wrk, FALSE, conf);
    
    // update current block
    __unblk_qrbuild(&AL, &t1, W, A21.rows, 0, FALSE, conf);

    // zero top rows
    armas_x_mscale(&A01, 0.0, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}


/**
 * \brief Generate the orthogonal matrix Q
 *
 * Generate the M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * \param[in,out]  A
 *   On entry, the elementary reflectors as returned by qrfactor().
 *   stored below diagonal of the M by N matrix A.
 *   On exit, the orthogonal matrix Q
 * \param[in]  tau
 *   Scalar coefficents of elementary reflectors
 * \param[out]   W
 *    Workspace
 * \param[in]  K
 *    The number of elementary reflectors whose product define the matrix Q
 * \param[in,out] conf
 *    Blocking configuration
 *
 * Compatible with lapackd.ORGQR.
 * \ingroup lapack
 */
int armas_x_qrbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, wsneed, lb;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_qrbuild(A->rows, A->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_qrbuild(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_qrbuild);
    lb = min(lb, conf->lb);
  }
  if (lb == 0 || A->cols <= lb) {
    __unblk_qrbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    armas_x_dense_t T, Wrk;
    // block reflector at start of workspace
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    armas_x_make(&Wrk, A->cols-lb, lb, A->cols-lb, &armas_x_data(W)[armas_x_size(&T)]);

    __blk_qrbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
  return 0;
}


int armas_x_qrbuild_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qrbuild(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

