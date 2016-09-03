
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal Q of RQ factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_rqbuild) 
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
//! \endcond

static inline
int __ws_rqbuild(int M, int N, int lb)
{
  return lb > 0 ? lb*M : M;
}

/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(0)H(1)...H(k-1)  , 0 < k < M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(0)*H(1)...*H(k-1)*I ie. from top to bottom.
 *
 * Compatible to lapack.xORG2R subroutine.
 */
static
int __unblk_rqbuild(armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  DTYPE tauval;
  armas_x_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a12, A22, D;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;


  EMPTY(a11);

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk, nk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, mk, ARMAS_PTOP);
                 
  // zero the top part and to unit matrix
  if (mk > 0 && mayclear) {
    armas_x_mscale(&ATL, 0.0, ARMAS_ANY);
    armas_x_mscale(&ATR, 0.0, ARMAS_ANY);
    armas_x_diag(&D, &ATL, ATL.cols-mk);
    armas_x_add(&D, __ONE, conf);
  }

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           &a10,  &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a01), 1);

    __apply_householder2x1(&t1, &a10, &a01, &A00, &w12, ARMAS_RIGHT, conf);

    tauval = armas_x_get(&t1, 0, 0);
    armas_x_scale(&a10, -tauval, conf);
    armas_x_set(&a11, 0, 0, 1.0 - tauval);

    // zero
    armas_x_scale(&a12, __ZERO, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}

/*
 * Blocked code.
 */
static
int __blk_rqbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *T,
                  armas_x_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABL, ABR, ATR, A00, A01, A10, A11, A12, A22, AL, D;
  armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  int mk, nk, uk;
  
  mk = A->rows - K;
  nk = A->cols - K;
  uk = K % lb;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk+uk, nk+uk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,    /**/  tau, mk+uk, ARMAS_PTOP);
                 
  // zero the top part
  if (mk+uk > 0) {
    armas_x_mscale(&ATR, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // number of reflector is not multiple of blocking factor
      __unblk_rqbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      armas_x_mscale(&ATL, 0.0, ARMAS_ANY);
      armas_x_diag(&D, &ATL, ATL.cols-ATL.rows);
      armas_x_add(&D, __ONE, conf);
    }
  }

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           &A10,  &A11,  &A12,
                           __nil, __nil, &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, A11.cols, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    __merge1x2(&AL, &A10, &A11);

    // build block reflector
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    armas_x_mscale(&Tcur, __ZERO, ARMAS_ANY);
    __unblk_rq_reflector(&Tcur, &AL, &t1, conf);

    // update A00, A01
    armas_x_submatrix(&Wrk, W, 0, 0, A01.rows, A01.cols);
    __update_rq_right(&A01, &A00, &A11, &A10, &Tcur, &Wrk, TRUE, conf);

    // update current block
    __unblk_rqbuild(&AL, &t1, W, 0, A10.cols, FALSE, conf);

    // zero top rows
    armas_x_mscale(&A12, __ZERO, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        &ABL,  &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}

/**
 * \brief Generate the orthogonal Q matrix of RQ factorization
 *
 * Generates the M by N matrix Q with orthonormal rows which
 * are defined as the last M rows of the product of K elementary
 * reflectors of order N.
 *
 *   \f$ Q = H_0 H_1...H_{k-1} , 0 < k < M, H_i = I - tau*v*v^T \f$
 *
 * \param[in,out]  A
 *     On entry, the elementary reflectors as returned by rqfactor().
 *     On exit, the orthogonal matrix Q
 *
 * \param[in]  tau
 *    Scalar coefficents of elementary reflectors
 *
 * \param[out]  W
 *      Workspace
 *
 * \param[in]  K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * \param[in,out] conf
 *     Optional blocking configuration.
 *
 * \retval  0 Succes
 * \retval -1 Failure, `conf.error` holds error code.
 *
 * Compatible with lapackd.ORGRQ.
 */
int armas_x_rqbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_rqbuild(A->rows, A->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_rqbuild(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_rqbuild);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    // start row: A->rows - K, column: A.cols - K
    __unblk_rqbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    armas_x_dense_t T, Wrk;
    // block reflector at start of workspace
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));
    // temporary space after block reflector T, M(A)-lb-by-lb matrix
    armas_x_make(&Wrk, A->rows-lb, lb, A->rows-lb, &armas_x_data(W)[armas_x_size(&T)]);

    __blk_rqbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
  return 0;
}


int armas_x_rqbuild_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_rqbuild(A->rows, A->cols, conf->lb);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

