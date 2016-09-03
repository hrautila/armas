
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal matrix Q of QL factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qlbuild) 
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

static inline
int __ws_qlbuild(int M, int N, int lb)
{
  return lb > 0 ? lb*N : N;
}


/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameter nk is last nk elementary reflectors that are not used in computing
 * the matrix Q. Parameter mk length of the first unused elementary reflectors
 * First nk columns are zeroed and subdiagonal mk-nk is set to unit.
 *
 * Compatible with lapack.DORG2L subroutine.
 */
static
int __unblk_qlbuild(armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a21, A22;
  armas_x_dense_t tT, tB, t0, t1, t2, w12, D;
  DTYPE tauval;

  EMPTY(a11);

  // (mk, nk) = (rows, columns) of upper left partition
  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk, nk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, nk, ARMAS_PTOP);
                 
  // zero the left side
  if (nk > 0 && mayclear) {
    armas_x_mscale(&ABL, 0.0, ARMAS_ANY);
    armas_x_mscale(&ATL, 0.0, ARMAS_ANY);
    armas_x_diag(&D, &ATL, nk-mk);
    armas_x_add(&D, 1.0, conf);
  }

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           &a10,  &a11,  __nil,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a10), 1);
    __apply_householder2x1(&t1, &a01, &a10, &A00, &w12, ARMAS_LEFT, conf);
    
    tauval = armas_x_get(&t1, 0, 0);
    armas_x_scale(&a01, -tauval, conf);
    armas_x_set(&a11, 0, 0, 1.0 - tauval);

    // zero bottom elements
    armas_x_scale(&a21, 0.0, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}


/*
 * Blocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * If the number K of elementary reflectors is not multiple of the blocking
 * factor lb, then unblocked code is used first to generate the upper left corner
 * of the matrix Q. 
 *
 * Compatible with lapack.DORGQL subroutine.
 */
static
int __blk_qlbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *T,
                  armas_x_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABL, ATR, ABR, A00, A01, A10, A11, A21, A22, AT;
  armas_x_dense_t tT, tB, t0, t1, t2, D, Tcur, Wrk;
  int mk, nk, uk;

  nk = A->cols - K;
  mk = A->rows - K;
  uk = K % lb;

  // (mk, nk) = (rows, columns) of upper left partition
  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk+uk, nk+uk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, nk+uk, ARMAS_PTOP);
                 
  // zero the left side
  if (nk+uk > 0) {
    armas_x_mscale(&ABL, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // number of reflectors is not multiple of blocking factor
      // do the first part with unblocked code.
      __unblk_qlbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      armas_x_mscale(&ATL, 0.0, ARMAS_ANY);
      armas_x_diag(&D, &ATL, nk-mk);
      armas_x_add(&D, 1.0, conf);
    }
  }

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           &A10,  &A11,  __nil,
                           __nil, &A21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    __merge2x1(&AT, &A01, &A11);
    
    // build block reflector
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update left side i.e A00 and A00 with (I - Y*T*Y.T)
    armas_x_submatrix(&Wrk, W, 0, 0, A10.cols, A10.rows);
    __update_ql_left(&A10, &A00, &A11, &A01, &Tcur, &Wrk, FALSE, conf);

    // update current block
    __unblk_qlbuild(&AT, &t1, W, A01.rows, 0, FALSE, conf);

    // zero bottom rows
    armas_x_mscale(&A21, 0.0, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}


/**
 * \brief Generate orthogonal Q matrix of QL factorization
 *
 * Generate the M-by-N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * \param[in,out]  A
 *     On entry, the elementary reflectors as returned by qlfactor().
 *     stored below diagonal of the M by N matrix A.
 *     On exit, the orthogonal matrix Q
 *
 * \param[in]  tau
 *    Scalar coefficents of elementary reflectors
 *
 * \param[out] W
 *     Workspace
 *
 * \param[in]   K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * \param[in,out] conf
 *     Blocking configuration
 *
 * \retval  0 Succes
 * \retval -1 Failure, conf.error holds error code.
 *
 * Compatible with lapackd.ORGQL.
 */
int armas_x_qlbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_qlbuild(A->rows, A->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_qlbuild(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, armas_x_size(W), __ws_qlbuild);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    __unblk_qlbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    armas_x_dense_t T, Wrk;
    // block reflector at start of workspace
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    armas_x_make(&Wrk, A->cols-lb, lb, A->cols-lb, &armas_x_data(W)[armas_x_size(&T)]);

    __blk_qlbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
  return 0;
}


int armas_x_qlbuild_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qlbuild(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

