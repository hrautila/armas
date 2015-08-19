
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_qlbuild) 
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
int __unblk_qlbuild(__armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a21, A22;
  __armas_dense_t tT, tB, t0, t1, t2, w12, D;
  DTYPE tauval;

  EMPTY(a11);

  // (mk, nk) = (rows, columns) of upper left partition
  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk, nk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, nk, ARMAS_PTOP);
                 
  // zero the left side
  if (nk > 0 && mayclear) {
    __armas_mscale(&ABL, 0.0, ARMAS_ANY);
    __armas_mscale(&ATL, 0.0, ARMAS_ANY);
    __armas_diag(&D, &ATL, nk-mk);
    __armas_add(&D, 1.0, conf);
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
    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a10), 1);
    __apply_householder2x1(&t1, &a01, &a10, &A00, &w12, ARMAS_LEFT, conf);
    
    tauval = __armas_get(&t1, 0, 0);
    __armas_scale(&a01, -tauval, conf);
    __armas_set(&a11, 0, 0, 1.0 - tauval);

    // zero bottom elements
    __armas_scale(&a21, 0.0, conf);
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
int __blk_qlbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *T,
                  __armas_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ATR, ABR, A00, A01, A10, A11, A21, A22, AT;
  __armas_dense_t tT, tB, t0, t1, t2, D, Tcur, Wrk;
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
    __armas_mscale(&ABL, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // number of reflectors is not multiple of blocking factor
      // do the first part with unblocked code.
      __unblk_qlbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      __armas_mscale(&ATL, 0.0, ARMAS_ANY);
      __armas_diag(&D, &ATL, nk-mk);
      __armas_add(&D, 1.0, conf);
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
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update left side i.e A00 and A00 with (I - Y*T*Y.T)
    __armas_submatrix(&Wrk, W, 0, 0, A10.cols, A10.rows);
    __update_ql_left(&A10, &A00, &A11, &A01, &Tcur, &Wrk, FALSE, conf);

    // update current block
    __unblk_qlbuild(&AT, &t1, W, A01.rows, 0, FALSE, conf);

    // zero bottom rows
    __armas_mscale(&A21, 0.0, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  &ATR,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}


/*
 * Generate the M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by DecomposeQR().
 *         stored below diagonal of the M by N matrix A.
 *         On exit, the orthogonal matrix Q
 *
 *   tau   Scalar coefficents of elementary reflectors
 *
 *   W     Workspace
 *
 *   K     The number of elementary reflector whose product define the matrix Q
 *
 * Compatible with lapackd.ORGQL.
 */
int __armas_qlbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_qlbuild(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_qlbuild(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, __armas_size(W), __ws_qlbuild);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    __unblk_qlbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    __armas_dense_t T, Wrk;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, N(A)-lb-by-lb matrix
    __armas_make(&Wrk, A->cols-lb, lb, A->cols-lb, &__armas_data(W)[__armas_size(&T)]);

    __blk_qlbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
  return 0;
}


int __armas_qlbuild_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qlbuild(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

