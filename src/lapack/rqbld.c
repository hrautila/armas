
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_rqbuild) 
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
int __unblk_rqbuild(__armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  DTYPE tauval;
  __armas_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a12, a21, A22, D;
  __armas_dense_t tT, tB, t0, t1, t2, w12;


  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, mk, nk, ARMAS_PTOPLEFT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, mk, ARMAS_PTOP);
                 
  // zero the top part and to unit matrix
  if (mk > 0 && mayclear) {
    __armas_mscale(&ATL, 0.0, ARMAS_ANY);
    __armas_mscale(&ATR, 0.0, ARMAS_ANY);
    __armas_diag(&D, &ATL, ATL.cols-mk);
    __armas_add(&D, __ONE, conf);
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
    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a01), 1);

    __apply_householder2x1(&t1, &a10, &a01, &A00, &w12, ARMAS_RIGHT, conf);

    tauval = __armas_get(&t1, 0, 0);
    __armas_scale(&a10, -tauval, conf);
    __armas_set(&a11, 0, 0, 1.0 - tauval);

    // zero
    __armas_scale(&a12, __ZERO, conf);
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
int __blk_rqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *T,
                  __armas_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, ATR, A00, A01, A10, A11, A12, A22, AL, D;
  __armas_dense_t tT, tB, t0, t1, t2, w12, Tcur, Wrk;
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
    __armas_mscale(&ATR, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // number of reflector is not multiple of blocking factor
      __unblk_rqbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      __armas_mscale(&ATL, 0.0, ARMAS_ANY);
      __armas_diag(&D, &ATL, ATL.cols-ATL.rows);
      __armas_add(&D, __ONE, conf);
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
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, __ZERO, ARMAS_ANY);
    __unblk_rq_reflector(&Tcur, &AL, &t1, conf);

    // update A00, A01
    __armas_submatrix(&Wrk, W, 0, 0, A01.rows, A01.cols);
    __update_rq_right(&A01, &A00, &A11, &A10, &Tcur, &Wrk, TRUE, conf);

    // update current block
    __unblk_rqbuild(&AL, &t1, W, 0, A10.cols, FALSE, conf);

    // zero top rows
    __armas_mscale(&A12, __ZERO, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        &ABL,  &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PBOTTOM);
  }
  return 0;
}

/*
 * Generates the M by N matrix Q with orthonormal rows which
 * are defined as the last M rows of the product of K elementary
 * reflectors of order N.
 *
 *  Q = H(0)H(1)...H(k-1)  , 0 < k < M, where H(i) = I - tau*v*v.T
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by rqfactor().
 *         On exit, the orthogonal matrix Q
 *
 *   tau   Scalar coefficents of elementary reflectors
 *
 *   W     Workspace
 *
 *   K     The number of elementary reflector whose product define the matrix Q
 *
 *   conf  Optional blocking configuration.
 *
 * Compatible with lapackd.ORGRQ.
 */
int __armas_rqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_rqbuild(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_rqbuild(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, __armas_size(W), __ws_rqbuild);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    // start row: A->rows - K, column: A.cols - K
    __unblk_rqbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    __armas_dense_t T, Wrk;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, M(A)-lb-by-lb matrix
    __armas_make(&Wrk, A->rows-lb, lb, A->rows-lb, &__armas_data(W)[__armas_size(&T)]);

    __blk_rqbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
}


int __armas_rqbuild_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_rqbuild(A->rows, A->cols, conf->lb);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
