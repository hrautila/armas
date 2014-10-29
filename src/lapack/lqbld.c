
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_lqbuild) 
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
int __ws_lqbuild(int M, int N, int lb)
{
  return lb > 0 ? lb*M : M;
}

/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(k)H(k-1)...H(1)  , 0 < k <= M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(k)*H(k-1)...*H(1)*I ie. from bottom to top.
 *
 * If k < M rows k+1:M are cleared and diagonal entries [k+1:M,k+1:M] are
 * set to unit. Then the matrix Q is generated by right multiplying elements below
 * of i'th elementary reflector H(i).
 * 
 * Compatible to lapack.xORG2L subroutine.
 */
static
int __unblk_lqbuild(__armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int mk, int nk, int mayclear, armas_conf_t *conf)
{
  DTYPE tauval;
  __armas_dense_t ATL, ABL, ABR, A00, a10, a11, a12, a21, A22, D;
  __armas_dense_t tT, tB, t0, t1, t2, w12;

  __partition_2x2(&ATL, __nil,
                  &ABL, &ABR,   /**/  A, mk, nk, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, mk, ARMAS_PBOTTOM);
                 
  // zero the bottom part
  if (mk > 0 && mayclear) {
    __armas_mscale(&ABL, 0.0, ARMAS_ANY);
    __armas_mscale(&ABR, 0.0, ARMAS_ANY);
    __armas_diag(&D, &ABR, 0);
    __armas_add(&D, __ONE, conf);
  }

  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &a10,  &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __armas_submatrix(&w12, W, 0, 0, __armas_size(&a21), 1);

    __apply_householder2x1(&t1, &a12, &a21, &A22, &w12, ARMAS_RIGHT, conf);

    tauval = __armas_get(&t1, 0, 0);
    __armas_scale(&a12, -tauval, conf);
    __armas_set(&a11, 0, 0, 1.0 - tauval);

    // zero
    __armas_scale(&a10, __ZERO, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}

/*
 * Blocked code.
 */
static
int __blk_lqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *T,
                  __armas_dense_t *W, int K, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, A00, A10, A11, A12, A21, A22, AL, D;
  __armas_dense_t tT, tB, t0, t1, t2, w12, Tcur, Wrk;
  int mk, nk, uk;
  
  mk = A->rows - K;
  nk = A->cols - K;
  uk = K % lb;

  __partition_2x2(&ATL, __nil,
                  &ABL, &ABR,   /**/  A, mk+uk, nk+uk, ARMAS_PBOTTOMRIGHT);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, mk+uk, ARMAS_PBOTTOM);
                 
  // zero the bottom part
  if (mk+uk > 0) {
    __armas_mscale(&ABL, 0.0, ARMAS_ANY);
    if (uk > 0) {
      // number of reflector is not multiple of blocking factor
      __unblk_lqbuild(&ABR, &tB, W, ABR.rows-uk, ABR.cols-uk, TRUE, conf);
    } else {
      // blocking factor is multiple of K
      __armas_mscale(&ABR, 0.0, ARMAS_ANY);
      __armas_diag(&D, &ABR, 0);
      __armas_add(&D, __ONE, conf);
    }
  }

  while (ATL.rows > 0 && ATL.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &A10,  &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PTOPLEFT);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, ARMAS_PTOP);
    // ---------------------------------------------------------------------------
    __merge1x2(&AL, &A11, &A12);

    // build block reflector
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, __ZERO, ARMAS_ANY);
    __unblk_lq_reflector(&Tcur, &AL, &t1, conf);

    // update A21, A22
    __armas_submatrix(&Wrk, W, 0, 0, A21.rows, A21.cols);
    __update_lq_right(&A21, &A22, &A11, &A12, &Tcur, &Wrk, FALSE, conf);

    // update current block
    __unblk_lqbuild(&AL, &t1, W, 0, A12.cols, FALSE, conf);

    // zero top rows
    __armas_mscale(&A10, __ZERO, ARMAS_ANY);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        &ABL,  &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PTOPLEFT);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, ARMAS_PTOP);
  }
  return 0;
}

/*
 * Generate the M by N matrix Q with orthogonal rows which
 * are defined as the first M rows of the product of K first elementary
 * reflectors.
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by DecomposeLQ().
 *         stored right of diagonal of the M by N matrix A.
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
 * Compatible with lapackd.ORGLQ.
 */
int __armas_lqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W, int K,
                    armas_conf_t *conf)
{
  int wsmin, lb, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_lqbuild(A->rows, A->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_lqbuild(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, __armas_size(W), __ws_lqbuild);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    __unblk_lqbuild(A, tau, W, A->rows-K, A->cols-K, TRUE, conf);
  } else {
    __armas_dense_t T, Wrk;
    // block reflector at start of workspace
    __armas_make(&T, lb, lb, lb, __armas_data(W));
    // temporary space after block reflector T, M(A)-lb-by-lb matrix
    __armas_make(&Wrk, A->rows-lb, lb, A->rows-lb, &__armas_data(W)[__armas_size(&T)]);

    __blk_lqbuild(A, tau, &T, &Wrk, K, lb, conf);
  }
  return 0;
}


int __armas_lqbuild_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_lqbuild(A->rows, A->cols, conf->lb);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

