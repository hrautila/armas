
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_bkfactor) && defined(__armas_bksolve)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__ldlbk) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

static inline
int __ws_ldlfactor(int M, int N, int lb)
{
  return lb == 0 ? 2*N : (lb+1)*N;
}


/*
 * Compute LDL^T factorization of real symmetric matrix.
 *
 * Computes of a real symmetric matrix A using Bunch-Kauffman pivoting method.
 * The form of factorization is 
 *
 *    A = L*D*L.T  or A = U*D*U.T
 *
 * where L (or U) is product of permutation and unit lower (or upper) triangular matrix
 * and D is block diagonal symmetric matrix with 1x1 and 2x2 blocks.
 *
 * Arguments
 *  A     On entry, the N-by-N symmetric matrix A. If flags bit LOWER (or UPPER) is set then
 *        lower (or upper) triangular matrix and strictly upper (or lower) part is not
 *        accessed. On exit, the block diagonal matrix D and lower (or upper) triangular
 *        product matrix L (or U).
 *
 *  W     Workspace, size as returned by WorksizeBK().
 *
 *  P     Pivot vector. On exit details of interchanges and the block structure of D. If
 *        P[k] > 0 then D[k,k] is 1x1 and rows and columns k and P[k]-1 were changed.
 *        If P[k] == P[k+1] < 0 then D[k,k] is 2x2. If A is lower then rows and
 *        columns k+1 and ipiv[k]-1  were changed. And if A is upper then rows and columns
 *        k and P[k]-1 were changed.
 *
 *  flags Indicator bits, LOWER or UPPER.
 *
 *  confs Optional blocking configuration. If not provided then default blocking
 *        as returned by DefaultConf() is used. 
 *
 *  Unblocked algorithm is used if blocking configuration LB is zero or if N < LB.
 *
 *  Compatible with lapack.SYTRF.
 */
int __armas_bkfactor(__armas_dense_t *A, __armas_dense_t *W,
                     armas_pivot_t *P, int flags, armas_conf_t *conf)
{
  __armas_dense_t Wrk;
  int lb, k, wsmin, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_ldlfactor(A->rows, A->cols, lb);
  if (__armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_ldlfactor(A->rows, A->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(A->rows, A->cols, wsneed, __ws_ldlfactor);
    lb = min(lb, conf->lb);
    if (lb < 5)
      lb = 0;
  }

  if (A->rows != A->cols || A->cols != armas_pivot_size(P)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  // clear pivots
  for (k = 0; k < armas_pivot_size(P); k++) {
    armas_pivot_set(P, k, 0);
  }

  if (lb == 0 || A->cols <= lb) {
    __armas_make(&Wrk, A->rows, 2, A->rows, __armas_data(W));
    if (flags & ARMAS_UPPER) {
      __unblk_bkfactor_upper(A, &Wrk, P, conf);
    }
    else {
      __unblk_bkfactor_lower(A, &Wrk, P, conf);
    }
  }
  else {
    __armas_make(&Wrk, A->rows, lb+1, A->rows, __armas_data(W));
    if (flags & ARMAS_UPPER) {
      __blk_bkfactor_upper(A, &Wrk, P, lb, conf);
    }
    else {
      __blk_bkfactor_lower(A, &Wrk, P, lb, conf);
    }
  }
  return 0;
}



int __armas_bkfactor_work(__armas_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_ldlfactor(A->rows, A->cols, conf->lb);
}




/*
 * Solve A*X = B with symmetric real matrix A.
 *
 * Solves a system of linear equations A*X = B with a real symmetric matrix A using
 * the factorization A = U*D*U**T or A = L*D*L**T computed by ldlfactor().
 *
 * Arguments
 *  B     On entry, right hand side matrix B. On exit, the solution matrix X.
 *
 *  A     Block diagonal matrix D and the multipliers used to compute factor U
 *        (or L) as returned by ldlfactor_sym().
 *
 *  P     Block structure of matrix D and details of interchanges.
 *
 *  flags Indicator bits, LOWER or UPPER.
 *
 *  confs Optional blocking configuration.
 *
 * Currently only unblocked algorightm implemented. Compatible with lapack.SYTRS.
 */
int __armas_bksolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *W,
                    armas_pivot_t *P, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (A->cols != B->rows) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (flags & ARMAS_LOWER) {
    // first part: Z = D.-1*(L.-1*B)
    __unblk_bksolve_lower(B, A, P, 1, conf);
    // second part: X = L.-T*Z
    __unblk_bksolve_lower(B, A, P, 2, conf);
  }
  else {
    // first part: Z = D.-1*(U.-1*B)
    __unblk_bksolve_upper(B, A, P, 1, conf);
    // second part: X = U.-T*Z
    __unblk_bksolve_upper(B, A, P, 2, conf);
  }
}




#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

