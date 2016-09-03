
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix factorization

//! \ingroup lapack

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bkfactor) && defined(armas_x_bksolve)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__ldlbk) 
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
int __ws_ldlfactor(int M, int N, int lb)
{
  return lb == 0 ? 2*N : (lb+1)*N;
}


/**
 * \brief Compute LDL^T factorization of real symmetric matrix.
 *
 * Computes of a real symmetric matrix A using Bunch-Kauffman pivoting method.
 * The form of factorization is 
 *
 *    \f$ A = LDL^T \f$ or \f$ A = UDU^T \f$
 *
 * where L (or U) is product of permutation and unit lower (or upper) triangular matrix
 * and D is block diagonal symmetric matrix with 1x1 and 2x2 blocks.
 *
 * \param[in,out] A
 *      On entry, the N-by-N symmetric matrix A. If flags bit *ARMAS_LOWER* (or *ARMSA_UPPER*)
 *      is set then lower (or upper) triangular matrix and strictly upper (or lower) part
 *      is not accessed. On exit, the block diagonal matrix D and lower (or upper) triangular
 *      product matrix L (or U).
 *  \param[in] W
 *      Workspace, size as returned by `bkfactor_work()`.
 *  \param[out] P
 *      Pivot vector. On exit details of interchanges and the block structure of D. If
 *      \f$ P[k] > 0 \f$ then \f$ D[k,k] \f$ is 1x1 and rows and columns k and \f$ P[k]-1 \f$ 
 *      were changed. If \f$ P[k] == P[k+1] < 0 \f$ then \f$ D[k,k] \f$ is 2x2.
 *      If A is lower then rows and columns \f$ k+1,  P[k]-1 \f$ were changed. 
 *      And if A is upper then rows and columns \f$ k, P[k]-1 \f$ were changed.
 * \param[in] flags 
 *      Indicator bits, *ARMAS_LOWER* or *ARMAS_UPPER*.
 *  \param[in,out] conf 
 *      Optional blocking configuration. If not provided then default blocking
 *      as returned by `armas_conf_default()` is used. 
 *
 *  Unblocked algorithm is used if blocking configuration `conf.lb` is zero or 
 *  if `N < conf.lb`.
 *
 *  Compatible with lapack.SYTRF.
 * \ingroup lapack
 */
int armas_x_bkfactor(armas_x_dense_t *A, armas_x_dense_t *W,
                     armas_pivot_t *P, int flags, armas_conf_t *conf)
{
  armas_x_dense_t Wrk;
  int lb, k, wsmin, wsneed;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  wsmin = __ws_ldlfactor(A->rows, A->cols, lb);
  if (armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = __ws_ldlfactor(A->rows, A->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
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
    armas_x_make(&Wrk, A->rows, 2, A->rows, armas_x_data(W));
    if (flags & ARMAS_UPPER) {
      __unblk_bkfactor_upper(A, &Wrk, P, conf);
    }
    else {
      __unblk_bkfactor_lower(A, &Wrk, P, conf);
    }
  }
  else {
    armas_x_make(&Wrk, A->rows, lb+1, A->rows, armas_x_data(W));
    if (flags & ARMAS_UPPER) {
      __blk_bkfactor_upper(A, &Wrk, P, lb, conf);
    }
    else {
      __blk_bkfactor_lower(A, &Wrk, P, lb, conf);
    }
  }
  return 0;
}

/**
 * \brief Compute workspace size for `bkfactor()`
 *
 * \param[in] A
 *    The input matrix
 * \param[in] conf
 *    Blocking configuration
 *
 * \return Workspace size as number of elements.
 * \ingroup lapack
 */
int armas_x_bkfactor_work(armas_x_dense_t *A, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_ldlfactor(A->rows, A->cols, conf->lb);
}

/**
 * \brief Solve \f$ AX = B \f$ with symmetric real matrix A.
 *
 * Solves a system of linear equations AX = B with a real symmetric matrix A using
 * the factorization \f$ A = UDU^T \f$ or \f$ A = LDL^T \f$ computed by `bkfactor()`.
 *
 * \param[in,out] B
 *      On entry, right hand side matrix B. On exit, the solution matrix X.
 * \param[in] A
 *      Block diagonal matrix D and the multipliers used to compute factor U
 *      (or L) as returned by `bkfactor()`.
 *  \param[in] W
 *      Workspace, not used at the moment.
 * \param[in] P
 *      Block structure of matrix D and details of interchanges.
 * \param[in] flags
 *      Indicator bits, *ARMAS_LOWER* or *ARMAS_UPPER*.
 * \param[in,out] conf 
 *      Optional blocking configuration.
 *
 * Currently only unblocked algorightm implemented. Compatible with lapack.SYTRS.
 * \ingroup lapack
 */
int armas_x_bksolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *W,
                    armas_pivot_t *P, int flags, armas_conf_t *conf)
{
  int err = 0;
  if (!conf)
    conf = armas_conf_default();

  if (A->cols != B->rows) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (flags & ARMAS_LOWER) {
    // first part: Z = D.-1*(L.-1*B)
    if ((err = __unblk_bksolve_lower(B, A, P, 1, conf)) < 0)
      return err;
    // second part: X = L.-T*Z
    err = __unblk_bksolve_lower(B, A, P, 2, conf);
  }
  else {
    // first part: Z = D.-1*(U.-1*B)
    if ((err = __unblk_bksolve_upper(B, A, P, 1, conf)) < 0)
      return err;
    // second part: X = U.-T*Z
    err =__unblk_bksolve_upper(B, A, P, 2, conf);
  }
  return err;
}




#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

