
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Solve system of linear inequalities

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_lqsolve) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_lqmult) 
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
int __ws_lqsolve(int M, int N, int lb)
{
  return lb == 0 ? M : lb*(M+lb);
}

/**
 * \brief Solve a system of linear equations
 *
 * Solve a system of linear equations A*X = B with general M-by-N (M < N)
 * matrix A using the LQ factorization computed by lqfactor().
 *
 * If *ARMAS_TRANS is set:
 *   find the minimum norm solution of an overdetermined system \f$ A^T X = B \f$
 *   i.e \f$ min ||X|| s.t A^T X = B \f$
 *
 * Otherwise:
 *   find the least squares solution of an overdetermined system, i.e.,
 *   solve the least squares problem: \f$ min || B - A X || \f$
 *
 * \param[in,out] B
 *     On entry, the right hand side N-by-P matrix B.
 *     On exit, the solution matrix X.
 *
 * \param[in] A
 *     The elements on and below the diagonal contain the min(M,N)-by-N lower
 *     trapezoidal matrix `L`. The elements right the diagonal with the vector `tau`, 
 *     represent the ortogonal matrix Q as product of elementary reflectors.
 *     Matrix `A` and `tau` are as returned by lqfactor()
 *
 * \param[in] tau
 *   The vector of N scalar coefficients that together with triuu(A) define
 *   the ortogonal matrix Q as \f$ Q = H_1 H_2...H_{N-1} \f$
 *
 * \param[out] W
 *     Workspace, size required returned lqsolve_work().
 *
 * \param[in] flags 
 *    Indicator flags, *ARMAS_TRANS*
 *
 * \param[in,out] conf
 *     Optinal blocking configuration. If not given default will be used. Unblocked
 *     invocation is indicated with conf.lb == 0.
 *
 * Compatible with lapack.GELS (the m >= n part)
 */
int armas_x_lqsolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t L, BL, BB;
  int wsmin, ok;

  if (!conf)
    conf = armas_conf_default();

  ok = B->rows == A->cols;
  if ( !ok ) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  wsmin = __ws_lqsolve(B->rows, B->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  armas_x_submatrix(&L, A, 0, 0, A->rows, A->rows);
  armas_x_submatrix(&BL, B, 0, 0, A->rows, B->cols);

  if (flags & ARMAS_TRANS) {
    // solve least square problem min || A.T*X - B ||

    // B' = Q.T*B
    ONERROR(armas_x_lqmult(B, A, tau, W, ARMAS_LEFT, conf));
    
    // X = L.-1*B'
    ONERROR(armas_x_solve_trm(&BL, &L, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSA, conf));

  } else {
    // solve underdetermined system A*X = B
    // B' = L.-1*B
    ONERROR(armas_x_solve_trm(&BL, &L, 1.0, ARMAS_LEFT|ARMAS_LOWER, conf));

    // clear bottom part of B
    armas_x_submatrix(&BB, B, A->rows, 0, -1, -1);
    armas_x_mscale(&BB, 0.0, ARMAS_ANY);
    
    // X = Q.T*B'
    ONERROR(armas_x_lqmult(B, A, tau, W, ARMAS_LEFT|ARMAS_TRANS, conf));
  }
  return 0;
}


//! \brief Workspace size for lqsolve.
int armas_x_lqsolve_work(armas_x_dense_t *B, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_lqsolve(B->rows, B->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
  
// Local Variables
// indent-tabs-mode: nil
// End:

