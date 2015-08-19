
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_qrsolve) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_qrmult) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

static inline
int __ws_qrsolve(int M, int N, int lb)
{
  return lb == 0 ? N : lb*(N+lb);
}

/*
 * Solve a system of linear equations A*X = B with general M-by-N
 * matrix A using the QR factorization computed by __qrfactor().
 *
 * If flags&TRANS != 0:
 *   find the minimum norm solution of an overdetermined system A.T * X = B.
 *   i.e min ||X|| s.t A.T*X = B
 *
 * Otherwise:
 *   find the least squares solution of an overdetermined system, i.e.,
 *   solve the least squares problem: min || B - A*X ||.
 *
 * Arguments:
 *  B     On entry, the right hand side N-by-P matrix B.
 *        On exit, the solution matrix X.
 *
 *  A     The elements on and above the diagonal contain the min(M,N)-by-N upper
 *        trapezoidal matrix R. The elements below the diagonal with the vector 'tau', 
 *        represent the ortogonal matrix Q as product of elementary reflectors.
 *        Matrix A and T are as returned by DecomposeQR()
 *
 *  tau   The vector of N scalar coefficients that together with trilu(A) define
 *        the ortogonal matrix Q as Q = H(1)H(2)...H(N)
 *
 *  W     Workspace, size required returned WorksizeMultQ().
 *
 *  flags Indicator flags
 *
 *  conf  Optinal blocking configuration. If not given default will be used. Unblocked
 *        invocation is indicated with conf.LB == 0.
 *
 * Compatible with lapack.GELS (the m >= n part)
 */
int __armas_qrsolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t R, BT, BB;
  int wsmin, ok;

  if (!conf)
    conf = armas_conf_default();

  ok = B->rows == A->rows;
  if ( !ok ) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  wsmin = __ws_qrsolve(B->rows, B->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  __armas_submatrix(&R, A, 0, 0, A->cols, A->cols);
  __armas_submatrix(&BT, B, 0, 0, A->cols, B->cols);

  if (flags & ARMAS_TRANS) {
    // solve ovedetermined system A.T*X = B

    // B' = R.-1*B
    ONERROR(__armas_solve_trm(&BT, &R, 1.0, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA, conf));

    // clear bottom part of B
    __armas_submatrix(&BB, B, A->cols, 0, -1, -1);
    __armas_mscale(&BB, 0.0, ARMAS_ANY);
    
    // X = Q*B
    ONERROR(__armas_qrmult(B, A, tau, W, ARMAS_LEFT, conf));
  } else {
    // solve least square problem min || A*X - B ||

    // B' = Q.T*B
    ONERROR(__armas_qrmult(B, A, tau, W, ARMAS_LEFT|ARMAS_TRANS, conf));
    
    // X = R.-1*B'
    ONERROR(__armas_solve_trm(&BT, &R, 1.0, ARMAS_LEFT|ARMAS_UPPER, conf));
  }
  return 0;
}


int __armas_qrsolve_work(__armas_dense_t *B, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  return __ws_qrsolve(B->rows, B->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
  
// Local Variables
// indent-tabs-mode: nil
// End:

