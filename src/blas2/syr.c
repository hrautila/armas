
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank update

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvupdate_sym)
#define __ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(__update_trmv_unb) && defined(__update_trmv_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

#if EXT_PRECISION && defined(__update_trmv_ext_unb)
#define HAVE_EXT_PRECISION 1
extern int __update_trmv_ext_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                                 DTYPE alpha, int flags, int N, int M);
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"
//! \endcond

/**
 * @brief Symmetric matrix rank-1 update.
 *
 * Computes 
 *    - \f$ A = A + alpha \times X X^T \f$
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag *ARMAS_LOWER* (*ARMAR_UPPER*) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      X source vector
 * @param[in]      flags flag bits 
 * @param[in]      conf configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvupdate_sym(armas_x_dense_t *A,
                         DTYPE alpha, const armas_x_dense_t *X,
                         int flags, armas_conf_t *conf)
{
  mvec_t x;
  mdata_t A0;
  int nx = armas_x_size(X);

  if (armas_x_size(A) == 0 || armas_x_size(X) == 0)
    return 0;
  
  if (!conf)
    conf = armas_conf_default();

  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (A->cols != nx || A->rows != nx) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  // if extended precision enable and requested
  if (HAVE_EXT_PRECISION && (conf->optflags&ARMAS_OEXTPREC)) {
    __update_trmv_ext_unb(&A0, &x, &x, alpha, flags, nx, nx);
    return 0;
  }

  // default precision
  switch (conf->optflags) {
  case ARMAS_ORECURSIVE:
    __update_trmv_recursive(&A0, &x, &x, alpha, flags, nx, nx);
    break;

  case ARMAS_ONAIVE:
  default:
    __update_trmv_unb(&A0, &x, &x, alpha, flags, nx, nx);
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
