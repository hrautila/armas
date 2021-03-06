
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_mvupdate_trm_unsafe) && defined(armas_mvupdate_sym)
#define ARMAS_PROVIDES 1
#endif

#if defined(armas_ext_axpby_dx_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 * (M is rows, N is columns.)
 */
int armas_ext_mvupdate_trm_unsafe(
    DTYPE beta,
    armas_dense_t *A,
    DTYPE alpha,
    const armas_dense_t *X,
    const armas_dense_t *Y,
    int flags)
{
    DTYPE p, r, xk;
    armas_dense_t a0;

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        for (int i = 0; i < A->rows; ++i) {
            armas_submatrix_unsafe(&a0, A, i, i, 1, A->cols-i);
            xk = armas_get_at(X, i);
            twoprod(&p, &r, alpha, xk);
            armas_ext_axpby_dx_unsafe(beta, &a0, p, r, Y);
        }
        break;
    case ARMAS_LOWER:
    default:
        for (int j = 0; j < A->cols; ++j) {
            armas_submatrix_unsafe(&a0, A, j, j, A->rows-j, 1);
            xk = armas_get_at(Y, j);
            twoprod(&p, &r, alpha, xk);
            armas_ext_axpby_dx_unsafe(beta, &a0, p, r, X);
        }
        break;
    }
    return 0;
}

/**
 * @brief Symmetric matrix rank-1 update in extended precision..
 *
 * Computes
 *    \f$ A = beta \times A + alpha \times x x^T \f$
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag *ARMAS_LOWER* (*ARMAR_UPPER*) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * @param[in]      beta scalar multiplier
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      x source vector
 * @param[in]      flags flag bits
 * @param[in]      conf configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blasext
 */
int armas_ext_mvupdate_sym(
  DTYPE beta,
  armas_dense_t *A,
  DTYPE alpha,
  const armas_dense_t *x,
  int flags,
  armas_conf_t *conf)
{
  int nx = armas_size(x);

  if (armas_size(A) == 0 || nx == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();

  if (!armas_isvector(x)) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (A->cols != nx || A->rows != nx) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  armas_ext_mvupdate_trm_unsafe(beta, A, alpha, x, x, flags);
  return 0;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
