
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal matrix for bidiagonal reduction

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bdbuild) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_qrbuild) && defined(armas_x_lqbuild)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
//! \endcond

/**
 * \brief Generate orthogonal matrix Q or P
 *
 * Generate one of the orthogonal matrices  Q or \f$ P^T \f$ determined by bdreduce() when
 * reducing a real matrix A to bidiagonal form. Q and \f$ P^T \f$ are defined as products
 * elementary reflectors \f$ H_i \f$ or \f$ G_i \f$ respectively.
 *
 * Orthogonal matrix Q is generated if flag *ARMSA_WANTQ* is set. And matrix P respectively
 * of flag *ARMAS_WANTP* is set.
 *
 * \param[in,out] A
 *   On entry the bidiagonal reduction as returned by bdreduce(). On exit the requested
 *   orthogonal matrix defined as the product of K first elementary reflectors.
 * \param[in] tau
 *   Scalar coefficients of the elementary reflectors.
 * \param[out] W
 *   Workspace
 * \param[in] K
 *   Number elementary reflectors used to generate orthogonal matrix. \f$ 0 < K <= n(A) \f$
 * \param[in] flags
 *   Indicator flags, *ARMAS_WANTQ* or *ARMAS_WANTP*.
 * \param[in,out] conf
 *   Blocking configuration
 *
 * \retval 0 Success
 * \retval -1 fail, `conf.error` set to error code.
 *
 * \ingroup lapack
 */
int armas_x_bdbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                    int K, int flags, armas_conf_t *conf)
{
    armas_x_dense_t Qh, Ph, tauh, d, s;
    int j, err = 0;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(A) == 0)
        return 0;

    if (A->rows > A->cols || (A->rows == A->cols && !(flags & ARMAS_LOWER))) {
        switch (flags & (ARMAS_WANTQ|ARMAS_WANTP)) {
        case ARMAS_WANTQ:
            armas_x_submatrix(&tauh, tau, 0, 0, A->cols, 1);
            err = armas_x_qrbuild(A, &tauh, W, K, conf);
            break;
        case ARMAS_WANTP:
            // shift P matrix embedded in A down and fill first column and
            // row to unit vectors
            for (j = A->cols-1; j > 0; j--) {
                armas_x_submatrix(&s, A, j-1, j, 1, A->cols-j);
                armas_x_submatrix(&d, A, j,   j, 1, A->cols-j);
                armas_x_copy(&d, &s, conf);
                armas_x_set(A, j, 0, __ZERO);
            }
            // zero first row
            armas_x_row(&d, A, 0);
            armas_x_scale(&d, __ZERO, conf);
            armas_x_set(&d, 0, 0, __ONE);

            armas_x_submatrix(&Ph, A, 1, 1, A->cols-1, A->cols-1);
            armas_x_submatrix(&tauh, tau, 0, 0, A->cols-1, 1);
            if (K > A->cols-1 || K < 0)
                K = A->cols - 1;
            err = armas_x_lqbuild(&Ph, &tauh, W, K, conf);
            break;
        default:
            break;
        }
    } else {
        // here A->rows < A-cols || (A->rows == A->cols && flags&ARMAS_LOWER)
        switch (flags & (ARMAS_WANTQ|ARMAS_WANTP)) {
        case ARMAS_WANTQ:
            // shift Q matrix embedded in A right and fill first column and
            // row to unit vectors
            for (j = A->rows-1; j > 0; j--) {
                armas_x_submatrix(&s, A, j, j-1, A->rows-j, 1);
                armas_x_submatrix(&d, A, j, j,   A->rows-j, 1);
                armas_x_copy(&d, &s, conf);
                armas_x_set(A, 0, j, __ZERO);
            }
            // zero first column
            armas_x_column(&d, A, 0);
            armas_x_scale(&d, __ZERO, conf);
            armas_x_set(&d, 0, 0, __ONE);

            armas_x_submatrix(&Qh, A, 1, 1, A->rows-1, A->rows-1);
            armas_x_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
            if (K > A->rows-1 || K < 0)
                K = A->rows - 1;
            err = armas_x_qrbuild(&Qh, &tauh, W, K, conf);
            break;
        case ARMAS_WANTP:
            armas_x_submatrix(&tauh, tau, 0, 0, A->rows, 1);
            err = armas_x_lqbuild(A, &tauh, W, K, conf);
            break;
        default:
            break;
        }
    }
    return err;
}

//! \brief Workspace size for bdbuild().
//! \ingroup lapack
int armas_x_bdbuild_work(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  if (flags & ARMAS_WANTP) {
      return armas_x_lqbuild_work(A, conf);
  }
  return armas_x_qrbuild_work(A, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

