
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
#if defined(armas_x_bdbuild) && defined(armas_x_bdbuild_w) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_qrbuild_w) && defined(armas_x_lqbuild_w)
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
int armas_x_bdbuild(armas_x_dense_t *A,
                    const armas_x_dense_t *tau,
                    int K,
                    int flags,
                    armas_conf_t *conf)
{
    int err;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_bdbuild_w(A, tau, K, flags, &wb, conf) < 0)
        return -1;
    
    if (!armas_walloc(&wb, wb.bytes)) {
        conf->error = ARMAS_EMEMORY;
        return -1;
    }
    
    err = armas_x_bdbuild_w(A, tau, K, flags, &wb, conf);
    armas_wrelease(&wb);
    return err;
}


int armas_x_bdbuild_w(armas_x_dense_t *A,
                      const armas_x_dense_t *tau, 
                      int K,
                      int flags,
                      armas_wbuf_t *wb,
                      armas_conf_t *conf)
{
    armas_x_dense_t Qh, Ph, tauh, d, s;   
    int j, err = 0;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;       
    }

    if (wb && wb->bytes == 0) {
        // if M >= N then size is f(N), if M < N then size is f(M)
        if (A->rows >= A->cols) {
            if (conf->lb > 0 && A->cols > conf->lb)
                wb->bytes = (A->cols * conf->lb) * sizeof(DTYPE);
            else
                wb->bytes = A->cols * sizeof(DTYPE);
        }
        else {
            if (conf->lb > 0 && A->rows > conf->lb)
                wb->bytes = (A->rows * conf->lb) * sizeof(DTYPE);
            else
                wb->bytes = A->rows * sizeof(DTYPE);
        }
        return 0;
    }

    if (armas_x_size(A) == 0)
        return 0;

    if (A->rows > A->cols || (A->rows == A->cols && !(flags & ARMAS_LOWER))) {
        switch (flags & (ARMAS_WANTQ|ARMAS_WANTP)) {
        case ARMAS_WANTQ:
            armas_x_submatrix(&tauh, tau, 0, 0, A->cols, 1);
            err = armas_x_qrbuild_w(A, &tauh, K, wb, conf);
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
            err = armas_x_lqbuild_w(&Ph, &tauh, K, wb, conf);
            break;
        default:
            conf->error = ARMAS_EINVAL;
            err = -1;
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
            err = armas_x_qrbuild_w(&Qh, &tauh, K, wb, conf);
            break;
        case ARMAS_WANTP:
            armas_x_submatrix(&tauh, tau, 0, 0, A->rows, 1);
            err = armas_x_lqbuild_w(A, &tauh, K, wb, conf);
            break;
        default:
            conf->error = ARMAS_EINVAL;
            err = -1;
            break;
        }
    }
    return err;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

