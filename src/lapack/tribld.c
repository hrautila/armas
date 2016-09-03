
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Orthogonal matrix Q of tridiagonally reduced matrix

#include <stdio.h>
#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdbuild) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_qrbuild) && defined(armas_x_qlbuild)
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

/**
 * @brief Generate orthogonal matrix Q for tridiagonally reduced matrix.
 *
 * @param A [in,out]
 *    On entry tridiagonal reduction as returned by trdreduce(). On exit
 *    the orthogonal matrix Q.
 * @param tau [in]
 *    Scalar coefficients of the elementary reflectors.
 * @param W   [in]
 *    Workspace
 * @param K  [in]
 *    Number of elementary reflector that define the Q matrix, n(A) > K > 0
 * @param flags [in]
 *    Flag bits, either upper tridiagonal (ARMAS_UPPER) or lower tridiagonal (ARMAS_LOWER)
 * @param conf [in]
 *    Optional blocking configuration
 *
 * @return
 *    0 on success, -1 on failure and sets conf.error value.
 */
int armas_x_trdbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                     int K, int flags, armas_conf_t *conf)
{
    armas_x_dense_t Qh, tauh, d, s;
    int j, err = 0;

    if (!conf) 
        conf = armas_conf_default();

    if (armas_x_size(A) == 0)
        return 0;

  // default to lower triangular if uplo not defined
    if ((flags & (ARMAS_LOWER|ARMAS_UPPER)) == 0) 
        flags |= ARMAS_LOWER;

    if (K > A->rows - 1)
        K = A->rows - 1;

    switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
    case ARMAS_LOWER:
        // shift Q matrix embedded in A right and fill first column to unit vector
        for (j = A->rows-1; j > 0; j--) {
            armas_x_submatrix(&s, A, j, j-1, A->rows-j, 1);
            armas_x_submatrix(&d, A, j, j,   A->rows-j, 1);
            armas_x_copy(&d, &s, conf);
            armas_x_set(A, 0, j, __ZERO);
        }
        // zero first row
        armas_x_column(&d, A, 0);
        armas_x_scale(&d, __ZERO, conf);
        armas_x_set(&d, 0, 0, __ONE);

        armas_x_submatrix(&Qh, A, 1, 1, A->rows-1, A->rows-1);
        armas_x_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
        err = armas_x_qrbuild(&Qh, &tauh, W, K, conf);
        break;

    case ARMAS_UPPER:
        // shift Q matrix embedded in A left and fill first column to unit vector
        for (j = 1; j < A->rows; j++) {
            armas_x_submatrix(&s, A, 0, j,   j, 1);
            armas_x_submatrix(&d, A, 0, j-1, j, 1);
            armas_x_copy(&d, &s, conf);
            armas_x_set(A, -1, j-1, __ZERO);
        }
        // zero first row
        armas_x_column(&d, A, A->rows-1);
        armas_x_scale(&d, __ZERO, conf);
        armas_x_set(&d, -1, 0, __ONE);

        armas_x_submatrix(&Qh, A, 0, 0, A->rows-1, A->rows-1);
        armas_x_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
        err = armas_x_qlbuild(&Qh, &tauh, W, K, conf);
        break;

    default:
        break;
    }
    return err;
}

int armas_x_trdbuild_work(armas_x_dense_t *A, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();
    // QR, QL workspace requirements same
    return armas_x_qrbuild_work(A, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

