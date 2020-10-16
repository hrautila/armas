
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Orthogonal matrix Q of tridiagonally reduced matrix

#include <stdio.h>
#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_trdbuild) && defined(armas_trdbuild_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_qrbuild_w) && defined(armas_qlbuild_w)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

/**
 * @brief Generate orthogonal matrix Q for tridiagonally reduced matrix.
 * @see armas_trdbuild_w
 * @ingroup lapack
 */
int armas_trdbuild(armas_dense_t * A,
                     const armas_dense_t * tau,
                     int K, int flags, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_trdbuild_w(A, tau, K, flags, &wb, conf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_trdbuild_w(A, tau, K, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Generate orthogonal matrix Q for tridiagonally reduced matrix.
 *
 * @param [in,out] A
 *    On entry tridiagonal reduction as returned by trdreduce(). On exit
 *    the orthogonal matrix Q.
 * @param [in] tau
 *    Scalar coefficients of the elementary reflectors.
 * @param [in] K
 *    Number of elementary reflector that define the Q matrix, n(A) > K > 0
 * @param [in] flags
 *    Flag bits, either upper tridiagonal (ARMAS_UPPER) or lower tridiagonal (ARMAS_LOWER)
 * @param [in,out] wb
 *    Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *    immediately.
 * @param [in, out] conf
 *    Configuration block.
 *
 * @retval  0  Success
 * @retval <0  Failure, sets conf.error value.
 * @ingroup lapack
 */
int armas_trdbuild_w(armas_dense_t * A,
                       const armas_dense_t * tau,
                       int K, int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t Qh, tauh, d, s;
    int j, err = 0;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (wb && wb->bytes == 0) {
        if (flags & ARMAS_UPPER)
            err = armas_qlbuild_w(A, tau, K, wb, conf);
        else
            err = armas_qrbuild_w(A, tau, K, wb, conf);
        return err;
    }

    if (armas_size(A) == 0)
        return 0;

    if (!tau || armas_size(tau) != A->cols) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    if (A->cols != A->rows) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (K > A->rows - 1)
        K = A->rows - 1;

    switch (flags & (ARMAS_LOWER | ARMAS_UPPER)) {
    case ARMAS_UPPER:
        // shift Q matrix embedded in A left and fill last column to unit vector
        for (j = 1; j < A->cols; j++) {
            armas_submatrix(&s, A, 0, j, j, 1);
            armas_submatrix(&d, A, 0, j - 1, j, 1);
            armas_copy(&d, &s, conf);
            armas_set(A, A->rows - 1, j - 1, ZERO);
        }
        // zero last column
        armas_column(&d, A, A->cols - 1);
        armas_scale(&d, ZERO, conf);
        armas_set(&d, d.rows - 1, 0, ONE);

        armas_submatrix(&Qh, A, 0, 0, A->rows - 1, A->rows - 1);
        armas_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
        err = armas_qlbuild_w(&Qh, &tauh, K, wb, conf);
        break;

    case ARMAS_LOWER:
    default:
        // shift Q matrix embedded in A right and fill first column to unit vector
        for (j = A->cols - 1; j > 0; j--) {
            armas_submatrix(&s, A, j, j - 1, A->rows - j, 1);
            armas_submatrix(&d, A, j, j, A->rows - j, 1);
            armas_copy(&d, &s, conf);
            armas_set(A, 0, j, ZERO);
        }
        // zero first row
        armas_column(&d, A, 0);
        armas_scale(&d, ZERO, conf);
        armas_set(&d, 0, 0, ONE);

        armas_submatrix(&Qh, A, 1, 1, A->rows - 1, A->rows - 1);
        armas_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
        err = armas_qrbuild_w(&Qh, &tauh, K, wb, conf);
        break;
    }
    return err;
}
#else
#Warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
