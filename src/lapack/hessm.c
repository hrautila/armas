
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Hessenberg reduction
#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_hessmult)  && defined(armas_x_hessmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_qrmult_w)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

/**
 * @brief Multiply with the orthogonal matrix Q of Hessenberg reduction.
 *
 * @see armas_x_hessmult_w
 * @ingroup lapack
 */
int armas_x_hessmult(armas_x_dense_t * C,
                     const armas_x_dense_t * A,
                     const armas_x_dense_t * tau,
                     int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_hessmult_w(C, A, tau, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_x_hessmult_w(C, A, tau, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Multiply with the orthogonal matrix Q of Hessenberg reduction
 *
 * Multiply and replace C with product of C and Q where Q is a real orthogonal matrix
 * defined as the product of K = n(A) elementary reflectors.
 *
 *    \f$ Q = H_1  H_2 . . . H_K \f$
 *
 * @param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then N-by-M matrix
 *     On exit C is overwritten by \f$ Q C \f$ or \f$ Q^T C \f$. If bit *ARMAS_RIGHT* is
 *     set then C is overwritten by \f$ CQ \f$ or \f$ C Q^T \f$
 *
 * @param[in] A
 *      Hessenberg reduction as returned by hessreduce() where the lower trapezoidal
 *      part, on and below first subdiagonal, holds the elementary reflectors.
 *
 * @param[in] tau
 *     The scalar factors of the elementary reflectors. A column vector.
 *
 * @param[out] wb
 *     Workspace. If wb.size is zero then required workspace size is computed and
 *     returned immediately.
 *
 * @param[in] flags
 *    Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT* and *ARMAS_TRANS*
 *
 * @param[in,out] conf
 *    Blocking configuration. Field conf.lb defines block size. If it is zero
 *    unblocked invocation is assumed.
 *
 *```md
 *        flags        result
 *        -------------------------------------
 *        LEFT         C = Q*C     n(A) == m(C)
 *        RIGHT        C = C*Q     n(C) == m(A)
 *        TRANS|LEFT   C = Q.T*C   n(A) == m(C)
 *        TRANS|RIGHT  C = C*Q.T   n(C) == m(A)
 *```
 *
 * @ingroup lapack
 */
int armas_x_hessmult_w(armas_x_dense_t * C,
                       const armas_x_dense_t * A,
                       const armas_x_dense_t * tau,
                       int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t Qh, Ch, tauh;
    if (!conf)
        conf = armas_conf_default();

    if (!C) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    armas_x_submatrix(&Qh, A, 1, 0, A->rows - 1, A->cols - 1);
    armas_x_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
    if (flags & ARMAS_RIGHT) {
        armas_x_submatrix(&Ch, C, 0, 1, C->rows, C->cols - 1);
    } else {
        armas_x_submatrix(&Ch, C, 1, 0, C->rows - 1, C->cols);
    }

    if (wb && wb->bytes == 0) {
        return armas_x_qrmult_w(&Ch, __nil, __nil, flags, wb, conf);
    }
    return armas_x_qrmult_w(&Ch, &Qh, &tauh, flags, wb, conf);
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
