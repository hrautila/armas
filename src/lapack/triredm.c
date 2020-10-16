
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_trdmult) && defined(armas_trdmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_qrmult_w) && defined(armas_qlmult_w)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

/**
 * @brief Multiply matrix C with orthogonal matrix Q.
 * @see armas_trdmult_w
 * @ingroup lapack
 */
int armas_trdmult(armas_dense_t * C,
                    const armas_dense_t * A,
                    const armas_dense_t * tau, int flags, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_trdmult_w(C, A, tau, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_trdmult_w(C, A, tau, flags, wbs, conf);
    armas_wrelease(&wb);
    return stat;
}

/**
 * @brief Multiply matrix C with orthogonal matrix Q.
 *
 * @param[in,out] C
 *    On entry matrix C. On exit product of C and orthogonal matrix Q.
 * @param[in] A
 *    Orthogonal matrix C as elementary reflectors saved in upper (lower) triangular
 *    part of A. See trdreduce().
 * @param[in] tau
 *    Scalar coeffients of elementary reflectors.
 * @param[in] flags
 *    Indicator flags, combination of *ARMAS_LOWER*, *ARMAS_UPPER*, *ARMAS_LEFT*,
 *    *ARMAS_RIGHT* and *ARMAS_TRANS*.
 * @param[out] wb
 *    Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *    immediately.
 * @param[in] conf
 *    Configuration block.
 *
 * @retval 0  Sucess
 * @retval <0 Error
 * @ingroup lapack
 */
int armas_trdmult_w(armas_dense_t * C,
                      const armas_dense_t * A,
                      const armas_dense_t * tau,
                      int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t Ch, Qh, tauh;
    int err = 0;

    if (!conf)
        conf = armas_conf_default();

    if (!C) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    if (wb && wb->bytes == 0) {
        if (flags & ARMAS_UPPER)
            err = armas_qlmult_w(C, A, tau, flags, wb, conf);
        else
            err = armas_qrmult_w(C, A, tau, flags, wb, conf);
        return err;
    }

    if (!A || !tau || armas_size(tau) != A->cols) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    int P = (flags & ARMAS_RIGHT) != 0 ? C->cols : C->rows;
    if (P != A->rows || A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (!wb) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (flags & ARMAS_UPPER) {
        if (flags & ARMAS_RIGHT) {
            armas_submatrix(&Ch, C, 0, 0, C->rows, C->cols - 1);
        } else {
            armas_submatrix(&Ch, C, 0, 0, C->rows - 1, C->cols);
        }
        armas_submatrix(&Qh, A, 0, 1, A->rows - 1, A->rows - 1);
        armas_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
        err = armas_qlmult_w(&Ch, &Qh, &tauh, flags, wb, conf);
    } else {
        if (flags & ARMAS_RIGHT) {
            armas_submatrix(&Ch, C, 0, 1, C->rows, C->cols - 1);
        } else {
            armas_submatrix(&Ch, C, 1, 0, C->rows - 1, C->cols);
        }
        armas_submatrix(&Qh, A, 1, 0, A->rows - 1, A->rows - 1);
        armas_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
        err = armas_qrmult_w(&Ch, &Qh, &tauh, flags, wb, conf);
    }
    return err;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
