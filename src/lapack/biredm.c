
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Bidiagonal reduction

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bdmult) && defined(armas_x_bdmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_qrmult_w) && defined(armas_x_lqmult_w)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


/*
 * Multiply and replace C with product of C and Q or P where Q and P are real orthogonal
 * matrices defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)   and   P = G(1) G(2). . . G(k)
 *
 * as returned by bdreduce().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     Bidiagonal reduction as returned by bdreduce() where the lower trapezoidal
 *        part, on and below first subdiagonal, holds the product Q. The upper
 *        trapezoidal part holds the product P.
 *
 *  tau   The scalar factors of the elementary reflectors. If flag MULTQ is set then holds
 *        scalar factors for Q. If flag MULTP is set then holds scalar factors for P.
 *        Expected to be column vector is size min(M(A), N(A)).
 *
 *  bits  Indicators, valid bits LEFT, RIGHT, TRANS, MULTQ, MULTP
 *
 *
 * Additional information
 *
 *   Order N(Q) of orthogonal matrix Q is M(A) if M >= N and M(A)-1 if M < N
 *   The order N(P) of the orthogonal  matrix P is N(A)-1 if M >=N and N(A) if M < N. 
 *
 *        flags              result
 *        ------------------------------------------
 *        MULTQ,LEFT         C = Q*C     m(A) == m(C)
 *        MULTQ,TRANS,LEFT   C = Q.T*C   m(A) == m(C)
 *        MULTQ,RIGHT        C = C*Q     n(C) == m(A)
 *        MULTQ,TRANS,RIGHT  C = C*Q.T   n(C) == m(A)
 *        MULTP,LEFT         C = P*C     n(A) == m(C)
 *        MULTP,TRANS,LEFT   C = P.T*C   n(A) == m(C)
 *        MULTP,RIGHT        C = C*P     n(C) == n(A)
 *        MULTP,TRANS,RIGHT  C = C*P.T   n(C) == n(A)
 *
 * \ingroup lapack
 */
int armas_x_bdmult(armas_x_dense_t * C,
                   const armas_x_dense_t * A,
                   const armas_x_dense_t * tau, int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_bdmult_w(C, A, tau, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_x_bdmult_w(C, A, tau, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}

int armas_x_bdmult_w(armas_x_dense_t * C,
                     const armas_x_dense_t * A,
                     const armas_x_dense_t * tau,
                     int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t Qh, Ch, Ph, tauh;
    int err;

    if (!conf)
        conf = armas_conf_default();

    if (!C) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    if (wb && wb->bytes == 0) {
        // Depending on LEFT,RIGHT flags and m(C), n(C) the workspace
        // size that is needed for unblocked computation is max(m(C), n(C))
        // elements
        armas_wbuf_t w0 = ARMAS_WBNULL;
        if (armas_x_qrmult_w(C, A, tau, flags, &w0, conf) < 0)
            return -1;
        if (armas_x_lqmult_w(C, A, tau, flags, wb, conf) < 0)
            return -1;
        size_t wsmin = (C->rows > C->cols ? C->rows : C->cols) * sizeof(DTYPE);
        if (w0.bytes > wb->bytes)
            wb->bytes = w0.bytes;
        if (wb->bytes < wsmin)
            wb->bytes = wsmin;
        return 0;
    }
    // default to multiplication from left
    if (!(flags & (ARMAS_LEFT | ARMAS_RIGHT)))
        flags |= ARMAS_LEFT;

    // NOTE: sizes are checked in QR/LQ functions.
    if (flags & ARMAS_MULTP) {
        flags = flags & ARMAS_TRANS ? flags ^ ARMAS_TRANS : flags | ARMAS_TRANS;
    }

    if (A->rows > A->cols || (A->rows == A->cols && !(flags & ARMAS_LOWER))) {
        // M >= N
        switch (flags & (ARMAS_MULTQ | ARMAS_MULTP)) {
        case ARMAS_MULTQ:
            armas_x_submatrix(&tauh, tau, 0, 0, A->cols, 1);
            err = armas_x_qrmult_w(C, A, &tauh, flags, wb, conf);
            break;
        case ARMAS_MULTP:
            armas_x_submatrix(&Ph, A, 0, 1, A->cols - 1, A->cols - 1);
            armas_x_submatrix(&tauh, tau, 0, 0, A->cols - 1, 1);
            if (flags & ARMAS_RIGHT) {
                armas_x_submatrix(&Ch, C, 0, 1, C->rows, C->cols - 1);
            } else {
                armas_x_submatrix(&Ch, C, 1, 0, C->rows - 1, C->cols);
            }
            err = armas_x_lqmult_w(&Ch, &Ph, &tauh, flags, wb, conf);
            break;
        default:
            conf->error = ARMAS_EINVAL;
            return -1;
        }
    } else {
        // M < N
        switch (flags & (ARMAS_MULTQ | ARMAS_MULTP)) {
        case ARMAS_MULTQ:
            armas_x_submatrix(&Qh, A, 1, 0, A->rows - 1, A->rows - 1);
            armas_x_submatrix(&tauh, tau, 0, 0, A->rows - 1, 1);
            if (flags & ARMAS_RIGHT) {
                armas_x_submatrix(&Ch, C, 0, 1, C->rows, C->cols - 1);
            } else {
                armas_x_submatrix(&Ch, C, 1, 0, C->rows - 1, C->cols);
            }
            err = armas_x_qrmult_w(&Ch, &Qh, &tauh, flags, wb, conf);
            break;
        case ARMAS_MULTP:
            armas_x_submatrix(&tauh, tau, 0, 0, A->rows, 1);
            err = armas_x_lqmult_w(C, A, &tauh, flags, wb, conf);
            break;
        default:
            conf->error = ARMAS_EINVAL;
            return -1;
        }
    }
    return err;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
