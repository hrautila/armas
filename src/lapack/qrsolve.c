
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Least squares or minimum norm solution

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qrsolve) && defined(armas_qrsolve_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_qrmult)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

/**
 * @brief Solve a system of linear equations \f$ AX = B \f$
 *
 * @see armas_qrsolve_w
 * @ingroup lapack
 */
int armas_qrsolve(armas_dense_t * B,
                    const armas_dense_t * A,
                    const armas_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_qrsolve_w(B, A, tau, flags, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_qrsolve_w(B, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Solve a system of linear equations \f$ AX = B \f$
 *
 * Solve a system of linear equations AX = B with general M-by-N
 * matrix A using the QR factorization computed by armas_qrfactor_w().
 *
 * If flag *ARMAS_TRANS* is set
 * find the minimum norm solution of an overdetermined system \f$ A^TX = B \f$
 * i.e \f$ min ||X|| s.t A^T X = B \f$
 *
 * Otherwise find the least squares solution of an overdetermined system, i.e.,
 *   solve the least squares problem: \f$ min || B - A*X || \f$
 *
 * @param[in,out] B
 *   On entry, the right hand side N-by-P matrix B.  On exit, the solution matrix X.
 *
 * @param[in] A
 *   The elements on and above the diagonal contain the min(M,N)-by-N upper
 *   trapezoidal matrix R. The elements below the diagonal with the vector 'tau',
 *   represent the ortogonal matrix Q as product of elementary reflectors.
 *   Matrix A and vector tau are as returned by armas_qrfactor()
 *
 * @param[in] tau
 *   The vector of N scalar coefficients that together with trilu(A) define
 *   the ortogonal matrix Q as \f$ Q = H(1)H(2)...H(N) \f$
 *
 * @param[in] flags
 *   Indicator flags
 *
 * @param wb
 *   Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *   immediately.
 *
 * @param[in,out] conf
 *   Configuration options.
 *
 * Compatible with lapack.GELS (the m >= n part)
 * @ingroup lapack
 */
int armas_qrsolve_w(armas_dense_t * B,
                      const armas_dense_t * A,
                      const armas_dense_t * tau,
                      int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t R, BT, BB;
    size_t wsmin, wsz = 0;

    if (!conf)
        conf = armas_conf_default();

    if (!B || !A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    if (wb && wb->bytes == 0) {
        return armas_qrmult_w(B, A, tau, ARMAS_LEFT, wb, conf);
    }

    if (B->rows != A->rows) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    wsmin = B->cols * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    armas_submatrix(&R, A, 0, 0, A->cols, A->cols);
    armas_submatrix(&BT, B, 0, 0, A->cols, B->cols);

    if (flags & ARMAS_TRANS) {
        // solve ovedetermined system A.T*X = B

        // B' = R.-1*B
        armas_solve_trm(&BT, ONE, &R,
                          ARMAS_LEFT | ARMAS_UPPER | ARMAS_TRANSA, conf);

        // clear bottom part of B
        armas_submatrix(&BB, B, A->cols, 0, -1, -1);
        armas_mscale(&BB, 0.0, 0, conf);

        // X = Q*B
        armas_qrmult_w(B, A, tau, ARMAS_LEFT, wb, conf);
    } else {
        // solve least square problem min || A*X - B ||

        // B' = Q.T*B
        armas_qrmult_w (B, A, tau, ARMAS_LEFT | ARMAS_TRANS, wb, conf);

        // X = R.-1*B'
        armas_solve_trm (&BT, ONE, &R, ARMAS_LEFT | ARMAS_UPPER, conf);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
