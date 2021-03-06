
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Solve system of linear inequalities

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_lqsolve) && defined(armas_lqsolve_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_lqmult)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
//! \endcond
static inline
int ws_lqsolve(int M, int N, int lb)
{
    return lb == 0 ? M : lb * (M + lb);
}

/**
 * @brief Solve a system of linear equations
 *
 * @see armas_lqsolve_w
 * @ingroup lapack
 */
int armas_lqsolve(armas_dense_t * B,
                    const armas_dense_t * A,
                    const armas_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_lqsolve_w(B, A, tau, flags, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_lqsolve_w(B, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Solve a system of linear equations
 *
 * Solve a system of linear equations A*X = B with general M-by-N (M < N)
 * matrix A using the LQ factorization computed by armas_lqfactor_w().
 *
 * If *ARMAS_TRANS is set:
 *   find the minimum norm solution of an overdetermined system \f$ A^T X = B \f$
 *   i.e \f$ min ||X|| s.t A^T X = B \f$
 *
 * Otherwise:
 *   find the least squares solution of an overdetermined system, i.e.,
 *   solve the least squares problem: \f$ min || B - A X || \f$
 *
 * @param[in,out] B
 *     On entry, the right hand side N-by-P matrix B.
 *     On exit, the solution matrix X.
 *
 * @param[in] A
 *     The elements on and below the diagonal contain the min(M,N)-by-N lower
 *     trapezoidal matrix `L`. The elements right the diagonal with the vector `tau`, 
 *     represent the ortogonal matrix Q as product of elementary reflectors.
 *     Matrix `A` and `tau` are as returned by lqfactor()
 *
 * @param[in] tau
 *   The vector of N scalar coefficients that together with triuu(A) define
 *   the ortogonal matrix Q as \f$ Q = H_1 H_2...H_{N-1} \f$
 *
 * @param[in] flags 
 *    Indicator flags, *ARMAS_TRANS*
 *
 * @param[out] wb
 *    Workspace. If *wb.bytes* is zero then the size of required workspace is
 *    computed and returned immediately.
 *
 * @param[in,out] conf
 *     Configuration options.
 *
 * @retval  0  Success
 * @retval <0  Error, last error in conf.error.
 *
 * Compatible with lapack.GELS (the m >= n part)
 * @ingroup lapack
 */
int armas_lqsolve_w(armas_dense_t * B,
                      const armas_dense_t * A,
                      const armas_dense_t * tau,
                      int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t L, BL, BB;
    size_t wsmin, wsz = 0;

    if (!conf)
        conf = armas_conf_default();

    if (!B || !A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (wb && wb->bytes == 0) {
        return armas_lqmult_w(B, A, tau, ARMAS_LEFT, wb, conf);
    }

    if (B->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    wsmin = B->rows * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    armas_submatrix(&L, A, 0, 0, A->rows, A->rows);
    armas_submatrix(&BL, B, 0, 0, A->rows, B->cols);

    if (flags & ARMAS_TRANS) {
        // solve least square problem min || A.T*X - B ||

        // B' = Q.T*B
        armas_lqmult_w(B, A, tau, ARMAS_LEFT, wb, conf);

        // X = L.-1*B'
        armas_solve_trm(&BL, ONE, &L,
                          ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSA, conf);

    } else {
        // solve underdetermined system A*X = B
        // B' = L.-1*B
        armas_solve_trm(&BL, ONE, &L, ARMAS_LEFT|ARMAS_LOWER, conf);

        // clear bottom part of B
        armas_submatrix(&BB, B, A->rows, 0, -1, -1);
        armas_mscale(&BB, ZERO, 0, conf);

        // X = Q.T*B'
        armas_lqmult_w(B, A, tau, ARMAS_LEFT|ARMAS_TRANS, wb, conf);
    }
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
