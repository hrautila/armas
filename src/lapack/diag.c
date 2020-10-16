
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Diagonal matrix

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mult_diag) && defined(armas_solve_diag)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_scale)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

/**
 * @brief Compute \f$ A = alpha*A*diag(D) \f$ or \f$ A = alpha*diag(D)*A \f$
 *
 * @param A
 *      Target matrix or vector.
 * @param alpha
 *      Scalar.
 * @param D
 *      Diagonal vector or square matrix
 * @param flags
 *      Flag bits, ARMAS_LEFT or ARMAS_RIGHT
 * @param conf
 *      Optional blocking configuration
 *
 * @retval 0 Success
 * @retval <0 Failure
 *
 * @ingroup lapack
 */
int armas_mult_diag(armas_dense_t * A, DTYPE alpha,
                      const armas_dense_t * D, int flags, armas_conf_t * conf)
{
    armas_dense_t c, d0;
    const armas_dense_t *d;
    int k;

    if (!conf)
        conf = armas_conf_default();

    d = D;
    if (!armas_isvector(D)) {
        armas_diag(&d0, D, 0);
        d = &d0;
    }

    if (armas_isvector(A)) {
        if (armas_size(d) != armas_size(A)) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        for (k = 0; k < armas_size(d); k++) {
            DTYPE aval =
                armas_get_at_unsafe(A, k) * armas_get_at_unsafe(d, k);
            armas_set_at_unsafe(A, k, alpha * aval);
        }
        return 0;
    }

    switch (flags & (ARMAS_LEFT | ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        if (armas_size(d) != A->cols) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        // scale columns; 
        for (k = 0; k < armas_size(d); k++) {
            armas_column(&c, A, k);
            armas_scale(&c, alpha * armas_get_at_unsafe(d, k), conf);
        }
        break;
    case ARMAS_LEFT:
    default:
        if (armas_size(d) != A->rows) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        // scale rows; for each column element-wise multiply of D element
        for (k = 0; k < armas_size(d); k++) {
            armas_row(&c, A, k);
            armas_scale(&c, alpha * armas_get_at_unsafe(d, k), conf);
        }
        break;
    }
    return 0;
}

/**
 * @brief Compute \f$ A = alpha*A*diag(D)^{-1} \f$ or \f$ A = alpha*diag(D)^{-1}*A \f$
 *
 * @param A
 *      Target matrix or vector.
 * @param alpha
 *      Scalar.
 * @param D
 *      Diagonal vector or square matrix
 * @param flags
 *      Flag bits, ARMAS_LEFT or ARMAS_RIGHT
 * @param conf
 *      Optional blocking configuration
 *
 * @retval 0 Success
 * @retval <0 Failure
 *
 * @ingroup lapack
 */
int armas_solve_diag(armas_dense_t * A, DTYPE alpha,
                       const armas_dense_t * D, int flags,
                       armas_conf_t * conf)
{
    armas_dense_t c, d0;
    const armas_dense_t *d;
    int k;

    if (!conf)
        conf = armas_conf_default();

    d = D;
    if (!armas_isvector(D)) {
        armas_diag(&d0, D, 0);
        d = &d0;
    }

    if (armas_isvector(A)) {
        if (armas_size(d) != armas_size(A)) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        for (k = 0; k < armas_size(d); k++) {
            DTYPE aval =
                armas_get_at_unsafe(A, k) / armas_get_at_unsafe(d, k);
            armas_set_at_unsafe(A, k, alpha * aval);
        }
        return 0;
    }

    switch (flags & (ARMAS_LEFT | ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        if (armas_size(d) != A->cols) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        // scale columns; 
        for (k = 0; k < armas_size(d); k++) {
            DTYPE aval = alpha * armas_get_at_unsafe(d, k);
            armas_column(&c, A, k);
            armas_scale(&c, ONE/aval, conf);
        }
        break;
    case ARMAS_LEFT:
    default:
        if (armas_size(d) != A->rows) {
            conf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        if (armas_size(A) == 0)
            return 0;
        // scale rows; for each column element-wise multiply of D element
        for (k = 0; k < armas_size(d); k++) {
            DTYPE aval = alpha * armas_get_at_unsafe(d, k);
            armas_row(&c, A, k);
            armas_scale(&c, ONE/aval, conf);
        }
        break;
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
