
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank-2k update

//! \cond
#include <stdio.h>
#include <string.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_update2_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_update_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

/**
 * @brief Symmetric matrix rank-2k update
 *
 * Computes
 * - \f$ C = beta \times C + alpha \times A B^T + alpha \times B A^T \f$
 * - \f$ C = beta \times C + alpha \times A^T B + alpha \times B^T A \f$ if *ARMAS_TRANSA* set
 *
 * Matrix C has elements stored in the  upper (lower) triangular part
 * if flag bit *ARMAS_UPPER* (*ARMAS_LOWER*) is set.
 * If matrix is upper (lower) then the strictly lower (upper) part is not referenced.
 *
 * @param[in] beta scalar constant
 * @param[in,out] C result matrix
 * @param[in] alpha scalar constant
 * @param[in] A first operand matrix
 * @param[in] B second operand matrix
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval 0  Operation succeeded
 * @retval <0 Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int armas_x_update2_sym(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    if (armas_x_size(C) == 0 || armas_x_size(A) == 0 || armas_x_size(B) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    int ok = A->rows == B->rows && A->cols == B->cols && C->rows == C->cols;
    switch (flags & ARMAS_TRANS) {
    case ARMAS_TRANS:
        ok = ok && C->rows == A->cols;
        break;
    default:
        ok = ok && C->rows == A->rows;
        break;
    }
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    // do it twice with triangular update
    if (flags & ARMAS_TRANS) {
        if (armas_x_update_trm(beta, C, alpha, A, B, ARMAS_TRANSA, conf) < 0)
            return -1;
        return armas_x_update_trm(beta, C, alpha, B, A, ARMAS_TRANSA, conf);
    }
    if (armas_x_update_trm(beta, C, alpha, A, B, ARMAS_TRANSB, conf) < 0)
        return -1;
    return armas_x_update_trm(beta, C, alpha, B, A, ARMAS_TRANSB, conf);
}
#else
#warning "Missing defines; no code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
