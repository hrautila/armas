
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank-k update

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_update_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_update_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"

/**
 * @brief Symmetric matrix rank-k update
 *
 * Computes
 *    - \f$ C = beta \times C + alpha \times A A^T \f$
 *    - \f$ C = beta \times C + alpha \times A^T A \f$  if *ARMAS_TRANS*
 *
 * Matrix C is upper (lower) triangular if flag bit *ARMAS_UPPER* (*ARMAS_LOWER*)
 * is set. If matrix is upper (lower) then
 * the strictly lower (upper) part is not referenced.
 *
 * @param[in] beta scalar constant
 * @param[in,out] C symmetric result matrix
 * @param[in] alpha scalar constant
 * @param[in] A first operand matrix
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval 0  Operation succeeded
 * @retval <0 Failed, conf.error set to actual error code.
 *
 * @ingroup blas
 */
int armas_update_sym(
    DTYPE beta,
    armas_dense_t *C,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_size(A) == 0 || armas_size(C) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    switch (flags & ARMAS_TRANS) {
    case ARMAS_TRANS:
        ok = C->rows == A->cols && C->rows == C->cols;
        break;
    default:
        ok = C->rows == A->rows && C->rows == C->cols;
        break;
    }
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    int uflags = flags & (ARMAS_UPPER | ARMAS_LOWER);
    uflags |= (flags & ARMAS_TRANS ? ARMAS_TRANSA : ARMAS_TRANSB);

    return armas_update_trm(beta, C, alpha, A, A, uflags, conf);
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
