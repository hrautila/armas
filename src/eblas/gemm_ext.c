
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

//! \cond
#include <stdlib.h>
#include <stdint.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mult)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_ext_mult_kernel)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

// ----------------------------------------------------------------------------------------
// exported public functions


/**
 * @brief General matrix-matrix multiplication in extended precision.
 *
 * Computes
 *   - \f$ C = alpha \times A B + beta \times C \f$
 *   - \f$ C = alpha \times A^T B + beta \times C \f$  if _ARMAS_TRANSA_ is set
 *   - \f$ C = alpha \times A B^T + beta \times C \f$  if _ARMAS_TRANSB_ is set
 *   - \f$ C = alpha \times A^T B^T + beta \times C \f$ if _ARMAS_TRANSA_ and _ARMAS_TRANSB_ are set
 * @param[in,out] C result matrix
 * @param[in] A first operand matrix
 * @param[in] B second operand matrix
 * @param[in] alpha scalar constant
 * @param[in] beta scalar constant
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval 0 Operation succeeded
 * @retval -1 Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int armas_x_ext_mult(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_x_size(A) == 0 || armas_x_size(B) == 0 || armas_x_size(C) == 0)
        return  0;

    if (!conf)
        conf = armas_conf_default();

    switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB|ARMAS_CTRANSA|ARMAS_CTRANSB)) {
    case ARMAS_TRANSA|ARMAS_TRANSB:
    case ARMAS_TRANSA|ARMAS_CTRANSB:
    case ARMAS_CTRANSA|ARMAS_CTRANSB:
    case ARMAS_CTRANSA|ARMAS_TRANSB:
        ok = C->rows == A->cols && C->cols == B->rows && A->rows == B->cols;
        break;
    case ARMAS_TRANSA:
    case ARMAS_CTRANSA:
        ok = C->rows == A->cols && C->cols == B->cols && A->rows == B->rows;
        break;
    case ARMAS_TRANSB:
    case ARMAS_CTRANSB:
        ok = C->rows == A->rows && C->cols == B->rows && A->cols == B->cols;
        break;
    default:
        ok = C->rows == A->rows && C->cols == B->cols && A->cols == B->rows;
        break;
    }
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    cache_t cache;
    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -1;
    }
    armas_cache_setup(&cache, &cbuf, 3, sizeof(DTYPE));

    armas_x_ext_mult_kernel(beta, C, alpha, A, B, flags, &cache);
    armas_cbuf_release(&cbuf);

    return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
