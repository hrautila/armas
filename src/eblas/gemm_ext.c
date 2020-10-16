
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_mult)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_ext_mult_kernel)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

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
 * @retval <0 Failed, conf.error set to actual error code.
 *
 * @ingroup blasext
 */
int armas_ext_mult(
    DTYPE beta,
    armas_dense_t *C,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_size(A) == 0 || armas_size(B) == 0 || armas_size(C) == 0)
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
        return -ARMAS_ESIZE;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    cache_t cache;
    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    armas_cache_setup(&cache, &cbuf, 3, sizeof(DTYPE));

    armas_ext_mult_kernel(beta, C, alpha, A, B, flags, &cache);
    armas_cbuf_release(&cbuf);

    return 0;
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
