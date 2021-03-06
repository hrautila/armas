
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mult) 
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_mult_kernel)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "accel.h"

// ----------------------------------------------------------------------------------------
// exported public functions


/**
 * @brief General matrix-matrix multiplication
 *
 * Computes
 *   - \f$ C = alpha \times A B + beta \times C \f$
 *   - \f$ C = alpha \times A^T B + beta \times C \f$  if _ARMAS_TRANSA_ is set
 *   - \f$ C = alpha \times A B^T + beta \times C \f$  if _ARMAS_TRANSB_ is set
 *   - \f$ C = alpha \times A^T B^T + beta \times C \f$ if _ARMAS_TRANSA_ and _ARMAS_TRANSB_ are set
 *
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
 * @ingroup blas
 */
int armas_mult(
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

    require(A->step >= A->rows && B->step >= B->rows && C->step >= C->rows);

    if (!conf)
        conf = armas_conf_default();

    // check consistency
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
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (CONFIG_ACCELERATORS) {
        struct armas_ac_blas3 args;
        armas_ac_set_blas3_args(&args, beta, C, alpha, A, B, flags);
        int rc = armas_ac_dispatch(conf->accel, ARMAS_AC_GEMM, &args, conf);
        if (rc != -ARMAS_EIMP)
            return rc;
        /* fallthru to local version. */
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;

    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, &cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_mult_kernel(beta, C, alpha, A, B, flags, &cache);
    armas_cbuf_release(&cbuf);

    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
