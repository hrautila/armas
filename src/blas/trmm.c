
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Triangular matrix multiplication

//! \cond
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mult_trm) 
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_trmm_unb) && \
  defined(armas_x_trmm_recursive) && \
  defined(armas_x_trmm_blk)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#ifdef CONFIG_ACCELERATORS
#include "accel.h"
#endif
//! \endcond


/**
 * @brief Triangular matrix-matrix multiply
 *
 * If flag bit *ARMAS_LEFT* is set then computes 
 *    - \f$ B = alpha \times A B \f$
 *    - \f$ B = alpha \times A^T B  \f$ if *ARMAS_TRANS* set
 *
 * If flag bit *ARMAS_RIGHT* is set then computes
 *    - \f$ B = alpha \times B A \f$
 *    - \f$ B = alpha \times B A^T \f$ if *ARMAS_TRANS*  set
 *
 * The matrix A is upper (lower) triangular matrix if *ARMAS_UPPER* (*ARMAS_LOWER*) is
 * set. If matrix A is upper (lowert) then the strictly lower (upper) part is not
 * referenced. Flag bit *ARMAS_UNIT* indicates that matrix A is unit diagonal and the diagonal
 * entries are not accessed.
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out] B  Result matrix
 * @param[in]   alpha scalar multiplier
 * @param[in]   A Triangular operand matrix
 * @param[in]   flags option bits
 * @param[in,out] conf environment configuration
 *
 * @retval 0  Succeeded
 * @retval <0 Failed, conf.error set to error code.
 *
 * @ingroup blas3
 */
int armas_x_mult_trm(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_x_size(B) == 0 || armas_x_size(A) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    require(A->step >= A->rows && B->step >= B->rows);
    // check consistency
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        ok = B->cols == A->rows && A->cols == A->rows;
        break;
    case ARMAS_LEFT:
    default:
        ok = B->rows == A->cols && A->cols == A->rows;
        break;
    }
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (conf->optflags & ARMAS_ONAIVE) {
        armas_x_trmm_unb(B, alpha, A, flags);
        return 0;
    }

    if (CONFIG_ACCELERATORS) {
        struct armas_ac_blas3 args;
        armas_ac_set_blas3_args(&args, ZERO, __nil, alpha, A, B, flags);
        int rc = armas_ac_dispatch(conf->accel, ARMAS_AC_TRMM, &args, conf);
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

    switch (conf->optflags & ARMAS_ORECURSIVE) {
    case ARMAS_ORECURSIVE:
        armas_x_trmm_recursive(B, alpha, A, flags, &cache);
        break;
    default:
        armas_x_trmm_blk(B, alpha, A, flags, &cache);
        break;
    }
    armas_cbuf_release(&cbuf);
    return 0;
}

void armas_x_mult_trm_unsafe(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    armas_x_trmm_blk(B, alpha, A, flags, cache);
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
