
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Triangular matrix solve

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_solve_trm)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_solve_unb) && \
    defined(armas_solve_recursive) && defined(armas_solve_blocked)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include <math.h>
#include "matrix.h"
#include "internal.h"
#include "accel.h"


/**
 * @brief Triangular solve with multiple right hand sides
 *
 * If flag bit *ARMAS_LEFT* is set then computes
 *    - \f$ B = alpha \times A^{-1} B \f$
 *    - \f$ B = alpha \times A^{-T} B \f$ if *ARMAS_TRANS* set
 *
 * If flag bit *ARMAS_RIGHT* is set then computes
 *    - \f$ B = alpha \times B A^{-1} \f$
 *    - \f$ B = alpha \times B A^{-T} \f$ if *ARMAS_TRANS* set
 *
 * The matrix A is upper (lower) triangular matrix if *ARMAS_UPPER* (*ARMAS_LOWER*) is
 * set. If matrix A is upper (lower) then the strictly lower (upper) part is not
 * referenced. Flag bit *ARMAS_UNIT* indicates that matrix A is unit diagonal and the diagonal
 * elements are not accessed.
 *
 * @param[in,out] B  Result matrix
 * @param[in]   alpha scalar multiplier
 * @param[in]   A Triangular operand matrix
 * @param[in]   flags option bits
 * @param[in,out] conf environment configuration
 *
 * @retval 0 Succeeded
 * @retval <0 Failed, *conf.error* set to error code.
 *
 * @ingroup blas
 */
int armas_solve_trm(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_size(B) == 0 || armas_size(A) == 0)
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
        ok = A->cols == A->rows && A->cols == B->rows;
        break;
    }
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (CONFIG_ACCELERATORS) {
        struct armas_ac_blas3 args;
        armas_ac_set_blas3_args(&args, ZERO, __nil, alpha, A, B, flags);
        int rc = armas_ac_dispatch(conf->accel, ARMAS_AC_TRSM, &args, conf);
        if (rc != -ARMAS_EIMP)
            return rc;
        /* fallthru to local version. */
    }

    if (conf->optflags & ARMAS_ONAIVE) {
        armas_solve_unb(B, alpha, A, flags);
        return 0;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, &cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    // otherwise; normal precision here
    switch (conf->optflags & ARMAS_ORECURSIVE) {
    case ARMAS_ORECURSIVE:
        armas_solve_recursive(B, alpha, A, flags, &cache);
        break;
    default:
        armas_solve_blocked(B, alpha, A, flags, &cache);
        break;
    }
    return 0;
}

void armas_solve_trm_unsafe(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    cache_t *cache)
{
    armas_solve_blocked(B, alpha, A, flags, cache);
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
