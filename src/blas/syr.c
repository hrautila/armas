
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank update

//! \cond
#include <stdio.h>
#include <stdint.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mvupdate_sym)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(armas_x_mvupdate_trm)
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
 * @brief Symmetric matrix rank-1 update.
 *
 * Computes 
 *    - \f$ A = A + alpha \times X X^T \f$
 *
 * where A is symmetric matrix stored in lower (upper) triangular part of matrix A.
 * If flag *ARMAS_LOWER* (*ARMAR_UPPER*) is set matrix is store in lower (upper) triangular
 * part of A and upper (lower) triangular part is not referenced.
 *
 * @param[in,out]  A target matrix
 * @param[in]      alpha scalar multiplier
 * @param[in]      X source vector
 * @param[in]      flags flag bits 
 * @param[in]      conf configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas2
 */
int armas_x_mvupdate_sym(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *x,
    int flags,
    armas_conf_t *conf)
{
    return armas_x_mvupdate_trm(beta, A, alpha, x, x, flags, conf);
}

#else
#warning "Missing defines; no code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
