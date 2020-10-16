
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank update

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mvupdate_sym)
#define ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(armas_mvupdate_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

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
 * @param[in] beta
 *  Scalar multiplier
 * @param[in,out]  A
 *  Target matrix
 * @param[in] alpha
 *  Scalar multiplier
 * @param[in] x
 *  Source vector
 * @param[in] flags
 *  Flag bits
 * @param[in] conf
 *  Configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blas
 */
int armas_mvupdate_sym(
    DTYPE beta,
    armas_dense_t *A,
    DTYPE alpha,
    const armas_dense_t *x,
    int flags,
    armas_conf_t *conf)
{
    return armas_mvupdate_trm(beta, A, alpha, x, x, flags, conf);
}

#else
#warning "Missing defines; no code"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
