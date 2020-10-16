// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Matrix copy operators
 */

#include <stdio.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_mcopy) && defined(armas_copy)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "matcpy.h"

static
void vector_copy(armas_dense_t *X,  const armas_dense_t *Y, int N)
{
    register int i, kx, ky;
    register DTYPE f0, f1, f2, f3;
    register int yinc = Y->rows == 1 ? Y->step : 1;
    register int xinc = X->rows == 1 ? X->step : 1;
    for (i = 0; i < N-3; i += 4) {
        f0 = Y->elems[(i+0)*yinc];
        f1 = Y->elems[(i+1)*yinc];
        f2 = Y->elems[(i+2)*yinc];
        f3 = Y->elems[(i+3)*yinc];
        X->elems[(i+0)*xinc] = f0;
        X->elems[(i+1)*xinc] = f1;
        X->elems[(i+2)*xinc] = f2;
        X->elems[(i+3)*xinc] = f3;
    }
    if (i == N)
        return;

    // calculate indexes only once
    kx = i*xinc;
    ky = i*yinc;
    switch (N-i) {
    case 3:
        X->elems[kx] = Y->elems[ky];
        kx += xinc; ky += yinc;
    case 2:
        X->elems[kx] = Y->elems[ky];
        kx += xinc; ky += yinc;
    case 1:
        X->elems[kx] = Y->elems[ky];
    }
}

/**
 * @brief Copy vector, \f$ Y := X \f$
 *
 * @param[out] Y target vector
 * @param[in]  X source vector
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval < 0 Failed, conf->error holds error code
 *
 * @ingroup blas
 */
int armas_copy(armas_dense_t *Y, const armas_dense_t *X, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (armas_size(X) == 0 || armas_size(Y) == 0) {
        return 0;
    }
    // only for column or row vectors
    if (!(armas_isvector(X) && armas_isvector(Y))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    int N = armas_size(X) < armas_size(Y) ? armas_size(X) : armas_size(Y);

    vector_copy(Y, X, N);
    return 0;
}

/**
 * @brief Copy matrix or vector
 *
 * Copy matrix or vector to another matrix or vector, ie. A = B. Sizes of
 * of operand must match, for matrix size(A) == size(B), for vector len(A) == len(B)
 *
 * @param [out] A
 *     Destination matrix or vector, on exit copy of source matrix
 * @param [in] B
 *     Source matrix or vector
 * @param [in] flags
 *     If ARMAS_TRANS then B^T is copied.
 * @param [in,out] cf
 *     Configuration block.
 *
 * @retval  0 Success
 * @retval <0 Failure, *cf.error* holds error code
 *
 * @ingroup matrix
 */
int armas_mcopy(armas_dense_t *A, const armas_dense_t *B, int flags, armas_conf_t *cf)
{
    int ok;

    if (!cf)
        cf = armas_conf_default();

    if (armas_isvector(A) && armas_isvector(B)) {
        if (armas_size(A) != armas_size(B)) {
            cf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
        // blas1 vector copy
        vector_copy(A, B, armas_size(A));
        return 0;
    }
    switch (flags & ARMAS_TRANS) {
    case ARMAS_TRANS:
        ok = A->rows == B->cols && A->cols == B->rows;
        break;
    default:
        ok = A->rows == B->rows && A->cols == B->cols;
        break;
    }
    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (flags & ARMAS_TRANS) {
        CPTRANS(A->elems, A->step, B->elems, B->step, B->rows, B->cols);
    } else {
        CP(A->elems, A->step, B->elems, B->step, B->rows, B->cols);
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
