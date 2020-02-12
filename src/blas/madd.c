// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Matrix copy operators
 */

//! \cond
#include <stdio.h>

#include "dtype.h"
//! \endcond
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_madd)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"

static
void vec_add(armas_x_dense_t *X,  const DTYPE alpha, int N)
{
    register int i, k;
    register DTYPE *x0;
    register int xinc = X->rows == 1 ? X->step : 1;

    for (i = 0; i < N-3; i += 4) {
        X->elems[(i+0)*xinc] += alpha;
        X->elems[(i+1)*xinc] += alpha;
        X->elems[(i+2)*xinc] += alpha;
        X->elems[(i+3)*xinc] += alpha;
    }
    if (i == N)
        return;

    x0 = &X->elems[i*xinc];
    k = 0;
    switch(N-i) {
    case 3:
        x0[k] += alpha;
        k += xinc;
    case 2:
        x0[k] += alpha;
        k += xinc;
    case 1:
        x0[k] += alpha;
    }
}

/**
 * @brief Element-wise increment of matrix by real constant.
 *
 * Affected elements are selected with flag bits. If ARMAS_UPPER (ARMAS_LOWER)
 * is set then upper (lower) triangular part is scaled. If bit ARMAS_UNIT
 * is set then diagonal entry is not touched.
 *
 * @param [in,out] A matrix
 * @param [in] alpha constant
 * @param [in] flags flag bits (ARMAS_UPPER,ARMAS_LOWER,ARMAS_UNIT)
 * \ingroup matrix
 */
int armas_x_madd(armas_x_dense_t *A, DTYPE alpha, int flags, armas_conf_t *cf)
{
    int c, n;
    armas_x_dense_t C;

    if (!cf)
        cf = armas_conf_default();

    if (armas_x_isvector(A)) {
        vec_add(A, alpha, armas_x_size(A));
        return 0;
    }

    switch (flags & (ARMAS_SYMM|ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        // scale strictly upper triangular part, if UNIT set, don't touch diagonal
        // (works for upper trapezoidal matrix too)
        n = (flags & ARMAS_UNIT) != 0 ? 1 : 0;
        for (c = n; c < A->cols; ++c) {
            armas_x_submatrix_unsafe(&C, A, 0, c+n, c+1-n, 1);
            vec_add(&C, alpha, C.rows);
        }
        break;

    case ARMAS_LOWER:
        // scale strictly lower triangular part. if UNIT set, don't touch diagonal
        // (works for lower trapezoidal matrix too)
        n = (flags & ARMAS_UNIT) != 0 ? 1 : 0;
        for (c = 0; c < A->cols-n; ++c) {
            armas_x_submatrix_unsafe(&C, A, c+n, c, A->rows-c-n, 1);
            vec_add(&C, alpha, C.rows);
        }
        break;

    case ARMAS_SYMM:
        if (A->rows != A->cols) {
            cf->error = ARMAS_ESIZE;
            return -1;
        }
        // fall through to do it.
    default:
        for (c = 0; c < A->cols; c++) {
            armas_x_submatrix_unsafe(&C, A, 0, c, A->rows, 1);
            vec_add(&C, alpha, C.rows);
        }
        break;
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
