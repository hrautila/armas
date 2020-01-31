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
#if defined(armas_x_mscale) && defined(armas_x_scale)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "matcpy.h"

static inline
void __vec_scale(armas_x_dense_t *X,  const DTYPE alpha, int N)
{
    register int i, k;
    register DTYPE *x0;
    register int xinc = X->rows == 1 ? X->step : 1;

    for (i = 0; i < N-3; i += 4) {
        X->elems[(i+0)*xinc] *= alpha;
        X->elems[(i+1)*xinc] *= alpha;
        X->elems[(i+2)*xinc] *= alpha;
        X->elems[(i+3)*xinc] *= alpha;
    }
    if (i == N)
        return;

    x0 = &X->elems[i*xinc];
    k = 0;
    switch(N-i) {
    case 3:
        x0[k] *= alpha;
        k += xinc;
    case 2:
        x0[k] *= alpha;
        k += xinc;
    case 1:
        x0[k] *= alpha;
    }
}

#if defined(armas_x_scale_unsafe)
/**
 * @brief Scale matrix or vector unsafely; no bounds checks.
 */
int armas_x_scale_unsafe(armas_x_dense_t *x, DTYPE alpha)
{
    if (armas_x_isvector(x)) {
        __vec_scale(x, alpha, armas_x_size(x));
    } else {
        __blk_scale(x, alpha, x->rows, x->cols);
    }
    return 0;
}
#endif

/**
 * @brief Computes \f$ x = alpha*x \f$
 *
 * @param[in,out] x vector
 * @param[in] alpha scalar multiplier
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup blas1
 */
int armas_x_scale(armas_x_dense_t *x, const DTYPE alpha, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(x) == 0)
        return 0;

    if (!armas_x_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }

    __vec_scale(x, alpha, armas_x_size(x));
    return 0;
}


/**
 * @brief Scale matrix by real constant.
 *
 * Element wise scaling of matrix element by a constant. Affected
 * elements are selected with flag bits. If ARMAS_UPPER (ARMAS_LOWER)
 * is set then upper (lower) triangular part is scaled. If bit ARMAS_UNIT
 * is set then diagonal entry is not touched.
 *
 * @param [in,out] m
 *      On entry, the unscaled matrix. On exit, matrix with selected elements scaled.
 * @param [in] alpha
 *      scaling constant
 * @param [in] flags
 *      flag bits (ARMAS_UPPER,ARMAS_LOWER,ARMAS_UNIT)
 *
 * \ingroup matrix
 */
int armas_x_mscale(armas_x_dense_t *m, const DTYPE alpha, int flags, armas_conf_t *cf)
{
    int c, n;
    armas_x_dense_t C;

    if (!cf)
        cf = armas_conf_default();

    switch (flags & (ARMAS_SYMM|ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        // scale strictly upper triangular part, if UNIT set, don't touch diagonal
        n = flags & ARMAS_UNIT ? 1 : 0;
        for (c = n; c < m->rows; c++) {
            armas_x_submatrix_unsafe(&C, m, c, c+n, 1, m->rows-c-n);
            __vec_scale(&C, alpha, C.rows);
        }
        break;

    case ARMAS_LOWER:
        // scale strictly lower triangular part. if UNIT set, don't touch diagonal
        n = flags & ARMAS_UNIT ? 1 : 0;
        for (c = 0; c < m->cols-n; c++) {
            armas_x_submatrix_unsafe(&C, m, c+n, c, m->rows-c-n, 1);
            __vec_scale(&C, alpha, C.rows);
        }
        break;

    case ARMAS_SYMM:
        if (m->rows != m->cols) {
            cf->error = ARMAS_ESIZE;
            return -1;
        }
        // fall through to do it.
    default:
        for (c = 0; c < m->cols; c++) {
            armas_x_submatrix_unsafe(&C, m, c, 0, m->rows, 1);
            __vec_scale(&C, alpha, C.rows);
        }
        break;
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
