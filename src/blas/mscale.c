// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Matrix copy operators
 */

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_mscale) && defined(armas_scale)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "matcpy.h"

static inline
void vec_scale(armas_dense_t *X,  const DTYPE alpha, int N)
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

/**
 * @brief Scale matrix or vector unsafely; no bounds checks.
 */
int armas_scale_unsafe(armas_dense_t *x, DTYPE alpha)
{
    if (armas_isvector(x)) {
        vec_scale(x, alpha, armas_size(x));
    } else {
        blk_scale(x, alpha, x->rows, x->cols);
    }
    return 0;
}

/**
 * @brief Computes \f$ x = alpha*x \f$
 *
 * @param[in,out] x vector
 * @param[in] alpha scalar multiplier
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval <0 Failed, conf->error holds error code
 *
 * @ingroup blas
 */
int armas_scale(armas_dense_t *x, const DTYPE alpha, armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (armas_size(x) == 0)
        return 0;

    require(x->step >= x->rows);

    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }

    vec_scale(x, alpha, armas_size(x));
    return 0;
}


/**
 * @brief Scale matrix by real constant.
 * 
 * Computes  \f$ A = \alpha \times A \f$
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
 * @param[in,out] cf
 *      Configuration block.
 *
 * @ingroup matrix
 */
int armas_mscale(armas_dense_t *m, const DTYPE alpha, int flags, armas_conf_t *cf)
{
    int c, n;
    armas_dense_t C;

    if (!cf)
        cf = armas_conf_default();

    require(m->step >= m->rows);

    switch (flags & (ARMAS_SYMM|ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        // scale strictly upper triangular part, if UNIT set, don't touch diagonal
        n = (flags & ARMAS_UNIT) ? 1 : 0;
        for (c = n; c < m->rows; c++) {
            armas_submatrix_unsafe(&C, m, c, c+n, 1, m->rows-c-n);
            vec_scale(&C, alpha, C.rows);
        }
        break;

    case ARMAS_LOWER:
        // scale strictly lower triangular part. if UNIT set, don't touch diagonal
        n = (flags & ARMAS_UNIT) ? 1 : 0;
        for (c = 0; c < m->cols-n; c++) {
            armas_submatrix_unsafe(&C, m, c+n, c, m->rows-c-n, 1);
            vec_scale(&C, alpha, C.rows);
        }
        break;

    case ARMAS_SYMM:
        if (m->rows != m->cols) {
            cf->error = ARMAS_ESIZE;
            return -1;
        }
        // fall through to do it.
    default:
        armas_scale_unsafe(m, alpha);
        break;
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
