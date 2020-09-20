
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! two norm of vector

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_nrm2)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

/*
 * Overflow is avoided by summing squares of scaled numbers and
 * then multiplying then with the scaling factor. Following is
 * is by Hammarling and included in BLAS reference libary.
 */

/*
 * Scaled summation of squares
 * ---------------------------
 *
 *  X = {x0, x1, x2, x3} and |x0| > |x_i|, i!=0
 *
 *  then w_0 = amax(X)
 *       S_0 = 1.0 + sum (|x_k|/|w|)^2, where k != iamax(X)
 *
 *  X_1  = X_0 + {x4} and |x_4| > amax(X_0)
 *
 *   then w_1 = |x_4|
 *        S_1 = 1.0 + (w_0/|x_4|)^2 * S_0
 *
 *  [nrm2] := nrm2(X, N)
 *     w = |x_0|
 *     S = 1.0
 *     for k = 1 : N-1
 *        |x_k| <= |w|:
 *           S = S + (|x_k|/|w|)^2
 *        |x_k| >  |w|:
 *           S = 1.0 + (|w|/|x_k|)^2 * S
 *           w = |x_k|
 *      return w * sqrt(S)
 *
 *   updated sum-of-squares
 *   [w, S] := ssumsq(X, N, w_0, S_0)
 *      w = w_0
 *      S = S_0
 *      for k = 0 : N-1:
 *        |x_k| <= |w|:
 *           S = S + (|x_k|/|w|)^2
 *        |x_k| >  |w|:
 *           S = 1.0 + (|w|/|x_k|)^2 * S
 *           w = |x_k|
 *      return [w, S]
 */

/**
 * @brief Norm2 of vector, \f$ ||x||_2 \f$
 *
 * @param[in] x vector
 * @param[in,out] conf configuration block
 *
 * @return Norm of the vector
 * @ingroup blas
 */
ABSTYPE armas_x_nrm2(const armas_x_dense_t *x, armas_conf_t *conf)
{
    register int i;
    register ABSTYPE a0, sum, scale;

    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return ABSZERO;
    }

    if (armas_x_size(x) == 0) {
        return ABSZERO;
    }
    if (armas_x_size(x) == 1) {
        return ABS(x->elems[0]);
    }

    int inc  = x->rows == 1 ? x->step : 1;
    sum = ABSONE;
    scale = ABSZERO;
    for (i = 0; i < armas_x_size(x); i += 1) {
        if (x->elems[(i+0)*inc] != ZERO) {
            a0 = ABS(x->elems[(i+0)*inc]);
            if (a0 > scale) {
                sum = ONE + sum * ((scale/a0)*(scale/a0));
                scale = a0;
            } else {
                sum = sum + (a0/scale)*(a0/scale);
            }
        }
    }
    return scale*SQRT(sum);
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */
