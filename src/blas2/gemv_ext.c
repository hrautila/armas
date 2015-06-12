
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#if HAVE_CONFIG
#include "config.h"
#endif

/** @defgroup xblas2 BLAS level 2 extended precision functions.
 *
 */
#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__gemv_ext_unb) && defined(__gemv_update_ext_unb)
#define __ARMAS_PROVIDES 1
#endif
// if extended precision enabled and requested
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

int __gemv_ext_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                  DTYPE alpha, DTYPE beta, int flags, int nX, int nY)
{
    int i, j;
    DTYPE y0, c0, s0, x0, p0;
    DTYPE y1, c1, s1, x1, p1;

    // A transposed
    if (flags & ARMAS_TRANS) {
        // here were proceed matrix A column wise and elements are close to each other
        for (i = 0; i < nY; i++) {
            y0 = 0.0; s0 = 0.0; y1 = 0.0; s1 = 0.0;
            for (j = 0; j < nX; j++) {
                twoprod(&x0, &p0, A->md[j+(i+0)*A->step], X->md[j*X->inc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            twoprod(&y0, &p0, y0+s0, alpha);
            // here: y_i + p_i = alpha*sum(A_i,j*x_j)
            twoprod(&x0, &s0, Y->md[(i+0)*Y->inc], beta);
            // here: x_i + s_i = beta*y_i
            p0 += s0;
            twosum(&y0, &s0, x0, y0);
            Y->md[(i+0)*Y->inc] = y0 + p0;
        }
        return 0;
    }

    // A not transposed; here were proceed matrix A rowwise and elements are at distance
    for (i = 0; i < nY; i++) {
        y0 = 0.0; s0 = 0.0; y1 = 0.0; s1 = 0.0;
        for (j = 0; j < nX; j++) {
            twoprod(&x0, &p0, A->md[(i+0)+j*A->step], X->md[j*X->inc]);
            twosum(&y0, &c0, y0, x0);
            s0 += c0 + p0;
        }
        twoprod(&y0, &p0, y0+s0, alpha);
        // here: y_i + p_i = alpha*sum(A_i,j*x_j)
        twoprod(&x0, &s0, Y->md[(i+0)*Y->inc], beta);
        // here: x_i + s_i = beta*y_i
        p0 += s0;
        twosum(&y0, &s0, x0, y0);
        Y->md[(i+0)*Y->inc] = y0 + p0;
    }
    return 0;
}


/*
 * Compute error free translation Y + dY = Y + dY +/- A*X
 *
 * If sign is non-zero computes Y + dY = Y + dY - A*x, otherwise Y + dY = Y + dY + A*x
 */
int __gemv_update_ext_unb(mvec_t *Y, mvec_t *dY, const mdata_t *A, const mvec_t *X,
                          int sign, int flags, int nX, int nY)
{
    int i, j;
    DTYPE y0, c0, s0, x0, p0;
    DTYPE y1, c1, s1, x1, p1;

    // A transposed
    if (flags & ARMAS_TRANS) {
        // here were proceed matrix A column wise and elements are close to each other
        for (i = 0; i < nY; i++) {
            y0 = Y->md[i*Y->inc];
            s0 = dY->md[i*dY->inc];
            if (sign) {
                // we will compute -Y + A*x
                y0 = -y0; s0 = -s0;
            }
            for (j = 0; j < nX; j++) {
                twoprod(&x0, &p0, A->md[j+(i+0)*A->step], X->md[j*X->inc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            if (sign) {
                // negate to get Y - A*x
                y0 = -y0; s0 = -s0;
            }
            Y->md[(i+0)*Y->inc] = y0;
            dY->md[i*dY->inc] = s0;
        }
        return 0;
    }

    // A not transposed; here were proceed matrix A rowwise and elements are at distance
    for (i = 0; i < nY; i++) {
        y0 = Y->md[i*Y->inc];
        s0 = dY->md[i*dY->inc];
        if (sign) {
            // we will compute -Y + A*x
            y0 = -y0; s0 = -s0;
        }
        for (j = 0; j < nX; j++) {
            twoprod(&x0, &p0, A->md[(i+0)+j*A->step], X->md[j*X->inc]);
            twosum(&y0, &c0, y0, x0);
            s0 += c0 + p0;
        }
        if (sign) {
            // negate to get Y - A*x
            y0 = -y0; s0 = -s0;
        }
        Y->md[(i+0)*Y->inc] = y0;
        dY->md[i*dY->inc] = s0;
    }
    return 0;
}

#if 0
/*
 * @brief General matrix-vector multiply.
 *
 * Computes
 *
 * > Y := alpha*A*X + beta*Y\n
 * > Y := alpha*A.T*X + beta*Y  if ARMAS_TRANS
 *
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      A   source operand matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      alpha, beta scalars
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 * @ingroup xblas2
 */
int __armas_ex_mvmult(__armas_dense_t *Y, const __armas_dense_t *A, const __armas_dense_t *X,
                      DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
    int ok;
    mvec_t x, y;
    mdata_t A0;
    int nx = __armas_size(X);
    int ny = __armas_size(Y);
  
    if (!conf)
        conf = armas_conf_default();

    if (__armas_size(A) == 0 || __armas_size(X) == 0 || __armas_size(Y) == 0)
        return 0;
  
    if (X->rows != 1 && X->cols != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (Y->rows != 1 && Y->cols != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }

    // check consistency
    switch (flags & ARMAS_TRANS) {
    case ARMAS_TRANS:
        ok = A->cols == ny && A->rows == nx;
        break;
    default:
        ok = A->rows == ny && A->cols == nx;
        break;
    }
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
    y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
    A0 = (mdata_t){A->elems, A->step};

    __gemv_ext_unb(&y, &A0, &x, alpha, flags, 0, nx, 0, ny);
    return 0;
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
