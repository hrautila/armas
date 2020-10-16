
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ext_mvmult_sym_unsafe) && defined(armas_ext_mvmult_sym)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

/*
 * Objective: read matrix A in memory order, along columns.
 *
 *  y0    a00 |  0   0     x0     y0 = a00*x0 + a10*x1 + a20*x2
 *  --    --------------   --
 *  y1    a10 | a11  0     x1     y1 = a10*x0 + a11*x1 + a21*x2
 *  y2    a20 | a21  a22   x2     y2 = a20*x0 + a21*x1 + a22*x2
 *
 *  y1 += (a11) * x1
 *  y2    (a21)
 *
 *  y1 += a21.T*x2
 *
 * UPPER:
 *  y0    a00 | a01 a02   x0     y0 = a00*x0 + a01*x1 + a02*x2
 *  --    --------------   --
 *  y1     0  | a11 a12   x1     y1 = a01*x0 + a11*x1 + a12*x2
 *  y2     0  |  0  a22   x2     y2 = a02*x0 + a12*x1 + a22*x2
 *
 *  (y0) += (a01) * x1
 *  (y1)    (a11)
 */


int armas_ext_mvmult_sym_unsafe(
    DTYPE beta,
    armas_dense_t *Y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *X,
    int flags)
{
    int i, j;
    DTYPE y0, s0, x0, p0, c0;
    int xinc = X->rows == 1 ? X->step : 1;
    int yinc = Y->rows == 1 ? Y->step : 1;
    int N = armas_size(Y);

    if ( N <= 0 )
        return 0;

    if (flags & ARMAS_LOWER) {
        for (i = 0; i < N; ++i) {
            y0 = s0 = 0.0;
            // follow row A[i,0:i] 
            for (j = 0; j < i; ++j) {
                twoprod(&x0, &p0, A->elems[(i+0)+j*A->step], X->elems[j*xinc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            // follow column A[i:N,i]
            for (j = i; j < N; ++j) {
                twoprod(&x0, &p0, A->elems[(j+0)+i*A->step], X->elems[j*xinc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            twoprod(&y0, &p0, y0+s0, alpha);
            // here: y0 + p0 = alpha*sum(A_i,j*x_j)
            twoprod(&x0, &s0, Y->elems[(i+0)*yinc], beta);
            // here: x0 + s0 = beta*y[i]
            p0 += s0;
            twosum(&y0, &s0, x0, y0);

            Y->elems[(i+0)*yinc] = y0 + p0;
        }
        return 0;
    }

    // Upper here;
    //  1. update elements 0:j with current column and x[j]
    //  2. update current element y[j] with product of a[0:j-1]*x[0:j-1]
    for (i = 0; i < N; ++i) {
        y0 = s0 = 0.0;
        // follow column A[0:i,i]
        for (j = 0; j < i; ++j) {
            twoprod(&x0, &p0, A->elems[(j+0)+i*A->step], X->elems[j*xinc]);
            twosum(&y0, &c0, y0, x0);
            s0 += c0 + p0;
        }
        // follow row A[i,i:N]
        for (j = i; j < N; ++j) {
            twoprod(&x0, &p0, A->elems[(i+0)+j*A->step], X->elems[j*xinc]);
            twosum(&y0, &c0, y0, x0);
            s0 += c0 + p0;
        }
        twoprod(&y0, &p0, y0+s0, alpha);
        // here: y_i + p_i = alpha*sum(A_i,j*x_j)
        twoprod(&x0, &s0, Y->elems[(i+0)*yinc], beta);
        // here: x_i + s_i = beta*y_i
        p0 += s0;
        twosum(&y0, &s0, x0, y0);

        Y->elems[(i+0)*yinc] = y0 + p0;
    }
    return 0;
}


/**
 * @brief Symmetric matrix-vector multiply in extended precission.
 *
 * Computes
 *    - \f$ Y = alpha \times A X + beta \times Y \f$
 *
 * Matrix A elements are stored on lower (upper) triangular part of the matrix
 * if flag bit *ARMAS_LOWER* (*ARMAS_UPPER*) is set.
 *
 *  @param[in]      beta scalar
 *  @param[in,out]  y   target and source vector
 *  @param[in]      alpha scalar
 *  @param[in]      A   symmetrix lower (upper) matrix
 *  @param[in]      x   source operand vector
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 *  @retval  0  Success
 *  @retval <0  Failed
 *
 * @ingroup blasext
 */
int armas_ext_mvmult_sym(
    DTYPE beta,
    armas_dense_t *y,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *x,
    int flags,
    armas_conf_t *conf)
{
    int ok;
    int nx = armas_size(x);
    int ny = armas_size(y);

    if (armas_size(A) == 0 || nx == 0 || ny == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    if (!armas_isvector(x)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (!armas_isvector(y)) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }

    ok = A->cols == A->rows && nx == ny && nx == A->cols;
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    armas_ext_mvmult_sym_unsafe(beta, y, alpha, A, x, flags);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
