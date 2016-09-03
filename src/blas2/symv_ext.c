
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#if HAVE_CONFIG
#include "config.h"
#endif

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__symv_ext_unb) //&& defined(armas_x_ex_mvmult_sym)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
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


int __symv_ext_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                   DTYPE alpha, DTYPE beta, int flags, int N)
{
    int i, j;
    DTYPE y0, s0, x0, p0, c0;
  
    if ( N <= 0 )
        return 0;

    if (flags & ARMAS_LOWER) {
        for (i = 0; i < N; i++) {
            y0 = s0 = 0.0;
            // follow row A[i,0:i] 
            for (j = 0; j < i; j++) {
                twoprod(&x0, &p0, A->md[(i+0)+j*A->step], X->md[j*X->inc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            // follow column A[i:N,i]
            for (j = i; j < N; j++) {
                twoprod(&x0, &p0, A->md[(j+0)+i*A->step], X->md[j*X->inc]);
                twosum(&y0, &c0, y0, x0);
                s0 += c0 + p0;
            }
            twoprod(&y0, &p0, y0+s0, alpha);
            // here: y0 + p0 = alpha*sum(A_i,j*x_j)
            twoprod(&x0, &s0, Y->md[(i+0)*Y->inc], beta);
            // here: x0 + s0 = beta*y[i]
            p0 += s0; 
            twosum(&y0, &s0, x0, y0);

            Y->md[(i+0)*Y->inc] = y0 + p0;
        }
        return 0;
    }

    // Upper here;
    //  1. update elements 0:j with current column and x[j]
    //  2. update current element y[j] with product of a[0:j-1]*x[0:j-1]
    for (i = 0; i < N; i++) {
        y0 = s0 = 0.0;
        // follow column A[0:i,i]
        for (j = 0; j < i; j++) {
            twoprod(&x0, &p0, A->md[(j+0)+i*A->step], X->md[j*X->inc]);
            twosum(&y0, &c0, y0, x0);
            s0 += c0 + p0;
        }
        // follow row A[i,i:N] 
        for (j = i; j < N; j++) {
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


#if 0
/**
 * @brief Symmetic matrix-vector multiply.
 *
 * Computes
 *
 * > Y := alpha*A*X + beta*Y
 *
 *  @param[in,out]  Y   target and source vector
 *  @param[in]      A   symmetrix lower (upper) matrix
 *  @param[in]      X   source operand vector
 *  @param[in]      alpha, beta scalars
 *  @param[in]      flags  flag bits
 *  @param[in]      conf   configuration block
 *
 */
int armas_x_ex_mvmult_sym(armas_x_dense_t *Y, const armas_x_dense_t *A, const armas_x_dense_t *X,
                          DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int ok;
  mvec_t x, y;
  mdata_t A0;
  int nx = armas_x_size(X);
  int ny = armas_x_size(Y);
  
  if (armas_x_size(A) == 0 || armas_x_size(X) == 0 || armas_x_size(Y) == 0)
    return 0;
  
  if (!conf)
    conf = armas_conf_default();

  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (Y->rows != 1 && Y->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }

  ok = A->cols == A->rows && nx == ny && nx == A->cols;
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  y = (mvec_t){Y->elems, (Y->rows == 1 ? Y->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  __symv_ext_unb(&y, &A0, &x, alpha, beta, flags, nx);
  return 0;
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
