
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__trsv_ext_unb)
#define __ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if EXT_PRECISION && defined(__gemv_ext_unb)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a10*b0)/a11
 *  b2 = (b'2 - a20*b0 - a21*b1)/a22
 */
static
void __trsv_ext_unb_ll(mvec_t *X, mvec_t *dX, const mdata_t *Ac, int unit, int N, int nD)
{
    register int i, j, k, nRE;
    mvec_t X0, X1, dX0, dX1;
    mdata_t A0, A1;
    DTYPE s0, u0, p0, r0, c0, x0;

    u0 = __ZERO;
    for (k = 0; k < N; k += nD) {
        nRE = k < N - nD ? nD : N-k;
        dX->md[0] = 0.0;
        __subvector(&X0,   X, k);
        __subvector(&dX0, dX, k);
        __subblock(&A0, Ac, k, k);
        for (i = 0; i < nRE; i++) {
            s0 = __get_at(&X0, i); //.md[i*X0.inc];
            u0 = __get_at(&dX0, i); //0.0;
            for (j = 0; j < i; j++) {
                // p + r = A[i,j]*X[j]
                twoprod(&p0, &r0, __get(&A0, i, j), __get_at(&X0, j));
                // s + c = s - A[i,j]*X[j]
                twosum(&s0, &c0, s0, -p0);
                u0 = u0 - (__get(&A0, i, j)*__get_at(&dX0, j) + r0 - c0);
            }
            if (unit) {
                __set_at(&dX0, i, u0);
                __set_at(&X0,  i, s0); 
                continue;
            }
            // if not unit diagonal then here
            approx_twodiv(&x0, &p0, s0, __get(&A0, i, i));
            __set_at(&dX0, i, (p0 + u0/__get(&A0, i, i)));
            __set_at(&X0,  i, x0);
        }

        // if nD < N then assume gemv update runs faster and update remaining values
        // with solution of current block.
        if (N-k > nRE) {
            // update vector values below with current block
            __subvector(&X1,  X, k+nRE);
            __subvector(&dX1, dX, k+nRE);
            __subblock(&A1, Ac, k+nRE, k);  // row k+nRE and below
            __gemv_update_ext_unb(&X1, &dX1, &A1, &X0, &dX0, -1, 0, nRE, N-k-nRE);
        }
    }
}


/*
 * LEFT-LOWER-TRANSPOSE
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = (b'0 - a10*b1 - a20*b2)/a00
 *  b1 =          (b'1 - a21*b2)/a11
 *  b2 =                     b'2/a22
 */
static
void __trsv_ext_unb_llt(mvec_t *X, mvec_t *dX, const mdata_t *Ac, int unit, int N, int nD)
{
    register int i, j, k, nRE;
    mvec_t X0, X1, dX0, dX1;
    mdata_t A0, A1;
    DTYPE s0, u0, p0, r0, c0, x0;

    for (k = N; k > 0; k -= nD) {
        nRE = k > nD ? nD : k;
        dX->md[nRE-1] = 0.0;
        __subvector(&X1,   X, k-nRE);
        __subvector(&dX1, dX, k-nRE);
        __subblock(&A1, Ac, k-nRE, k-nRE);
        for (i = nRE-1; i >= 0; i--) {
            s0 = __get_at(&X1, i); //.md[i*X1.inc];
            u0 = __get_at(&dX1, i); //0.0;
            for (j = i+1; j < nRE; j++) { 
                // p + r = A[i,j]*X[j]
                twoprod(&p0, &r0, __get(&A1, j, i), __get_at(&X1, j));
                // s + c = s - A[i,j]*X[j]
                twosum(&s0, &c0, s0, -p0);
                u0 = u0 - (__get(&A1, j, i)*__get_at(&dX1, j) + r0 - c0);
            }
            if (unit) {
                __set_at(&dX1, i, u0);
                __set_at(&X1, i, s0);
                continue;
            }
            // if not unit diagonal then here
            approx_twodiv(&x0, &p0, s0, __get(&A1, i, i));
            __set_at(&dX1, i, (p0 + u0/__get(&A1, i, i)));
            __set_at(&X1,  i, x0);
        }

        if (k > nRE) {
            // update vector values above with current block
            __subvector(&X0,   X, 0);
            __subvector(&dX0, dX, 0);
            __subblock(&A0, Ac, k-nRE, 0);  
            __gemv_update_ext_unb(&X0, &dX0, &A0, &X1, &dX1, -1, ARMAS_TRANS, nRE, k-nRE);
        }
    }
}

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = (b'0 - a01*b1 - a02*b2)/a00
 *    b1 =          (b'1 - a12*b2)/a11
 *    b2 =                     b'2/a22
 */
static
void __trsv_ext_unb_lu(mvec_t *X, mvec_t *dX, const mdata_t *Ac, int unit, int N, int nD)
{
    register int i, j, k, nRE;
    mvec_t X0, X1, dX0, dX1;
    mdata_t A0, A1;
    DTYPE s0, u0, p0, r0, c0, x0;

    for (k = N; k > 0; k -= nD) {
        nRE = k > nD ? nD : k;
        __subvector(&X1,   X, k-nRE);
        __subvector(&dX1, dX, k-nRE);
        __subblock(&A1, Ac, k-nRE, k-nRE);
        for (i = nRE-1; i >= 0; i--) {
            s0 = __get_at(&X1, i); //.md[i*X1.inc];
            u0 = __get_at(&dX1,i ); //0.0;
            for (j = i+1; j < nRE; j++) { 
                // p + r = A[i,j]*X[j]
                twoprod(&p0, &r0, __get(&A1, i, j), __get_at(&X1, j));
                // s + c = s - A[i,j]*X[j]
                twosum(&s0, &c0, s0, -p0);
                u0 = u0 - (__get(&A1, i, j)*__get_at(&dX1, j) + r0 - c0);
            }
            if (unit) {
                __set_at(&dX1, i, u0);
                __set_at(&X1,  i, s0);
                continue;
            }
            // if not unit diagonal then here
            approx_twodiv(&x0, &p0, s0, __get(&A1, i, i));
            __set_at(&dX1, i, u0/__get(&A1, i, i) + p0);
            __set_at(&X1,  i, x0);
        }
        if (k > nRE) {
            // update vector values above with current block
            __subvector(&X0,   X, 0);
            __subvector(&dX0, dX, 0);
            __subblock(&A0, Ac, 0, k-nRE);  
            __gemv_update_ext_unb(&X0, &dX0, &A0, &X1, &dX1, -1, 0, nRE, k-nRE);
        }
    }
}

/*
 * LEFT-UPPER-TRANSPOSE
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a01*b0)/a11
 *  b2 = (b'2 - a02*b0 - a12*b1)/a22
 */
static
void __trsv_ext_unb_lut(mvec_t *X, mvec_t *dX, const mdata_t *Ac, int unit, int N, int nD)
{
    register int i, j, k, nRE;
    mvec_t X0, X1, dX0, dX1;
    mdata_t A0, A1;
    DTYPE s0, u0, p0, r0, c0, x0;

    for (k = 0; k < N; k += nD) {
        nRE = k < N - nD ? nD : N-k;
        __subvector(&X0,   X, k);
        __subvector(&dX0, dX, k);
        __subblock(&A0, Ac, k, k);
        for (i = 0; i < nRE; i++) {
            s0 = __get_at(&X0, i); //.md[i*X0.inc];
            u0 = __get_at(&dX0, i);
            for (j = 0; j < i; j++) {
                // p + r = A[i,j]*X[j]
                twoprod(&p0, &r0, __get(&A0, j, i), __get_at(&X0, j));
                // s + c = s - A[i,j]*X[j]
                twosum(&s0, &c0, s0, -p0);
                u0 = u0 - (__get(&A0, j, i)*__get_at(&dX0, j) + r0 - c0);
            }
            if (unit) {
                __set_at(&dX0, i, u0);
                __set_at(&X0,  i, s0);
                continue;
            }
            // if not unit diagonal then here
            approx_twodiv(&x0, &p0, s0, __get(&A0, i, i));
            __set_at(&dX0, i, u0/__get(&A0, i, i) + p0);
            __set_at(&X0,  i, x0);
        }
        if (N-k > nRE) {
            // update vector values below with current block
            __subvector(&X1,   X, k+nRE);
            __subvector(&dX1, dX, k+nRE);
            __subblock(&A1, Ac, k, k+nRE);  // row k+nRE and below
            __gemv_update_ext_unb(&X1, &dX1, &A1, &X0, &dX0, -1, ARMAS_TRANS, nRE, N-k-nRE);
        }
    }
}


#ifndef MAX_EPREC_IBUF
#define MAX_EPREC_IBUF  128/sizeof(DTYPE)
#endif

int __trsv_ext_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
    DTYPE dbuf[MAX_NB];
    DTYPE *dp = (DTYPE *)0;
    mvec_t dX = (mvec_t){dbuf, 1};
    int i, unit = flags & ARMAS_UNIT ? 1 : 0;
    int nD = MAX_NB;

    if (N > MAX_NB) {
      dp = calloc(N, sizeof(DTYPE));
      if (!dp)
        return -1;
      dX.md = dp;
    } else {
      memset(dbuf, 0, sizeof(dbuf));
    }

    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        __trsv_ext_unb_lut(X, &dX, A, unit, N, nD);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        __trsv_ext_unb_llt(X, &dX, A, unit, N, nD);
        break;
    case ARMAS_UPPER:
        __trsv_ext_unb_lu(X, &dX, A, unit, N, nD);
        break;
    case ARMAS_LOWER:
    default:
        __trsv_ext_unb_ll(X, &dX, A, unit, N, nD);
        break;
    }
    if (alpha == __ONE) {
        for (i = 0; i < N; i++) {
            __set_at(X, i, __get_at(X, i)+__get_at(&dX, i));
        }
    } else if (alpha == - __ONE) {
        for (i = 0; i < N; i++) {
            __set_at(X, i, -(__get_at(X, i)+__get_at(&dX, i)));
        }
    } else {
        DTYPE p, q;
        for (i = 0; i < N; i++) {
            twoprod(&p, &q, alpha, __get_at(X, i));
            __set_at(X, i, (p + q + alpha*__get_at(&dX, i)));
        }
    }
    if (dp)
      free(dp);
    return 0;
}


#if 0
/*
 * @brief Triangular matrix-vector solve
 *
 * Computes
 *
 * > X = alpha*A.-1*X\n
 * > X = alpha*A.-T*X   if ARMAS_TRANS 
 *
 * where A is upper (lower) triangular matrix defined with flag bits ARMAS_UPPER
 * (ARMAS_LOWER).
 *
 * @param[in,out] X target and source vector
 * @param[in]     A matrix
 * @param[in]     alpha scalar multiplier
 * @param[in]     flags operand flags
 * @param[in]     conf  configuration block
 *
 * @ingroup blas2
 */
int __armas_ex_mvsolve_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
                           DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ok;
  mvec_t x, y, dx;
  mdata_t A0;
  DTYPE dbuf[16];
  int nx = __armas_size(X);
  
  if (__armas_size(A) == 0 || __armas_size(X) == 0)
    return 0;
  
  if (!conf)
    conf = armas_conf_default();

  if (X->rows != 1 && X->cols != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (A->cols != nx || A->rows != A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  if ((flags & ARMAS_TRANSA) && !(flags & ARMAS_TRANS)) {
    flags |= ARMAS_TRANS;
  }

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  A0 = (mdata_t){A->elems, A->step};
  dx = (mvec_t){dbuf, 1};

  __trsv_ext_unb(&x, &dx, &A0, alpha, flags, nx, sizeof(dbuf)/sizeof(DTYPE));
  return 0;
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
