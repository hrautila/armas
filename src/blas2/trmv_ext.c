
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__trmv_ext_unb) 
#define __ARMAS_PROVIDES 1
#endif
// this module requires no external public functions
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
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = a00*b'0
 *  b1 = a10*b'0 + a11*b'1
 *  b2 = a20*b'0 + a21*b'1 + a22*b'2
 */
static
void __trmv_ex_unb_ll(mvec_t *X, const mdata_t *A, DTYPE alpha, int unit, int N)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;

    //printf("..ex_unb_lut: LOWER-NOTRANSPOSE...\n");
    for (i = N-1; i >= 0; i--) {
        s0 = unit ? X->md[i*X->inc] : 0.0;
        u0 = 0.0;
        for (j = 0; j < i+(1-unit); j++) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->md[i+j*A->step], X->md[j*X->inc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->md[i*X->inc] = s0 + u0 + c0;
    }
}


/*
 * LEFT-LOWER-TRANSPOSE
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 */
static
void __trmv_ex_unb_llt(mvec_t *X, const mdata_t *A, DTYPE alpha, int unit, int N)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;

    //printf("..ex_unb_llt: LOWER-TRANSPOSE... unit=%d\n", unit);
    for (i = 0; i < N; i++) {
        s0 = unit ? X->md[i*X->inc] : 0.0;
        u0 = 0.0;
        for (j = i+unit; j < N; j++) { 
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->md[j+i*A->step], X->md[j*X->inc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->md[i*X->inc] = s0 + u0 + c0;
    }
}

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2
 *    b1 =           a11*b'1 + a12*b'2
 *    b2 =                     a22*b'2
 */
static
void __trmv_ex_unb_lu(mvec_t *X, const mdata_t *A, DTYPE alpha, int unit, int N)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;

    //printf("..ex_unb_lu: UPPER-NOTRANSPOSE...\n");
    for (i = 0; i < N; i++) {
        s0 = unit ? X->md[i*X->inc] : 0.0;
        u0 = 0.0;
        for (j = i+unit; j < N; j++) { 
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->md[i+j*A->step], X->md[j*X->inc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->md[i*X->inc] = s0 + u0 + c0;
    }
}

/*
 * LEFT-UPPER-TRANSPOSE
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static
void __trmv_ex_unb_lut(mvec_t *X, const mdata_t *A, DTYPE alpha, int unit, int N)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;

    for (i = N-1; i >= 0; i--) {
        s0 = unit ? X->md[i*X->inc] : 0.0;
        u0 = 0.0;
        for (j = 0; j < i+(1-unit); j++) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->md[j+i*A->step], X->md[j*X->inc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->md[i*X->inc] = s0 + u0 + c0;
    }
}


int __trmv_ext_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        __trmv_ex_unb_lut(X, A, alpha, unit, N);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        __trmv_ex_unb_llt(X, A, alpha, unit, N);
        break;
    case ARMAS_UPPER:
        __trmv_ex_unb_lu(X, A, alpha, unit, N);
        break;
    case ARMAS_LOWER:
    default:
        __trmv_ex_unb_ll(X, A, alpha, unit, N);
        break;
    }
    return 0;
}

#if 0
/*
 * @brief Triangular matrix-vector multiply
 *
 * Computes
 *
 * > X = alpha*A*X\n
 * > X = alpha*A.T*X  if ARMAS_TRANS
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
 */
int __armas_ex_mvmult_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
                          DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ok;
  mvec_t x, y;
  mdata_t A0;
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

  __trmv_ext_unb(&x, &A0, alpha, flags, nx);
  return 0;
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
