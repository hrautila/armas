
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvupdate_trm) && defined(__trmv_recursive)
#define __ARMAS_PROVIDES 1
#endif
// this module requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

#if EXT_PRECISION && defined(__trmv_ext_unb)
#define HAVE_EXT_PRECISION 1
extern int __trmv_ext_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N);
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0
 *     0 |a11|a12  b1
 *     0 | 0 |a22  b2
 *
 *    b00 = a00*b0 + a01*b1 + a02*b2
 *    b10 =          a11*b1 + a12*b2
 *    b20 =                   a22*b2
 */
static inline
void __trmv_unb_lu(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                 int incX, int ldA, int nRE)
{
  register int i;
  for (i = 0; i < nRE; i++) {
    // update all previous b-values with current A column and current B
    __vmult1axpy(&X[0], incX, &Ac[i*ldA], &X[i*incX], 1, alpha, i);
    X[i*incX] = unit ? X[i*incX] : alpha*X[i*incX]*Ac[i+i*ldA];
  }
}

static inline
void __trmv_unb_lu_abs(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                       int incX, int ldA, int nRE)
{
  register int i;
  for (i = 0; i < nRE; i++) {
    // update all previous b-values with current A column and current B
    __vmult1axpy_abs(&X[0], incX, &Ac[i*ldA], &X[i*incX], 1, alpha, i);
    X[i*incX] = unit ? __ABS(X[i*incX]) : alpha*__ABS(X[i*incX])*__ABS(Ac[i+i*ldA]);
  }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static inline
void __trmv_unb_lut(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                    int incX, int ldA, int nRE)
{
  register int i;
  DTYPE xtmp;

  for (i = nRE; i > 0; i--) {
    xtmp = unit ? alpha*X[(i-1)*incX] : 0.0;
    __vmult1dot(&xtmp, 1, &Ac[(i-1)*ldA], &X[0], incX, alpha, i-unit);
    X[(i-1)*incX] = xtmp;
  }
}

static inline
void __trmv_unb_lut_abs(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                        int incX, int ldA, int nRE)
{
  register int i;
  DTYPE xtmp;

  for (i = nRE; i > 0; i--) {
    xtmp = unit ? alpha*__ABS(X[(i-1)*incX]) : 0.0;
    __vmult1dot_abs(&xtmp, 1, &Ac[(i-1)*ldA], &X[0], incX, alpha, i-unit);
    X[(i-1)*incX] = xtmp;
  }
}

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
static inline
void __trmv_unb_ll(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                   int incX, int ldA, int nRE)
{
  register int i;

  for (i = nRE; i > 0; i--) {
    // update all b-values below with the current A column and current B
    __vmult1axpy(&X[i*incX], 1, &Ac[i+(i-1)*ldA], &X[(i-1)*incX], 1, alpha, nRE-i);
    X[(i-1)*incX] = alpha * (unit ? X[(i-1)*incX] : X[(i-1)*incX]*Ac[(i-1)+(i-1)*ldA]);
  }
}

static inline
void __trmv_unb_ll_abs(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                       int incX, int ldA, int nRE)
{
  register int i;

  for (i = nRE; i > 0; i--) {
    // update all b-values below with the current A column and current B
    __vmult1axpy_abs(&X[i*incX], 1, &Ac[i+(i-1)*ldA], &X[(i-1)*incX], 1, alpha, nRE-i);
    X[(i-1)*incX] = alpha * (unit ? __ABS(X[(i-1)*incX]) 
                             : __ABS(X[(i-1)*incX])*__ABS(Ac[(i-1)+(i-1)*ldA]));
  }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 */
static inline
void __trmv_unb_llt(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                    int incX, int ldA, int N)
{
  register int i;
  DTYPE xtmp;

  for (i = 0; i < N; i++) {
    xtmp = unit ? alpha*X[i*incX] : 0.0;
    __vmult1dot(&xtmp, 1, &Ac[(i+unit)+i*ldA], &X[(i+unit)*incX], incX, alpha, N-unit-i);
    X[i*incX] = xtmp;
  }
}

static inline
void __trmv_unb_llt_abs(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                        int incX, int ldA, int N)
{
  register int i;
  DTYPE xtmp;

  for (i = 0; i < N; i++) {
    xtmp = unit ? alpha*__ABS(X[i*incX]) : 0.0;
    __vmult1dot_abs(&xtmp, 1, &Ac[(i+unit)+i*ldA], &X[(i+unit)*incX], incX, alpha, N-unit-i);
    X[i*incX] = xtmp;
  }
}



static
void __trmv_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
  case ARMAS_UPPER|ARMAS_TRANS:
    if (flags & ARMAS_ABS) {
      __trmv_unb_lut_abs(X->md, A->md, alpha, unit, X->inc, A->step, N);
    } else {
      __trmv_unb_lut(X->md, A->md, alpha, unit, X->inc, A->step, N);
    }
    break;
  case ARMAS_UPPER:
    if (flags & ARMAS_ABS) {
      __trmv_unb_lu_abs(X->md, A->md, alpha, unit, X->inc, A->step, N);
    } else {
      __trmv_unb_lu(X->md, A->md, alpha, unit, X->inc, A->step, N);
    }
    break;
  case ARMAS_LOWER|ARMAS_TRANS:
    if (flags & ARMAS_ABS) {
      __trmv_unb_llt_abs(X->md, A->md, alpha, unit, X->inc, A->step, N);
    } else {
      __trmv_unb_llt(X->md, A->md, alpha, unit, X->inc, A->step, N);
    }
    break;
  case ARMAS_LOWER:
  default:
    if (flags & ARMAS_ABS) {
      __trmv_unb_ll_abs(X->md, A->md, alpha, unit, X->inc, A->step, N);
    } else {
      __trmv_unb_ll(X->md, A->md, alpha, unit, X->inc, A->step, N);
    }
    break;
  }
}

static
void __trmv_forward_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  // top part
  __subvector(&x0, X, 0);
  __subblock(&a0, A, 0, 0);
  __trmv_forward_recursive(&x0, &a0, alpha, flags, N/2);

  // update top with bottom
  __subvector(&x1, X, N/2);
  if (flags & ARMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __gemv_recursive(&x0, &a1, &x1, alpha, 1.0, flags, 0, N-N/2, 0, N/2);


  // bottom part
  __subblock(&a1, A, N/2, N/2);
  __trmv_forward_recursive(&x1, &a1, alpha, flags, N-N/2);
}

static
void __trmv_backward_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  //printf("__trmv_bk_recursive: N=%d\n", N);
  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  // bottom part
  __subvector(&x1, X, N/2);
  __subblock(&a1, A, N/2, N/2);
  __trmv_backward_recursive(&x1, &a1, alpha, flags, N-N/2);

  // update bottom with top
  __subvector(&x0, X, 0);
  if (flags & ARMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __gemv_recursive(&x1, &a0, &x0, alpha, 1.0, flags, 0, N/2, 0, N-N/2);


  // top part
  __subblock(&a0, A, 0, 0);
  __trmv_backward_recursive(&x0, &a0, alpha, flags, N/2);
}

void __trmv_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
  case ARMAS_LOWER|ARMAS_TRANS:
  case ARMAS_UPPER:
    __trmv_forward_recursive(X, A, alpha, flags, N);
    break;

  case ARMAS_UPPER|ARMAS_TRANS:
  case ARMAS_LOWER:
  default:
    __trmv_backward_recursive(X, A, alpha, flags, N);
    break;
  }
}


/**
 * @brief Triangular matrix-vector multiply
 *
 * Computes
 *
 * > X = alpha*A*X\n
 * > X = alpha*A.T*X      if ARMAS_TRANS
 * > X = alpha*|A|*|X|\n  if ARMAS_ABS
 * > X = alpha*|A.T|*|X|  if ARMAS_ABS|ARMAS_TRANS
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
int __armas_mvmult_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
                       DTYPE alpha, int flags, armas_conf_t *conf)
{
  mvec_t x;
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

  x = (mvec_t){X->elems, (X->rows == 1 ? X->step : 1)};
  A0 = (mdata_t){A->elems, A->step};

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC) != 0) {
    // trust that compiler dead-code pruning removes this block if extended
    // precision is not enabled
    __trmv_ext_unb(&x, &A0, alpha, flags, nx);
    return 0;
  }

  // normal precision here
  switch (conf->optflags) {
  case ARMAS_RECURSIVE:
    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_UPPER:
      __trmv_forward_recursive(&x, &A0, alpha, flags, nx);
      break;
    case ARMAS_UPPER|ARMAS_TRANS:
    case ARMAS_LOWER:
    default:
      __trmv_backward_recursive(&x, &A0, alpha, flags, nx);
      break;
    }
    break;

  case ARMAS_SNAIVE:
  default:
    __trmv_unb(&x, &A0, alpha, flags, nx);
    break;

  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
