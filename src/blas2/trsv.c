
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mvsolve_trm)
#define __ARMAS_PROVIDES 1
#endif
// this module requires external public functions
#if defined(__gemv_recursive) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

#if EXT_PRECISION
extern int __trsv_ext_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N);
#endif


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
static inline
void __trsv_unb_lu(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;
   for (i = nRE; i > 0; i--) {
     X[(i-1)*incX] = unit ? X[(i-1)*incX] : X[(i-1)*incX]/Ac[(i-1)+(i-1)*ldA];
    // update all previous b-values with current A column and current B
    __vmult1axpy(&X[0], incX, &Ac[(i-1)*ldA], &X[(i-1)*incX], 1, -1.0, i-1);
  }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a01*b0)/a11
 *  b2 = (b'2 - a02*b0 - a12*b1)/a22
 */
static inline
void __trsv_unb_lut(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;
  DTYPE xtmp;

  for (i = 0; i < nRE; i++) {
    xtmp = 0.0;
    __vmult1dot(&xtmp, 1, &Ac[i*ldA], &X[0], incX, 1.0, i);
    xtmp = X[i*incX] - xtmp;
    X[i*incX] = unit ? xtmp : xtmp/Ac[i+i*ldA];
  }
}

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
static inline
void __trsv_unb_ll(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;

  for (i = 0; i < nRE; i++) {
    X[i*incX] = unit ? X[i*incX] : X[i*incX]/Ac[i+i*ldA];
    // update all X-values below with the current A column and current X
    __vmult1axpy(&X[(i+1)*incX], incX, &Ac[(i+1)+i*ldA], &X[i*incX], incX, -1.0, nRE-i-1);
  }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = (b'0 - a10*b1 - a20*b2)/a00
 *  b1 =          (b'1 - a21*b2)/a11
 *  b2 =                     b'2/a22
 */
static inline
void __trsv_unb_llt(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int N)
{
  register int i;
  DTYPE xtmp;

  for (i = N; i > 0; i--) {
    xtmp = 0.0;
    __vmult1dot(&xtmp, 1, &Ac[i+(i-1)*ldA], &X[i*incX], incX, 1.0, N-i);
    xtmp = X[(i-1)*incX] - xtmp;
    X[(i-1)*incX] = unit ? xtmp : xtmp/Ac[(i-1)+(i-1)*ldA];
  }
}



static
void __trsv_unb(mvec_t *X, const mdata_t *A, int flags, int N)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
  case ARMAS_UPPER|ARMAS_TRANS:
    __trsv_unb_lut(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case ARMAS_UPPER:
    __trsv_unb_lu(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case ARMAS_LOWER|ARMAS_TRANS:
    __trsv_unb_llt(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case ARMAS_LOWER:
  default:
    __trsv_unb_ll(X->md, A->md, unit, X->inc, A->step, N);
    break;
  }
}

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *                                                     
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0           --> x0 = trsv(x'0, A00)
 *    x'1 = A01*x0 + A11*x1  --> x1 = trsv(x'1 - A01*x0)
 *  lower:
 *    x'0 = A00*x0           --> x0 = trsv(x'0, A00)
 *    x'1 = A10*x0 + A11*x1  --> x1 = trsv(x'1 - A10*x0, A11)
 *
 *   Forward substitution.
 */
static
void __trsv_forward_recursive(mvec_t *X, const mdata_t *A, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    return;
  }

  // top part
  __subvector(&x0, X, 0);
  __subblock(&a0, A, 0, 0);
  __trsv_forward_recursive(&x0, &a0, flags, N/2);

  // update bottom with top
  __subvector(&x1, X, N/2);
  if (flags & ARMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __gemv_recursive(&x1, &a1, &x0, -1.0, 1.0, flags, 0, N/2, 0, N-N/2);


  // bottom part
  __subblock(&a1, A, N/2, N/2);
  __trsv_forward_recursive(&x1, &a1, flags, N-N/2);
}

/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *                                                     
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0 + A01*x1  --> x0 = trsv(x'0 - A01*x1, A00)
 *    x'1 = A11*x1           --> x1 = trsv(x'1, A11)
 *  lower:
 *    x'0 = A00*x0 + A10*x1  --> x0 = trsv(x'0 - A10*x1, A00)
 *    x'1 = A11*x1           --> x1 = trsv(x'1, A11)
 *
 *   Backward substitution.
 */
static
void __trsv_backward_recursive(mvec_t *X, const mdata_t *A, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    return;
  }

  // bottom part
  __subvector(&x1, X, N/2);
  __subblock(&a1, A, N/2, N/2);
  __trsv_backward_recursive(&x1, &a1, flags, N-N/2);

  // update top with bottom
  __subvector(&x0, X, 0);
  if (flags & ARMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __gemv_recursive(&x0, &a0, &x1, -1.0, 1.0, flags, 0, N-N/2, 0, N/2);


  // top part
  __subblock(&a0, A, 0, 0);
  __trsv_backward_recursive(&x0, &a0, flags, N/2);
}

void __trsv_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    if (alpha != 1.0) {
      __vscale(X->md, X->inc, N, alpha);
    }
    return;
  }

  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
  case ARMAS_LOWER|ARMAS_TRANS:
  case ARMAS_UPPER:
    __trsv_backward_recursive(X, A, flags, N);
    break;

  case ARMAS_UPPER|ARMAS_TRANS:
  case ARMAS_LOWER:
  default:
    __trsv_forward_recursive(X, A, flags, N);
    break;
  }
  if (alpha != 1.0) {
    __vscale(X->md, X->inc, N, alpha);
  }
}


/**
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
int __armas_mvsolve_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
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

#if defined(__trsv_ext_unb)
  // if extended precision enabled and requested
  IF_EXPR(conf->optflags&ARMAS_OEXTPREC,
          __trsv_ext_unb(&x, &A0, alpha, flags, nx));
#endif
  // normal precision here
  switch (conf->optflags) {
  case ARMAS_RECURSIVE:

    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_UPPER:
      __trsv_backward_recursive(&x, &A0, flags, nx);
      break;
    case ARMAS_UPPER|ARMAS_TRANS:
    case ARMAS_LOWER:
    default:
      __trsv_forward_recursive(&x, &A0, flags, nx);
      break;
    }
    break;

  case ARMAS_SNAIVE:
  default:
    __trsv_unb(&x, &A0, flags, nx);
    break;
  }
  if (alpha != 1.0) {
    __vscale(x.md, x.inc, nx, alpha);
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
