
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__solve_left_unb) && defined(__solve_right_unb)
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"


static inline
void __y1_sub_dotax(DTYPE *y, int incy,
                    const DTYPE *a0, int inca,
                    const DTYPE *b, int incb, int nR)
{
  register int k;
  register DTYPE d0, d1, d2, d3;

  d0 = d1 = d2 = d3 = __ZERO;
  for (k = 0; k < nR-3; k += 4) {
    d0 += a0[(k+0)*inca]*b[(k+0)*incb];
    d1 += a0[(k+1)*inca]*b[(k+1)*incb];
    d2 += a0[(k+2)*inca]*b[(k+2)*incb];
    d3 += a0[(k+3)*inca]*b[(k+3)*incb];
  }
  if (k == nR)
    goto update;

  switch (nR-k) {
  case 3:
    d2 += a0[(k+0)*inca]*b[(k+0)*incb];
    k++;
  case 2:
    d1 += a0[(k+0)*inca]*b[(k+0)*incb];
    k++;
  case 1:
    d0 += a0[(k+0)*inca]*b[(k+0)*incb];
  }
 update:
  y[0] -= (d0 + d1) + (d2 + d3);
}


/*
 * Functions here solves the matrix equations
 *
 *   op(A)*X = alpha*B or X*op(A) = alpha*B
 */

/*
 *   LEFT-UPPER
 *
 *     b0     a00 | a01 : a02     b'0
 *     ==     ===============     ====
 *     b1  =   0  | a11 : a12  *  b'1
 *     --     ---------------     ----
 *     b2      0  |  0  : a22     b'2
 *
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a12*b'2)/a00
 *
 */
static
void __solve_unblk_lu(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B2;
  mvec_t a1, b1;
  
  if (!unit) {
    __rowvec(&b1, B, nRE-1, 0);
    __vec_scale(&b1, nB, __ONE/__get(A, nRE-1, nRE-1));
  }
  for (j = nRE-2; j >= 0; j--) {
    __subblock(&B2, B, j+1, 0);
    __rowvec(&b1, B, j, 0);
    __rowvec(&a1, A, j, j+1);
    // b1 = (b1 - B2.T*a1) / a11
    __gemv(&b1, &B2, &a1, -__ONE, ARMAS_TRANS, nRE-1-j, nB);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}


/*
 *    LEFT-UPPER-TRANS
 *
 *     b0     a00 | a01 : a02     b'0
 *     ==     ===============     ====
 *     b1  =   0  | a11 : a12  *  b'1
 *     --     ---------------     ----
 *     b2      0  |  0  : a22     b'2
 *
 *   b0 = a00*b'0                     --> b'0 =  b0/a00
 *   b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *   b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 */
static
void __solve_unblk_lut(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B0;
  mvec_t a1, b1;
  
  __subblock(&B0, B, 0, 0);
  for (j = 0; j < nRE; j++) {
    // j'th column
    __rowvec(&b1, B, j, 0);
    __colvec(&a1, A, 0, j);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B0, &a1, -__ONE, ARMAS_TRANS, j, nB);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}
/*
 *    LEFT-LOWER
 *
 *     b0     a00 |  0  :  0      b'0
 *     ==     ===============     ====
 *     b1  =  a10 | a11 :  0   *  b'1
 *     --     ---------------     ----
 *     b2     a20 | a12 : a22     b'2
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *    b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 */
static
void __solve_unblk_ll(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B0;
  mvec_t a1, b1;
  
  __subblock(&B0, B, 0, 0);
  for (j = 0; j < nRE; j++) {
    // j'th column
    __rowvec(&b1, B, j, 0);
    __rowvec(&a1, A, j, 0);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B0, &a1, -__ONE, ARMAS_TRANS, j, nB);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}
/*
 *   LEFT-LOWER-TRANS
 *
 *     b0     a00 |  0  :  0      b'0
 *     ==     ===============     ====
 *     b1  =  a10 | a11 :  0   *  b'1
 *     --     ---------------     ----
 *     b2     a20 | a12 : a22     b'2
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_unblk_llt(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B2;
  mvec_t a1, b1;
  
  if (!unit) {
    __rowvec(&b1, B, nRE-1, 0);
    __vec_scale(&b1, nB, __ONE/__get(A, nRE-1, nRE-1));
  }
  for (j = nRE-2; j >= 0; j--) {
    __subblock(&B2, B, j+1, 0);
    __rowvec(&b1, B, j,   0);
    __colvec(&a1, A, j+1, j);
    // b1 = (b1 - B2.T*a1) / a11
    __gemv(&b1, &B2, &a1, -__ONE, ARMAS_TRANS, nRE-1-j, nB);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}
/*
 *    RIGHT-UPPER
 *
 *                               a00 | a01 : a02  
 *                               ===============  
 *    b0|b1|b2 =  b'0|b'1|b'2 *   0  | a11 : a12  
 *                               ---------------  
 *                                0  |  0  : a22  
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *    b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 *
 */
static
void __solve_unblk_ru(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B0;
  mvec_t a1, b1;
  
  __subblock(&B0, B, 0, 0);
  for (j = 0; j < nRE; j++) {
    // j'th column
    __colvec(&b1, B, 0, j);
    __colvec(&a1, A, 0, j);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B0, &a1, -__ONE, 0, nB, j);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}


/*
 *    RIGHT-UPPER-TRANS
 *
 *                               a00 | a01 : a02  
 *                               ===============  
 *    b0|b1|b2 =  b'0|b'1|b'2 *   0  | a11 : a12  
 *                               ---------------  
 *                                0  |  0  : a22  
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1           - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_unblk_rut(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B2;
  mvec_t a1, b1;
  
  if (!unit) {
    __colvec(&b1, B, 0, nRE-1);
    __vec_scale(&b1, nB, __ONE/__get(A, nRE-1, nRE-1));
  }
  for (j = nRE-2; j >= 0; j--) {
    // j'th column
    __subblock(&B2, B, 0, j+1);
    __colvec(&b1, B, 0, j);
    __rowvec(&a1, A, j, j+1);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B2, &a1, -__ONE, 0, nB, nRE-1-j);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}
/*
 *    RIGHT-LOWER
 *                               a00 |  0  :  0  
 *                               ===============  
 *    b0|b1|b2 =  b'0|b'1|b'2 *  a10 | a11 :  0
 *                               ---------------  
 *                               a20 | a21 : a22  
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1           - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_unblk_rl(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B2;
  mvec_t a1, b1;
  
  if (!unit) {
    __colvec(&b1, B, 0, nRE-1);
    __vec_scale(&b1, nB, __ONE/__get(A, nRE-1, nRE-1));
  }
  for (j = nRE-2; j >= 0; j--) {
    // j'th column
    __subblock(&B2, B, 0, j+1);
    __colvec(&b1, B, 0, j);
    __colvec(&a1, A, j+1, j);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B2, &a1, -__ONE, 0, nB, nRE-1-j);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}

/*
 *    RIGHT-LOWER-TRANS
 *                               a00 |  0  :  0  
 *                               ===============  
 *    b0|b1|b2 =  b'0|b'1|b'2 *  a10 | a11 :  0
 *                               ---------------  
 *                               a20 | a12 : a22  
 *
 *    b00 = a00*b'00                       --> b'00 = b00/a00
 *    b01 = a10*b'00 + a11*b'01            --> b'01 = (b01 - a10*b'00)/a11
 *    b02 = a20*b'00 + a21*b'01 + a22*b'02 --> b'02 = (b02 - a20*b'00 - a21*b'01)/a22
 */
static
void __solve_unblk_rlt(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int j;
  mdata_t B0;
  mvec_t a1, b1;
  
  __subblock(&B0, B, 0, 0);
  for (j = 0; j < nRE; j++) {
    // j'th column
    __colvec(&b1, B, 0, j);
    __rowvec(&a1, A, j, 0);
    // b1 = (b1 - B0*a1) / a11
    __gemv(&b1, &B0, &a1, -__ONE, 0, nB, j);
    if (!unit)
      __vec_scale(&b1, nB, __ONE/__get(A, j, j));
  }
}


void __solve_left_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E)
{
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_unblk_lut(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_UPPER:
    __solve_unblk_lu(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_unblk_llt(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_LOWER:
  default:
    __solve_unblk_ll(B, A, alpha, flags, N, E-S);
    break;
  }
}


void __solve_right_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E)
{
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_unblk_rut(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_UPPER:
    __solve_unblk_ru(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_unblk_rlt(B, A, alpha, flags, N, E-S);
    break;

  case ARMAS_LOWER:
  default:
    __solve_unblk_rl(B, A, alpha, flags, N, E-S);
    break;
  }
}


#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
