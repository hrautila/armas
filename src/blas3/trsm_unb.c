
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
void __y1_sub0_dotax(DTYPE *y, int incy,
                     const DTYPE *a0, int inca,
                     const DTYPE *b, int incb, int nR)
{
  register int k;
  register DTYPE s0, c0, y0, t0;

  s0 = c0 = __ZERO;
  for (k = 0; k < nR; k += 1) {
    y0 = a0[(k+0)*inca]*b[(k+0)*incb] - c0;
    t0 = s0 + y0;
    c0 = (t0 - s0) - y0;
    s0 = t0;
  }
  y[0] -= (s0 + c0);
}

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

static inline
void __y2_sub_dotax(DTYPE *y, int incy, const DTYPE *a0, int inca,
                    const DTYPE *b0, const DTYPE *b1,
                    int incb, int nR)
{
  register int k;
  register DTYPE d0, d1, d2, d3, d4, d5, d6, d7;

  d0 = d1 = d2 = d3 = __ZERO;
  d4 = d5 = d6 = d7 = __ZERO;
  for (k = 0; k < nR-3; k += 4) {
    d0 += a0[(k+0)*inca]*b0[(k+0)*incb];
    d1 += a0[(k+1)*inca]*b0[(k+1)*incb];
    d2 += a0[(k+2)*inca]*b0[(k+2)*incb];
    d3 += a0[(k+3)*inca]*b0[(k+3)*incb];

    d4 += a0[(k+0)*inca]*b1[(k+0)*incb];
    d5 += a0[(k+1)*inca]*b1[(k+1)*incb];
    d6 += a0[(k+2)*inca]*b1[(k+2)*incb];
    d7 += a0[(k+3)*inca]*b1[(k+3)*incb];
  }
  if (k == nR)
    goto update;

  switch (nR-k) {
  case 3:
    d2 += a0[(k+0)*inca]*b0[(k+0)*incb];
    d6 += a0[(k+0)*inca]*b1[(k+0)*incb];
    k++;
  case 2:
    d1 += a0[(k+0)*inca]*b0[(k+0)*incb];
    d5 += a0[(k+0)*inca]*b1[(k+0)*incb];
    k++;
  case 1:
    d0 += a0[(k+0)*inca]*b0[(k+0)*incb];
    d4 += a0[(k+0)*inca]*b1[(k+0)*incb];
  }
 update:
  y[0] -= d0 + d1 + d2 + d3;
  y[incy] -= d4 + d5 + d6 + d7;
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
void __solve_unb_lu(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                  int ldB, int ldA, int nRE, int nB)
{
  // backward substitution
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  // update with A bottom-right element
  if (!unit) {
    for (j = 0; j < nB; j++) {
      Bc[(nRE-1)+j*ldB] /= Ac[(nRE-1)+(nRE-1)*ldA];
    }
  }

  // rest of the elements
  for (i = nRE-1; i > 0; i--) {
    for (j = 0; j < nB; j++) {
      // subtract from this row dot pwroduct of off-diagonal A and B elements below
      __y1_sub_dotax(&Bc[(i-1)+j*ldB], 1, &Ac[(i-1)+i*ldA], ldA, &Bc[i+j*ldB], 1, nRE-i);
      if (unit)
        continue;
      Bc[(i-1)+j*ldB] /= Ac[(i-1)+(i-1)*ldA];
    }
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
void __solve_unb_lut(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                     int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int i, j;

  if (!unit) {
    for (j = 0; j < nB; j++) {
      Bc[j*ldB] /= Ac[0];
    }
  }

  for (i = 1; i < nRE; i++) {
    for (j = 0; j < nB; j++) {
      __y1_sub_dotax(&Bc[i+j*ldB], 1, &Ac[i*ldA], 1, &Bc[j*ldB], 1, i);
      if (unit)
        continue;
      Bc[i+j*ldB] /= Ac[i+i*ldA];
    }
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
void __solve_unb_ll(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                    int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  // top-left
  if (! unit) {
    for (j = 0; j < nB; j++) {
      Bc[j*ldB] /= Ac[0];
    }
  }

  // rest of the elements
  for (i = 1; i < nRE; i++) {
    for (j = 0; j < nB; j++) {
      // substact from this row dot product of off-diagonal A and B elements above
      __y1_sub_dotax(&Bc[i+j*ldB], 1, &Ac[i], ldA, &Bc[j*ldB], 1, i);
      if (unit)
        continue;
      Bc[i+j*ldB] /= Ac[i+i*ldA];
    }
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
void __solve_unb_llt(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                     int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  if (!unit) {
    for (j = 0; j < nB; j++) {
      Bc[(nRE-1)+j*ldB] /= Ac[(nRE-1)+(nRE-1)*ldA];
    }
  }

  for (i = nRE-1; i > 0; i--) {
    for (j = 0; j < nB; j++) {
      __y1_sub_dotax(&Bc[(i-1)+j*ldB], 1, &Ac[i+(i-1)*ldA], 1, &Bc[i+j*ldB], 1, nRE-i);
      if (unit)
        continue;
      Bc[(i-1)+j*ldB] /= Ac[(i-1)+(i-1)*ldA];
    }
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
void __solve_unb_ru(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                    int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  register int i, j;

  if (!unit) {
    for (i = 0; i < nB; i++) {
      Bc[i] /= Ac[0];
    }
  }
  // nB is rows in B; nRE is rows/columns in A, cols in B
  for (j = 1; j < nRE; j++) {
    for (i = 0; i < nB; i++) {
      __y1_sub_dotax(&Bc[i+j*ldB], 1, &Ac[j*ldA], 1, &Bc[i], ldB, j);
      if (unit)
        continue;
      Bc[i+j*ldB] /= Ac[j+j*ldA];
    }
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
void __solve_unb_rut(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                     int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  // if not unit then update rightmost with bottom-right of A
  if (!unit) {
    for (i = 0; i < nB; i++) {
      Bc[i+(nRE-1)*ldB] /= Ac[(nRE-1)+(nRE-1)*ldA];
    }
  }
  for (j = nRE-1; j > 0; j--) {
    for (i = 0; i < nB; i++) {
      // update current B with dot-product of off-diagonal A and previous B
      __y1_sub_dotax(&Bc[i+(j-1)*ldB], 1, &Ac[(j-1)+j*ldA], ldA, &Bc[i+j*ldB], ldB, nRE-j);
      if (unit)
        continue;
      Bc[i+(j-1)*ldB] /= Ac[(j-1)+(j-1)*ldA];
    }
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
void __solve_unb_rl(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                    int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  if (!unit) {
    for (i = 0; i < nB; i++) {
      Bc[i+(nRE-1)*ldB] /= Ac[(nRE-1)+(nRE-1)*ldA];
    }
  }
  // backward along A diagonal from right to left
  for (j = nRE-1; j > 0; j--) {
    for (i = 0; i < nB; i++) {
      __y1_sub_dotax(&Bc[i+(j-1)*ldB], 1, &Ac[j+(j-1)*ldA], 1, &Bc[i+j*ldB], ldB, nRE-j);
      if (unit)
        continue;
      Bc[i+(j-1)*ldB] /= Ac[(j-1)+(j-1)*ldA];
    }
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
void __solve_unb_rlt(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int flags, 
                     int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  int unit = flags & ARMAS_UNIT ? 1 : 0;

  if (!unit) {
    // update left-most B with top-left A diagonal
    for (i = 0; i < nB; i++) {
      Bc[i] /= Ac[0];
    }
  }
  for (j = 1; j < nRE; j++) {
    for (i = 0; i < nB; i++) {
      // update current elemnet with off-diagonal A and preceeding B elements
      __y1_sub_dotax(&Bc[i+j*ldB], 1, &Ac[j], ldA, &Bc[i], ldB, j);

      Bc[i+j*ldB] = unit ? Bc[i+j*ldB] : Bc[i+j*ldB]/Ac[j+j*ldA];
    }
  }
}


void __solve_left_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E)
{
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_unb_lut(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_UPPER:
    __solve_unb_lu(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_unb_llt(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_LOWER:
  default:
    __solve_unb_ll(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;
  }
}


void __solve_right_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E)
{
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_unb_rut(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_UPPER:
    __solve_unb_ru(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_unb_rlt(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;

  case ARMAS_LOWER:
  default:
    __solve_unb_rl(B->md, A->md, 1.0, flags, B->step, A->step, N, E-S);
    break;
  }
}


#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
