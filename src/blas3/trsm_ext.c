
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__solve_ext_unb) && defined(__solve_ext_blk)
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#if EXT_PRECISION && defined(__kernel_ext_panel_inner)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"
#include "kernel_ext.h"

// alignment to 256 bit boundary (256/8)
#define __DTYPE_ALIGN_SIZE (32/sizeof(DTYPE))
#define __DTYPE_ALIGN_MASK (32/sizeof(DTYPE) - 1)

#define __ALIGNED(x)  ((x) & ~__DTYPE_ALIGN_MASK)

#define __ALIGNED_PTR(_T, _x)                                               \
  ((_T*)((unsigned long)( ((char *)(_x)) + __DTYPE_ALIGN_SIZE-1) & ~__DTYPE_ALIGN_MASK))

typedef struct cache_buf {
  DTYPE *data;  // pointer to data elements (aligned to 8bytes)
  int nelems;   // size in elements
  void *buf;    // allocated buffer; null if in-stack
  int nbytes;   // size in bytes
} cache_buf_t;

static inline
int cb_allocate(cache_buf_t *b, int nelems)
{
  b->buf = calloc(nelems+__DTYPE_ALIGN_SIZE, sizeof(DTYPE));
  if (!b->buf)
    return -1;
  b->data = __ALIGNED_PTR(DTYPE, b->buf);
  b->nbytes = (nelems+8)*sizeof(DTYPE);
  b->nelems = nelems;
  return 0;
}

static inline
void cb_release(cache_buf_t *b)
{
  if (b->buf) {
    free(b->buf);
  }
  b->data = (DTYPE *)0;
  b->buf  = (void *)0;
  b->nbytes = b->nelems = 0;
}

static inline
void cb_make(cache_buf_t *b, DTYPE *elems, int nelems)
{
  b->buf = (void *)0;
  b->nbytes = 0;
  b->data = elems;
  b->nelems = nelems;
}

// (1) Ph. Langlois, N. Louvet
//     Solving Triangular Systems More Accurately and Efficiently
//     

/*
 * Functions here solves the matrix equations
 *
 *   A*X   = alpha*B  --> X = alpha*A.-1*B
 *   A.T*X = alpha*B  --> X = alpha*A.-T*B
 *   X*A   = alpha*B  --> X = alpha*B*A.-1
 *   X*A.T = alpha*B  --> X = alpha*B*A.-T
 */

/*
 *   LEFT-UPPER
 *
 *     (b0)     (a00 a01 a02)   (b'0)
 *     (b1)  =  ( 0  a11 a12) * (b'1)
 *     (b2)     ( 0   0  a22)   (b'2)
 *
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *
 */
static
void __solve_ext_lu(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                    int unit, int nRE, int nB)
{
  // backward substitution
  register int i, j, k;
  DTYPE p0, q0, s0, r0, c0, u0, w0;

  // assume dB holds valid initial cumulative values
  for (j = 0; j < nB; j++) {
    u0 = __ZERO;
    for (i = nRE-1; i >= 0; i--) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = i+1; k < nRE; k++) {
        twoprod(&p0, &q0, __get(Ac, i, k), __get(Bc, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(Ac, i, k)*__get(dB, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, i, i));
      __set(Bc, i, j, p0);
      u0 = q0 + u0/__get(Ac, i, i);
      __set(dB, i, j, u0);
    }
  }
}

/*
 * LEFT-UPPER
 *   (B0)   (A00 A01 A02)   (B'0)
 *   (B1) = ( 0  A11 A12) * (B'1)
 *   (B2)   ( 0   0  A22)   (B'2)
 *
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = A00.-1*(B0 - A01*B'1 - A02*B'2)
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = A11.-1*(B1 - A12*B'2)
 *    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 *
 *  blocked computation for panel of size N*nES.
 */
static
void __solve_ext_blk_lu(mdata_t *B, mdata_t *dB, const mdata_t *A, const  DTYPE alpha,
                        int flags, int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ, nA, nB;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB; 

    // off-diagonal block (right side)
    __subblock(&A0, A, cI,  cI+nI);
    // diagonal
    __subblock(&A1, A, cI, cI);

    // old block; bottom N-cI-nI rows; starting at row cI+nI
    __subblock(&B0,  B,  cI+nI,  0);
    __subblock(&dB0, dB, cI+nI,  0);
    // current block; on diagonal rows; nI rows; nJ columns
    __subblock(&B1,  B,  cI, 0);
    __subblock(&dB1, dB, cI, 0);

    // scale initial value alpha*B1
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nI, nES);
    } else {
      clear_blk(&dB1, nI, nES);
    }

    // update with old solution; B0 holds the scaled solution
    __kernel_ext_panel_inner_dB(&B1, &dB1, &A0, &B0, &dB0,
                                -1.0, 0, nES, nI, N-cI-nI, cache);
    // solve current block in extended precision
    __solve_ext_lu(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, N, nES);
}


/*
 *    LEFT-UPPER-TRANS
 *
 *     (b0)     (a00 a01 a02)   (b'0)
 *     (b1)  =  ( 0  a11 a12) * (b'1)
 *     (b2)     ( 0   0  a22)   (b'2)
 *
 *   b0 = a00*b'0                     --> b'0 =  b0/a00
 *   b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *   b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 */
static
void __solve_ext_lut(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                     int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE p0, q0, s0, r0, c0, u0;

  for (j = 0; j < nB; j++) {
    u0 = __ZERO;
    for (i = 0; i < nRE; i++) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = 0; k < i; k++) {
        twoprod(&p0, &q0, __get(Ac, k, i), __get(Bc, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(Ac, k, i)*__get(dB, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, i, i));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, i, i)));
    }
  }
}

/*
 *   LEFT-UPPER-TRANS
 *    B0   (A00 A01 A02)   (B'0)
 *    B1 = ( 0  A11 A12) * (B'1)
 *    B2   ( 0   0  A22)   (B'2)
 *
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A01*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A01*B'0)
 *    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A02*B'0 - A12*B'1)
 *
 *  Solve one panel of size N,nES
 */
static
void __solve_ext_blk_lut(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha,
                        int flags,  int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;

    // off-diagonal block (top row, column cI)
    __subblock(&A0, A, 0, cI);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    __subblock(&B0,  B,  0, 0);
    __subblock(&dB0, dB, 0, 0);
    // current block
    __subblock(&B1,  B,  cI, 0);
    __subblock(&dB1, dB, cI, 0);

    if (alpha != __ONE) {
      // scale initial value alpha*B1
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nI, nES);
    } else {
      clear_blk(&dB1, nI, nES);
    }
    // update with old solution; B0 holds the scaled solution
    __kernel_ext_panel_inner_dB(&B1, &dB1, &A0, &B0, &dB0,
                                -1.0, ARMAS_TRANSA, nES, nI, i, cache);
    // solve diagonal block
    __solve_ext_lut(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, N, nES);
}

/*
 *    LEFT-LOWER
 *
 *     (b0)     (a00  0   0 )   (b'0)
 *     (b1)  =  (a10 a11  0 ) * (b'1)
 *     (b2)     (a20 a21 a22)   (b'2)
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *    b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 */
static
void __solve_ext_ll(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                    int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE p0, q0, s0, r0, c0, u0;

  // rest of the elements
  for (j = 0; j < nB; j++) {
    u0 = __ZERO;
    for (i = 0; i < nRE; i++) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = 0; k < i; k++) {
        twoprod(&p0, &q0, __get(Ac, i, k), __get(Bc, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(Ac, i, k)*__get(dB, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, i, i));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, i, i)));
    }
  }
}

/*
 *   LEFT-LOWER
 *    B0   (A00  0   0 )   (B'0)
 *    B1 = (A10 A11  0 ) * (B'1)
 *    B2   (A20 A21 A22)   (B'2)
 *
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A10*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A10*B'0)
 *    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A20*B'0 - A21*B'1)
 */
static
void __solve_ext_blk_ll(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha,
                        int flags,  int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;

    // off-diagonal block
    __subblock(&A0, A, cI, 0);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    // top block
    __subblock(&B0,   B, 0, 0);
    __subblock(&dB0, dB, 0, 0);
    // current block
    __subblock(&B1,   B, cI, 0);
    __subblock(&dB1, dB, cI, 0);
    // scale initial value alpha*B1
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nI, nES);
    } else {
      clear_blk(&dB1, nI, nES);
    }
    // update with old solution; B0 holds the scaled solution
    __kernel_ext_panel_inner_dB(&B1, &dB1, &A0, &B0, &dB0,
                                -1.0, 0, nES, nI, i, cache);
    // solve diagonal block
    __solve_ext_ll(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, N, nES);
}

/*
 * LEFT-LOWER-TRANS
 *     (b0)     (a00  0   0 )   (b'0)
 *     (b1)  =  (a10 a11  0 ) * (b'1)
 *     (b2)     (a20 a21 a22)   (b'2)
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1 - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_ext_llt(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                     int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE s0, u0, p0, q0, r0;

  for (j = 0; j < nB; j++) {
    u0 = __ZERO;
    for (i = nRE-1; i >= 0; i--) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = i+1; k < nRE; k++) {
        twoprod(&p0, &q0, __get(Ac, k, i), __get(Bc, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(Ac, k, i)*__get(dB, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, i, i));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, i, i)));
    }
  }
}

/*
 * LEFT-LOWER-TRANSA  
 *   (B0)    (A00  0   0 )   (B'0)
 *   (B1) =  (A10 A11  0 ) * (B'1)
 *   (B2)    (A20 A21 A22)   (B'2)
 *
 *    B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = A00.-1*(B0 - A10*B'1 - A20*B'2)
 *    B1 = A11*B'1 + A21*B'2           --> B'1 = A11.-1*(B1 - A21*B'2)
 *    B2 = A22*B'0                     --> B'2 = A22.-1*B2
 */
static
void __solve_ext_blk_llt(mdata_t *B, mdata_t *dB, const mdata_t *A, const  DTYPE alpha,
                         int flags, int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ, nA, nB;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;

    // off-diagonal block
    __subblock(&A0, A, cI, 0);
    // diagonal
    __subblock(&A1, A, cI, cI);

    // previous blocks below current block
    __subblock(&B0,   B, cI+nI,  0);
    __subblock(&dB0, dB, cI+nI,  0);
    // current block; on diagonal rows; nI rows; nJ columns
    __subblock(&B1,   B, cI, 0);
    __subblock(&dB1, dB, cI, 0);

    // scale current block
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nI, nES);
    } else {
      clear_blk(&dB1, nI, nES);
    }
    
    // update current solution with old solutions
    __kernel_ext_panel_inner_dB(&B1, &dB1, &A0, &B0, &dB0,
                                -1.0, ARMAS_TRANSA, nI, nJ, N-cI-nI, cache); 
    // solve diagonal block
    __solve_ext_llt(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, N, nES);
}

/*
 *    RIGHT-UPPER
 *                                (a00 a01 a02)
 *    (b0 b1 b2) =  (b'0 b'1 b'2) ( 0  a11 a12) 
 *                                ( 0   0  a22) 
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *    b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 */
static
void __solve_ext_ru(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                    int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE s0, u0, p0, q0, r0;

  // nB is rows in B; nRE is rows/columns in A, cols in B
  for (i = 0; i < nB; i++) {
    u0 = __ZERO;
    for (j = 0; j < nRE; j++) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = 0; k < j; k++) {
        twoprod(&p0, &q0, __get(Bc, i, k), __get(Ac, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(dB, i, k)*__get(Ac, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, j, j));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, j, j)));
    }
  }
}

/*
 *    RIGHT-UPPER
 *                                (A00 A01 A02)
 *    (B0 B1 B2) =  (B'0 B'1 B'2) ( 0  A11 A12) 
 *                                ( 0   0  A22) 
 *
 *    B0 = A00*B'0                     --> B'0 =  B0*A00.-1
 *    B1 = A01*B'0 + A11*B'1           --> B'1 = (B1 - A01*B'0)*A11.-1
 *    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = (B2 - A02*B'0 - A12*B'1)*A22.-1
 */
static
void __solve_ext_blk_ru(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha,
                        int flags,  int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;

    // off-diagonal block (top row,  column cI)
    __subblock(&A0, A, 0, cI);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    // left most block
    __subblock(&B0,  B,  0, 0);
    __subblock(&dB0, dB, 0, 0);
    // current block
    __subblock(&B1,  B,  0, cI);
    __subblock(&dB1, dB, 0, cI);

    if (alpha != __ONE) {
      // scale initial value alpha*B1
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nES, nI);
    } else {
      clear_blk(&dB1, nES, nI);
    }
    // update with old solution; B0 holds the scaled solution
    __kernel_ext_panel_inner_dA(&B1, &dB1, &B0, &dB0, &A0,
                                -1.0, 0, nI, nES, cI, cache);
    // solve diagonal block
    __solve_ext_ru(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, nES, N);
}

/*
 *    RIGHT-UPPER-TRANS
 *
 *                                (a00 a01 a02)
 *    (b0 b1 b2) =  (b'0 b'1 b'2) ( 0  a11 a12) 
 *                                ( 0   0  a22) 
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_ext_rut(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                     int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE s0, u0, p0, q0, r0;

  // nB (i) rows in B, nRE (j) rows/cols in A, cols in B
  for (i = 0; i < nB; i++) {
    u0 = __ZERO;
    for (j = nRE-1; j >= 0; j--) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = j+1; k < nRE; k++) {
        twoprod(&p0, &q0, __get(Bc, i, k), __get(Ac, j, k));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(dB, i, k)*__get(Ac, j, k);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, j, j));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, j, j)));
    }
  }
}

/*
 *    RIGHT-UPPER-TRANS
 *                                (A00 A01 A02)
 *    (B0 B1 B2) =  (B'0 B'1 B'2) ( 0  A11 A12) 
 *                                ( 0   0  A22) 
 *
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = (B0 - A01*B'1 - A02*B'2)*A00.-1
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = (B1 - A12*B'2)*A11.-1
 *    B2 = A22*B'2                     --> B'2 =  B2*A22.-1
 */
static
void __solve_ext_blk_rut(mdata_t *B, mdata_t *dB, const mdata_t *A, const  DTYPE alpha,
                        int flags, int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ, nA, nB;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB; 

    // off-diagonal block (right side)
    __subblock(&A0, A, cI,  cI+nI);
    // diagonal
    __subblock(&A1, A, cI, cI);

    // old block; bottom N-cI-nI rows; starting at row cI+nI
    __subblock(&B0,  B,  0, cI+nI);
    __subblock(&dB0, dB, 0, cI+nI);
    // current block; on diagonal rows; nI rows; nJ columns
    __subblock(&B1,  B,  0, cI);
    __subblock(&dB1, dB, 0, cI);

    // scale initial value alpha*B1
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nES, nI);
    } else {
      clear_blk(&dB1, nES, nI);
    }

    // update with old solution; B0 holds the scaled solution
    __kernel_ext_panel_inner_dA(&B1, &dB1, &B0, &dB0, &A0,
                                -1.0, ARMAS_TRANSB, nI, nES, N-cI-nI, cache);
    // solve current block in extended precision
    __solve_ext_rut(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, nES, N);
}

/*
 *    RIGHT-LOWER
 *                                (a00  0  :  0 )  
 *    (b0 b1 b2) =  (b'0 b'1 b'2) (a10 a11 :  0 )
 *                                (a20 a21 : a22) 
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1 - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void __solve_ext_rl(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                    int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE s0, u0, p0, q0, r0;
  // backward along A diagonal from right to left
  for (i = 0; i < nB; i++) {
    u0 = __ZERO;
    for (j = nRE-1; j >= 0; j--) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = j+1; k < nRE; k++) {
        twoprod(&p0, &q0, __get(Bc, i, k), __get(Ac, k, j));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(dB, i, k)*__get(Ac, k, j);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, j, j));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, j, j)));
    }
  }
}

/*
 *    RIGHT-LOWER
 *                                (A00  0   0 )  
 *    (B0 B1 B2) =  (B'0 B'1 B'2) (A10 A11  0 )
 *                                (A20 A21 A22) 
 *
 *    B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = (B0 - A10*B'1 - A20*B'2)*A00.-1
 *    B1 = A11*B'1 + A21*B'2           --> B'1 = (B1 - A21*B'2)*A11.-1
 *    B2 = A22*B'2                     --> B'2 =  B2*A22.-1
 */
static
void __solve_ext_blk_rl(mdata_t *B, mdata_t *dB, const mdata_t *A, const  DTYPE alpha,
                        int flags, int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ, nA, nB;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;

    // off-diagonal block
    __subblock(&A0, A, cI+nI, cI);
    // diagonal
    __subblock(&A1, A, cI, cI);

    // previous blocks left current block
    __subblock(&B0,   B, 0, cI+nI);
    __subblock(&dB0, dB, 0, cI+nI);
    // current block; 
    __subblock(&B1,   B, 0, cI);
    __subblock(&dB1, dB, 0, cI);

    // scale current block
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nES, nI);
    } else {
      clear_blk(&dB1, nES, nI);
    }
    
    // update current solution with old solutions
    __kernel_ext_panel_inner_dA(&B1, &dB1, &B0, &dB0, &A0,
                                -1.0, 0, nI, nES, N-cI-nI, cache); 
    // solve diagonal block
    __solve_ext_rl(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, nES, N);
}

/*
 *    RIGHT-LOWER-TRANS
 *                                (a00  0   0 )
 *    (b0 b1 b2) =  (b'0 b'1 b'2) (a10 a11  0 )
 *                                (a20 a21 a22) 
 *
 *    b0 = a00*b'0                     --> b'0 = b0/a00
 *    b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *    b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 *
 *  (nRE columns, nB rows)
 */
static
void __solve_ext_rlt(mdata_t *Bc, mdata_t *dB, const mdata_t *Ac,
                     int unit, int nRE, int nB)
{
  register int i, j, k;
  DTYPE s0, u0, p0, q0, r0;

  for (i = 0; i < nB; i++) {
    u0 = __ZERO;
    for (j = 0; j < nRE; j++) {
      s0  = __get(Bc, i, j);
      u0 += __get(dB, i, j);
      for (k = 0; k < j; k++) {
        twoprod(&p0, &q0, __get(Bc, i, k), __get(Ac, j, k));
        twosum(&s0, &r0, -p0, s0);
        u0 += (r0 - q0) - __get(dB, i, k)*__get(Ac, j, k);
      }
      if (unit) {
        __set(Bc, i, j, s0);
        __set(dB, i, j, u0);
        continue;
      }
      approx_twodiv(&p0, &q0, s0, __get(Ac, j, j));
      __set(Bc, i, j, p0);
      __set(dB, i, j, (q0 + u0/__get(Ac, j, j)));
    }
  }
}

/*
 *    RIGHT-LOWER-TRANS
 *                                (A00  0   0 )
 *    (B0 B1 B2) =  (B'0 B'1 B'2) (A10 A11  0 )
 *                                (A20 A21 A22) 
 *
 *    B0 = A00*B'0                     --> B'0 = B0*A00.-1
 *    B1 = A10*B'0 + A11*B'1           --> B'1 = (B1 - A10*B'0)*A11.-1
 *    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = (B2 - A20*B'0 - A21*B'1)*A22.-1
 */
static
void __solve_ext_blk_rlt(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha,
                        int flags,  int N, int nES, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1, dB0, dB1;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;

    // off-diagonal block; row panel
    __subblock(&A0, A, i, 0);
    // diagonal block
    __subblock(&A1, A, i, i);

    // leftmost block
    __subblock(&B0,   B, 0, 0);
    __subblock(&dB0, dB, 0, 0);
    // current block
    __subblock(&B1,   B, 0, i);
    __subblock(&dB1, dB, 0, i);
    // scale initial value alpha*B1
    if (alpha != __ONE) {
      __blk_scale_ext(&B1, &dB1, &B1, alpha, nES, nI);
    } else {
      clear_blk(&dB1, nES, nI);
    }
    // update with old solution; B0 holds the scaled solution (nES rows, nI columns)
    __kernel_ext_panel_inner_dA(&B1, &dB1, &B0, &dB0, &A0, 
                                -1.0, ARMAS_TRANSB, nI, nES, i, cache);
    // solve diagonal block
    __solve_ext_rlt(&B1, &dB1, &A1, unit, nI, nES);
  }
  // merge solution
  ext_merge(B, dB, nES, N);
}


static
void __solve_ext_unb(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha,
                     int flags, int N, int nES, cache_t *cache)
{
  mdata_t B0, dB0;
  int NB = cache->NB;
  int right = flags & ARMAS_RIGHT;
  int unit = flags & ARMAS_UNIT;
  int nJ, cJ, ib, nR, nC;

  // nES (columns/rows in B) may be larger than N 
  for (ib = 0; ib < nES; ib += NB) {
    nJ = ib < nES - NB ? NB : nES - ib;
    cJ = nJ < NB ? nES - nJ : ib;
    
    if (right) {
      __subblock(&B0, B, cJ, 0);
      __subblock(&dB0, dB, cJ, 0);
      nR = nJ; nC = N;
    } else {
      __subblock(&B0, B, 0,  cJ);
      __subblock(&dB0, dB, 0,  cJ);
      nR = N; nC = nJ;
    }
    if (alpha != __ONE) {
      __blk_scale_ext(&B0, &dB0, &B0, alpha, nR, nC);
    } else {
      clear_blk(&dB0, nR, nC);
    }

    // solve column or row panel;
    switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
      __solve_ext_rut(&B0, dB, A, unit, N, nJ);
      break;

    case ARMAS_UPPER|ARMAS_TRANSA:
      __solve_ext_lut(&B0, &dB0, A, unit, N, nJ);
      break;

    case ARMAS_RIGHT|ARMAS_UPPER:
      __solve_ext_ru(&B0, &dB0, A, unit, N, nJ);
      break;

    case ARMAS_UPPER:
      __solve_ext_lu(&B0, &dB0, A, unit, N, nJ);
      break;

    case ARMAS_RIGHT|ARMAS_TRANSA:
      __solve_ext_rlt(&B0, &dB0, A, unit, N, nJ);
      break;

    case ARMAS_TRANSA:
      __solve_ext_llt(&B0, &dB0, A, unit, N, nJ);
      break;

    case ARMAS_RIGHT:
      __solve_ext_rl(&B0, &dB0, A, unit, N, nJ);
      break;

    default: // LEFT, LOWER
      __solve_ext_ll(&B0, &dB0, A, unit, N, nJ);
      break;
    }
    ext_merge(&B0, &dB0, nR, nC);
  }
}

static
void __solve_ext_blk(mdata_t *B, const mdata_t *A, DTYPE alpha,
                     int flags, int N, int S, int E, cache_t *cache)
{
  int ib, nJ, cJ, nR, nC;
  mdata_t B0;
  mdata_t *dB = cache->dC;
  int right = flags & ARMAS_RIGHT;
  int NB = cache->NB;

  // nES (columns/rows in B) may be larger than N 
  for (ib = S; ib < E; ib += NB) {
    nJ = ib < E - NB ? NB : E - ib;
    cJ = nJ < NB ? E - nJ : ib;
    
    if (right) {
      __subblock(&B0, B, ib, 0);
      nR = nJ; nC = N;
    } else {
      __subblock(&B0, B, 0,  ib);
      nR = N; nC = nJ;
    }

    // solve column or row panel;
    switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
      __solve_ext_blk_rut(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_UPPER|ARMAS_TRANSA:
      __solve_ext_blk_lut(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_RIGHT|ARMAS_UPPER:
      __solve_ext_blk_ru(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_UPPER:
      __solve_ext_blk_lu(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_RIGHT|ARMAS_TRANSA:
      __solve_ext_blk_rlt(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_TRANSA:
      __solve_ext_blk_llt(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    case ARMAS_RIGHT:
      __solve_ext_blk_rl(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;

    default: // LEFT, LOWER
      __solve_ext_blk_ll(&B0, dB, A, alpha, flags, N, nJ, cache);
      break;
    }
  }
}

int __solve_ext(mdata_t *B, const mdata_t *A, DTYPE alpha,
                int flags, int N, int S, int E, int KB, int NB, int MB, int optflags)
{
  DTYPE *dptr = (DTYPE *)0;
  mdata_t *dB, B0;
  mdata_t Aa, Ba, Ca, dC;
  cache_t cache;
  int nRE, nB;
  DTYPE /*Cbuf[MAX_MB*MAX_NB/4],*/ Dbuf[MAX_NB*MAX_NB/4] __attribute__((aligned(64)));
  DTYPE Abuf[MAX_NB*MAX_NB/4], Bbuf[MAX_NB*MAX_NB/4] __attribute__((aligned(64)));
  int right = flags & ARMAS_RIGHT ? 1 : 0;

  if (E-S <= 0) {
    // nothing to do, zero columns or rows
    return 0;
  }

  // restrict block sizes as data is copied to aligned buffers of
  // predefined max sizes.
  if (NB > MAX_NB/2 || NB <= 0) {
    NB = __ALIGNED(MAX_NB/2);
  }
  if (MB > MAX_MB/2 || MB <= 0) {
    MB = __ALIGNED(MAX_MB/2);
  }
  if (KB  > MAX_KB/2 || KB <= 0) {
    KB = __ALIGNED(MAX_KB/2);
  }
  if (NB < __DTYPE_ALIGN_SIZE)
    NB = __DTYPE_ALIGN_SIZE;
  if (MB < __DTYPE_ALIGN_SIZE)
    MB = __DTYPE_ALIGN_SIZE;
  if (KB < __DTYPE_ALIGN_SIZE)
    KB = __DTYPE_ALIGN_SIZE;

  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));
  // setup cache area; Aa and Ba for single precision operators
  Aa = (mdata_t){Abuf, MAX_NB/2};
  Ba = (mdata_t){Bbuf, MAX_NB/2};


  // we need working space N*NB elements for error terms
  //if (N*NB > sizeof(Dbuf)/sizeof(DTYPE)) {
    long nelem = N*NB + __DTYPE_ALIGN_SIZE;
    dptr = calloc(nelem, sizeof(DTYPE));
    dC = (mdata_t){__ALIGNED_PTR(DTYPE, dptr), (right ? NB : N)};
#if 0
  } else {
    dC = (mdata_t){Dbuf, (right ? NB : N)};
  }
#endif

  //Ca = (mdata_t){Cbuf, MAX_MB/2};
  cache = (cache_t){&Aa, &Ba, NB, NB, NB, (mdata_t *)0/*&Ca*/, &dC};

  if (flags & ARMAS_RIGHT) {
    __subblock(&B0, B, S, 0);
    nRE = E-S; nB = N;
  } else {
    __subblock(&B0, B, 0, S);
      nRE = N; nB = E-S;
  }

  if (optflags & ARMAS_ONAIVE || N <= NB) {
    // small problem; solve with unblocked code
    __solve_ext_unb(&B0, &dC, A, alpha, flags, N, E-S, &cache);
  } else {
    __solve_ext_blk(B, A, alpha, flags, N, S, E, &cache);
  }

  // if allocated memory release it
  if (dptr) {
    free(dptr);
  }
  return 0;
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
