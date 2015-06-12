
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__trmm_ext_blk)
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#if EXT_PRECISION && defined(__kernel_ext_panel_inner)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "internal.h"
#include "matrix.h"
#include "eft.h"
#include "kernel_ext.h"

// Functions here implement various versions of TRMM operation.


/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0     b0 = a00*b0 + a01*b1 + a02*b2
 *     0 |a11|a12  b1     b1 =          a11*b1 + a12*b2
 *     0 | 0 |a22  b2     b2 =                   a22*b2
 */
static void
__trmm_ext_unb_upper(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  // Y is 
  DTYPE s0, u0, p0, r0, c0;
  register int i, j, k;

  for (j = 0; j < nC; j++) {
    // update all previous B-values with current A column and current B
    for (i = 0; i < nRE; i++) {
      s0 = unit ? __get(B, i, j) : __ZERO;
      u0 = __ZERO;
      for (k = i+unit; k < nRE; k++) {
        twoprod(&p0, &r0, __get(A, i, k), __get(B, k, j));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, i, j, s0); //->md[i+j*B->step] = s0;
      __set(dB, i, j, u0+c0); //->md[i+j*dB->step] = u0 + c0;
    }
  }
}

/* LEFT-UPPER
 *
 *   B0    (A00 A01 A02) (B0)      B0 = A00*B0 + A01*B1 + A02*B2
 *   B1 =  ( 0  A11 A12) (B1)      B1 = A11*B1 + A12*B2         
 *   B2    ( 0   0  A22) (B2)      B2 = A22*B2                  
 */
static
void __trmm_ext_blk_upper(mdata_t *B, const mdata_t *A, const DTYPE alpha,
                          int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB, *Bc;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;
  Bc = cache->C0;
  dB = cache->dC;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    // off diagonal part
    __subblock(&A0, A, i, i+nI);
    // diagonal block
    __subblock(&A1, A, i, i);

    //printf("__trmm_blk_u: A1 [%d,%d], A0: [%d,%d], nI=%d\n", i, i, i, i+nI, nI);
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;
      __subblock(&B1, B, i, j);
      __subblock(&B0, B, i+nI, j);

      //__blk_scale_ext(Bc, dB, &B1, __ONE, nI, nJ);
      clear_blk(dB, nI, nJ);

      // update current part with diagonal
      __trmm_ext_unb_upper(&B1, dB, &A1, alpha, unit, nI, nJ);
      // update current part with rest of the A, B panels
      __kernel_ext_panel_inner(&B1, dB, &A0, &B0, alpha, 0, nJ, nI, N-i-nI, cache);

      ext_merge(&B1, dB, nI, nJ);
    }
  }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0    b0 = a00*b'0                    
 *  b1 =   0 |a11|a12  b'1    b1 = a01*b'0 + a11*b'1          
 *  b2     0 | 0 |a22  b'2    b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static void
__trmm_ext_unb_u_trans(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0;

  for (j = 0; j < nC; j++) {
    for (i = nRE-1; i >= 0; i--) {
      s0 = unit ? __get(B, i, j) : __ZERO;
      u0 = __ZERO;
      for (k = 0; k < i+(1-unit); k++) {
        //twoprod(&p0, &r0, A->md[k+i*A->step], B->md[k+j*B->step]);
        twoprod(&p0, &r0, __get(A, k, i), __get(B, k, j));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, i, j, s0); //->md[i+j*B->step] = s0;
      __set(dB, i, j, u0+c0); //->md[i+j*dB->step] = u0 + c0;
    }
  }
}

/*  LEFT-UPPER-TRANS
 *
 *  B0    (A00 A01 A02) (B0)        B0 = A00*B0                    
 *  B1 =  ( 0  A11 A12) (B1)        B1 = A01*B0 + A11*B1           
 *  B2    ( 0   0  A22) (B1)        B2 = A02*B0 + A12*B1 + A22*B2  
 */
static
void __trmm_ext_blk_u_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                            int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;
  dB = cache->dC;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, 0,    i-nI);
    __subblock(&A1, A, i-nI, i-nI);  // diagonal

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;
      __subblock(&B0, B, 0,    j);
      __subblock(&B1, B, i-nI, j);
      // update current part with diagonal
      __trmm_ext_unb_u_trans(&B1, dB, &A1, alpha, unit, nI, nJ);
      // update current part with rest of the A, B panels
      __kernel_ext_panel_inner(&B1, dB, &A0, &B0, alpha, ARMAS_TRANSA, nJ, nI, i-nI,
                               cache); 

      ext_merge(&B1, dB, nI, nJ);
    }
  }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0    b0 = a00*b'0                    
 *  b1 =  a10|a11| 0   b'1    b1 = a10*b'0 + a11*b'1          
 *  b2    a20|a21|a22  b'2    b2 = a20*b'0 + a21*b'1 + a22*b'2
 */
static void
__trmm_ext_unb_lower(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0, x0;

  // for all columns in B
  for (j = 0; j < nC; j++) {
    // all rows in column of B
    for (i = nRE-1; i >= 0; i--) {
      //s0 = unit ? B->md[i+j*B->step] : __ZERO;
      s0 = unit ? __get(B, i, j) : __ZERO;
      u0 = __ZERO;
      for (k = 0; k < i+(1-unit); k++) {
        //twoprod(&p0, &r0, A->md[i+k*A->step], B->md[k+j*B->step]);
        twoprod(&p0, &r0, __get(A, i, k), __get(B, k, j));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, i, j, s0); //->md[i+j*B->step] = s0;
      __set(dB, i,j, u0+c0); //->md[i+j*dB->step] = u0 + c0;
    }
  }
}

/*  LEFT-LOWER
 *
 *   B0     (A00  0   0 ) (B0)       B0 = A00*B0                  
 *   B1 =   (A10 A11  0 ) (B1)       B1 = A10*B0 + A11*B1         
 *   B2     (A20 A21 A21) (B1)       B2 = A20*B0 + A21*B1 + A22*B2
 */
static
void __trmm_ext_blk_lower(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int NB = cache->NB;
  dB = cache->dC;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, i-nI, 0);
    __subblock(&A1, A, i-nI, i-nI);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nI, nJ);

      __subblock(&B0, B, 0,    j);
      __subblock(&B1, B, i-nI, j);
      // update current part with diagonal
      __trmm_ext_unb_lower(&B1, dB, &A1, alpha, unit, nI, nJ);
      // update current part with rest of the A, B panels
      __kernel_ext_panel_inner(&B1, dB, &A0, &B0, alpha, 0, nJ, nI, i-nI, cache); 

      ext_merge(&B1, dB, nI, nJ);
    }
  }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0    b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =  a10|a11| 0   b'1    b1 =           a11*b'1 + a21*b'2
 *  b2    a20|a21|a22  b'2    b2 =                     a22*b'2
 *
 */
static void
__trmm_ext_unb_l_trans(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  DTYPE s0, u0, p0, r0, c0;
  register int i, j, k;

  for (j = 0; j < nC; j++) {
    // for column of B
    for (i = 0; i < nRE; i++) {
      s0 = unit ? __get(B, i, j) : __ZERO;
      u0 = __ZERO;
      for (k = i+unit; k < nRE; k++) {
        //twoprod(&p0, &r0, A->md[k+i*A->step], B->md[k+j*B->step]);
        twoprod(&p0, &r0, __get(A, k, i), __get(B, k, j));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      c0 = __ZERO;
      if (alpha != __ONE) {
        // if FMA enabled then should not test just execute without branching
        twoprod(&s0, &c0, alpha, s0);
        u0 *= alpha;
      }
      __set(B, i, j, s0); //->md[i+j*B->step] = s0;
      __set(dB, i, j, u0+c0); //->md[i+j*dB->step] = u0 + c0;
    }
  }
}

/*
 *  LEFT-LOWER-TRANSA
 *
 *   B0     (A00  0   0 ) (B0)     B0 = A00*B0 + A10*B1 + A20*B2  
 *   B1  =  (A10 A11  0 ) (B1)     B1 = A11*B1 + A21*B2           
 *   B2     (A20 A21 A22) (B2)     B2 = A22*B2                    
 */
static
void __trmm_ext_blk_l_trans(mdata_t *B, const mdata_t *A,
                            DTYPE alpha, int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int NB = cache->NB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  dB = cache->dC;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;

    __subblock(&A0, A, i,    i);
    __subblock(&A1, A, i+nI, i);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nI, nJ);

      __subblock(&B0, B, i,    j);
      __subblock(&B1, B, i+nI, j);
      
      // update current part with diagonal
      __trmm_ext_unb_l_trans(&B0, dB, &A0, alpha, unit, nI, nJ);
      // update current part with rest of the A, B panels
      __kernel_ext_panel_inner(&B0, dB, &A1, &B1, alpha, ARMAS_TRANSA,
                               nJ, nI, N-i-nI, cache); 

      ext_merge(&B0, dB, nI, nJ);
    }
  }
}

/*
 *  RIGHT-UPPER
 *  
 *                          a00|a01|a02    b0 = b'0*a00                    
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12    b1 = b'0*a01 + a11*b'1          
 *                           0 | 0 |a22    b2 = b'0*a02 + a12*b'1 + a22*b'2
 *
 */
static void
__trmm_ext_unb_r_upper(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  // Y is 
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0;

  for (j = nC-1; j >= 0; j--) {
    for (i = nRE-1; i >= 0; i--) {
      //s0 = unit ? B->md[j+i*B->step] : __ZERO;
      s0 = unit ? __get(B, j, i) : __ZERO;
      u0 = __ZERO;
      for (k = 0; k < i+(1-unit); k++) {
        //twoprod(&p0, &r0, A->md[k+i*A->step], B->md[j+k*B->step]);
        twoprod(&p0, &r0, __get(A, k, i), __get(B, j, k));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, j, i, s0); //->md[j+i*B->step] = s0;
      __set(dB, j, i, u0+c0); //->md[j+i*dB->step] = u0 + c0;
    }
  }
}

/*
 *  RIGHT-UPPER
 *
 *                              (A00 A01 A02)    B0 = B0*A00                   
 *   (B0 B1 B2) =  (B0 B1 B2) * ( 0  A11 A12)    B1 = B0*A01 + B1*A11          
 *                              ( 0   0  A22)    B2 = B0*A02 + B1*A12 + B2*A22 
 */
static
void __trmm_ext_blk_r_upper(mdata_t *B, const mdata_t *A, DTYPE alpha,
                            int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int NB = cache->NB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  dB = cache->dC;

  // rows/columns of A; columns of B; nI is number of columns
  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, 0,    i-nI);
    __subblock(&A1, A, i-nI, i-nI);
    
    // rows of B; nJ is number of rows
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nJ, nI);

      __subblock(&B0, B, j, 0);
      __subblock(&B1, B, j, i-nI);
      
      // update current part with diagonal
      __trmm_ext_unb_r_upper(&B1, dB, &A1, alpha, unit, nI, nJ);
      __kernel_ext_panel_inner(&B1, dB, &B0, &A0, alpha, 0, nI, nJ, i-nI, cache); 

      ext_merge(&B1, dB, nJ, nI);
    }
  }
}

/*
 * LOWER, RIGHT,
 *  
 *                          a00| 0 | 0     b0 = b'0*a00 + b'1*a10 + b'2*a20 
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0     b1 =           b'1*a11 + b'2*a21           
 *                          a20|a21|a22    b2 =                     b'2*a22                     
 */
static void
__trmm_ext_unb_r_lower(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0;

  for (j = 0; j < nC; j++) {
    for (i = 0; i < nRE; i++) {
      //s0 = unit ? B->md[j+i*B->step] : __ZERO;
      s0 = unit ? __get(B, j, i) : __ZERO;
      u0 = __ZERO;
      for (k = i+unit; k < nRE; k++) {
        //twoprod(&p0, &r0, A->md[k+i*A->step], B->md[j+k*B->step]);
        twoprod(&p0, &r0, __get(A, k, i), __get(B, j, k));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, j, i, s0); //->md[j+i*B->step] = s0;
      __set(dB, j, i, u0+c0); //->md[j+i*dB->step] = u0 + c0;
    }
  }
}


/*
 * RIGHT-LOWER
 *
 *                            (A00  0   0 )     B0 = B0*A00 + B1*A01 + B2*A02 
 *  (B0 B1 B2) = (B0 B1 B2) * (A01 A11  0 )     B1 = B1*A11 + B2*A12          
 *                            (A02 A12 A22)     B2 = B2*A22                   
 */
static
void __trmm_ext_blk_r_lower(mdata_t *B, const mdata_t *A, DTYPE alpha,
                            int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int NB = cache->NB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  dB = cache->dC;

  // rows/columns of A; columns of B; nI is number of columns
  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    __subblock(&A0, A, i,    i);
    __subblock(&A1, A, i+nI, i);

    // rows of B; nJ is number of rows
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nJ, nI);

      __subblock(&B0, B, j, i);
      __subblock(&B1, B, j, i+nI);
      
      // update current part with diagonal; left with diagonal
      __trmm_ext_unb_r_lower(&B0, dB, &A0, alpha, unit, nI, nJ);
      __kernel_ext_panel_inner(&B0, dB, &B1, &A1, alpha, 0, nI, nJ, N-i-nI, cache); 

      ext_merge(&B0, dB, nJ, nI);
    }
  }
}

/*
 *  RIGHT-UPPER-TRANS
 *  
 *                          a00|a01|a02    b0 = b'0*a00 + b'1*a01 + b'2*a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12    b1 =           b'1*a11 + b'2*a12
 *                           0 | 0 |a22    b2 =                     b'2*a22
 */
static void
__trmm_ext_unb_ru_trans(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0;

  // nC is rows of B
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nRE; i++) {
      //s0 = unit ? B->md[j+i*B->step] : __ZERO;
      s0 = unit ? __get(B, j, i) : __ZERO;
      u0 = __ZERO;
      for (k = i+unit; k < nRE; k++) {
        //twoprod(&p0, &r0, A->md[i+k*A->step], B->md[j+k*B->step]);
        twoprod(&p0, &r0, __get(A, i, k), __get(B, j, k));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, s0, alpha);
      u0 *= alpha;
      __set(B, j, i, s0); //->md[j+i*B->step] = s0;
      __set(dB, j, i, u0+c0); //->md[j+i*dB->step] = u0 + c0;
    }
  }
}

/*
 *  RIGHT-UPPER-TRANS 
 *
 *                             (A00 A01 A02)     B0 = B0*A00 + B1*A01 + B2*A02
 *   (B0 B1 B2) = (B0 B1 B2) * ( 0  A11 A12)     B1 = B1*A11 + B2*A12         
 *                             ( 0   0  A22)     B2 = B2*A22                  
 */
static
void __trmm_ext_blk_ru_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                             int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int NB = cache->NB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  dB = cache->dC;

  // rows/columns of A; columns of B; nI is number of columns
  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    __subblock(&A0, A, i, i);
    __subblock(&A1, A, i, i+nI);

    // rows of B; nJ is number of rows
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nJ, nI);

      __subblock(&B0, B, j, i);
      __subblock(&B1, B, j, i+nI);
      
      // update current part with diagonal; left with diagonal
      __trmm_ext_unb_ru_trans(&B0, dB, &A0, alpha, unit, nI, nJ);
      __kernel_ext_panel_inner(&B0, dB, &B1, &A1, alpha, ARMAS_TRANSB, nI, nJ, N-i-nI, cache); 
      ext_merge(&B0, dB, nJ, nI);
    }
  }
}

/*
 * LOWER, RIGHT, TRANSA
 *  
 *                          a00| 0 | 0      b0 = b'0*a00                    
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0      b1 = b'0*a10 + b'1*a11          
 *                          a20|a21|a22     b2 = b'0*a20 + b'1*a21 + b'2*a22
 */
static void
__trmm_ext_unb_rl_trans(mdata_t *B, mdata_t *dB, const mdata_t *A, DTYPE alpha, int unit, int nRE, int nC)
{
  register int i, j, k;
  DTYPE s0, u0, p0, r0, c0;

  // nC is rows of B
  for (j = nC-1; j >= 0; j--) {
    for (i = nRE-1; i >= 0; i--) {
      //s0 = unit ? B->md[j+i*B->step] : __ZERO;
      s0 = unit ? __get(B, j, i) : __ZERO;
      u0 = __ZERO;
      for (k = 0; k < i+(1-unit); k++) {
        //twoprod(&p0, &r0, A->md[i+k*A->step], B->md[j+k*B->step]);
        twoprod(&p0, &r0, __get(A, i, k), __get(B, j, k));
        twosum(&s0, &c0, s0, p0);
        u0 += c0 + r0;
      }
      twoprod(&s0, &c0, alpha, s0);
      u0 *= alpha;
      __set(B, j, i, s0); //->md[j+i*B->step] = s0;
      __set(dB, j, i, u0+c0); //->md[j+i*dB->step] = u0 + c0;
    }
  }
}

/*
 *  RIGHT-LOWER-TRANSA
 *
 *                            (A00  0   0 )     B0 = B0*A00                  
 *  (B0 B1 B2) = (B0 B1 B2) * (A01 A11  0 )     B1 = B0*A01 + B1*A11         
 *                            (A02 A12 A22)     B2 = B0*A02 + B1*A12 + B2*A22
 */
static
void __trmm_ext_blk_rl_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                             int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1, *dB;
  int NB = cache->NB;
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  dB = cache->dC;

  // for columns/rows of A; columns of B; nI is column count
  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;

    __subblock(&A0, A, i-nI, 0);
    __subblock(&A1, A, i-nI, i-nI);

    // rows of B; nJ is number of rows
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      clear_blk(dB, nJ, nI);

      __subblock(&B0, B, j, 0);
      __subblock(&B1, B, j, i-nI);
      
      // update current part with diagonal
      __trmm_ext_unb_rl_trans(&B1, dB, &A1, alpha, unit, nI, nJ);
      __kernel_ext_panel_inner(&B1, dB, &B0, &A0, alpha, ARMAS_TRANSB, nI, nJ, i-nI, cache); 

      ext_merge(&B1, dB, nJ, nI);
    }
  }
}


static
void __trmm_ext_unb(mdata_t *Bc, mdata_t *dB, const mdata_t *A, DTYPE alpha, int flags, int N, int S, int E)
{
  // indicates if diagonal entry is unit (=1.0) or non-unit.
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  
  if (flags & ARMAS_RIGHT) {
    clear_blk(dB, E-S, N);
    if (flags & ARMAS_UPPER) {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_unb_ru_trans(Bc, dB, A, alpha, unit, E-S, N);
      } else {
        __trmm_ext_unb_r_upper(Bc, dB, A, alpha, unit, E-S, N);
      }
    } else {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_unb_rl_trans(Bc, dB, A, alpha, unit, E-S, N);
      } else {
        __trmm_ext_unb_r_lower(Bc, dB, A, alpha, unit, E-S, N);
      }
    }
    ext_merge(Bc, dB, E-S, N);
  } else {
    clear_blk(dB, N, E-S);
    if (flags & ARMAS_UPPER) {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_unb_u_trans(Bc, dB, A, alpha, unit, N, E-S);
      } else {
        __trmm_ext_unb_upper(Bc, dB, A, alpha, unit, N, E-S);
      }
    } else {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_unb_l_trans(Bc, dB, A, alpha, unit, N, E-S);
      } else {
        __trmm_ext_unb_lower(Bc, dB, A, alpha, unit, N, E-S);
      }
    }
    ext_merge(Bc, dB, N, E-S);
  }
}



void __trmm_ext_blk(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                    int N, int S, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy, Cpy, Dcpy;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_NB/4], Bbuf[MAX_KB*MAX_NB/4] __attribute__((aligned(64)));
  DTYPE Cbuf[MAX_KB*MAX_NB/4], Dbuf[MAX_KB*MAX_NB/4] __attribute__((aligned(64)));
  if (E-S <= 0 || N <= 0)
    return;

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB/2 || NB <= 0) {
    NB = MAX_NB/2;
  }
  if (MB > MAX_NB/2 || MB <= 0) {
    MB = MAX_NB/2;
  }
  if (KB > MAX_KB/2 || KB <= 0) {
    KB = MAX_KB/2;
  }
  // should KB >= NB ??
  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  // setup cache area
  Acpy = (mdata_t){Abuf, MAX_KB/2};
  Bcpy = (mdata_t){Bbuf, MAX_KB/2};
  Cpy  = (mdata_t){Cbuf, MAX_KB/2};
  Dcpy = (mdata_t){Dbuf, MAX_KB/2};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB, &Cpy, &Dcpy};

  if (N < 2*NB) {
    // do only unblocked
    __trmm_ext_unb(B, &Dcpy, A, alpha, flags, N, S, E);
    return;
  }

  if (flags & ARMAS_RIGHT) {
    // B = alpha*B*op(A)
    if (flags & ARMAS_UPPER) {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_blk_ru_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_ext_blk_r_upper(B, A, alpha, flags, N, S, E, &cache); 
      }
    } else {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_blk_rl_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_ext_blk_r_lower(B, A, alpha, flags, N, S, E, &cache); 
      }
    }

  } else {
    // B = alpha*op(A)*B
    if (flags & ARMAS_UPPER) {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_blk_u_trans(B, A, alpha, flags, N, S, E, &cache); 
      } else {
        __trmm_ext_blk_upper(B, A, alpha, flags, N, S, E, &cache); 
      }
    } else {
      if (flags & ARMAS_TRANSA) {
        __trmm_ext_blk_l_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_ext_blk_lower(B, A, alpha, flags, N, S, E, &cache); 
      }
    }
  }
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
