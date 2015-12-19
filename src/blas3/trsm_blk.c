
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__solve_blocked)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_colwise_inner_no_scale) && defined(__solve_blk_recursive)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"


/*
 * UPPER, LEFT                 LOWER, TRANSA, LEFT   
 *                                                   
 *    A00 | A01 | A02   B0       A00 |  0  |  0    B0
 *   ----------------   --      ----------------   --
 *     0  | A11 | A12   B1       A10 | A11 |  0    B1
 *   ----------------   --      ----------------   --
 *     0  |  0  | A22   B2       A20 | A21 | A22   B2
 *
 * upper:
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = A00.-1*(B0 - A01*B'1 - A02*B'2)
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = A11.-1*(B1           - A12*B'2)
 *    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 * lower:
 *    c0*B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = A00.-1*(c0*B0 - A10*B'1 - A20*B'2)
 *    c0*B1 = A11*B'1 + A21*B'2           --> B'1 = A11.-1*(c0*B1           - A21*B'2)
 *    c0*B2 = A22*B'2                     --> B'2 = A22.-1*c0*B2
 */
static
void __solve_blk_lu_llt(mdata_t *B, const mdata_t *A, const  DTYPE alpha, int flags,
                        int N, int S, int E, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;

    // off-diagonal block
    if (flags & ARMAS_UPPER)
      __subblock(&A0, A, 0,  cI);
    else
      __subblock(&A0, A, cI, 0);
    // diagonal
    __subblock(&A1, A, cI, cI);

    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;

      // top block; cI-nI rows
      __subblock(&B0, B, 0,  cJ);
      // bottom block; on diagonal rows; nI rows; nJ columns
      __subblock(&B1, B, cI, cJ);

      // solve bottom block
      __solve_blk_recursive(&B1, &A1, 1.0, flags, nI, 0, nJ, cache);
      // update top with bottom solution; 
      __kernel_colwise_inner_no_scale(&B0, &A0, &B1, -1.0, flags, nI, nJ, cI, cache); 
      // scale current block
      if (alpha != __ONE)
        __blk_scale(&B1, alpha, nI, nJ);
    }
  }
}

/*
 * LEFT-UPPER-TRANS              LEFT-LOWER
 *                                                     
 *    A00 | A01 | A02   B0         A00 |  0  |  0    B0
 *   ----------------   --        ----------------   --
 *     0  | A11 | A12   B1         A10 | A11 |  0    B1
 *   ----------------   --        ----------------   --
 *     0  |  0  | A22   B2         A20 | A21 | A22   B2
 *
 * upper:
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A01*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A01*B'0)
 *    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A02*B'0 - A12*B'1)
 * lower:
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A10*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A10*B'0)
 *    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A20*B'0 - A21*B'1)
 */
static
void __solve_blk_lut_ll(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                        int N, int S, int E,  cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;

    // off-diagonal block
    if (flags & ARMAS_UPPER)
      __subblock(&A0, A, 0,  cI);
    else
      __subblock(&A0, A, cI, 0);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;
      // top block
      __subblock(&B0, B, 0, cJ);
      __subblock(&B1, B, cI, cJ);
      // scale current block
      if (alpha != __ONE)
        __blk_scale(&B1, alpha, nI, nJ);
      // update block with old solutions
      __kernel_colwise_inner_no_scale(&B1, &A0, &B0, -1.0, flags, i, nJ, nI, cache);
      // solve diagonal block
      __solve_blk_recursive(&B1, &A1, 1.0, flags, nI, 0, nJ, cache);
    }
  }
}

/*
 *  RIGHT-UPPER                        LOWER, RIGHT, TRANSA           
 *                                                                    
 *                 A00 | A01 | A02                    A00 |  0  |  0  
 *                ----------------                   ---------------- 
 *    B0|B1|B2      0  | A11 | A12       B0|B1|B2     A10 | A11 |  0  
 *                ----------------                   ---------------- 
 *                  0  |  0  | A22                    A20 | A21 | A22 
 *
 * upper:
 *    B0 = B'0*A00                     --> B'0 = B'0*A00.-1
 *    B1 = B'0*A01 + B'1*A11           --> B'1 = (B1 - B'0*A01)*A11.-1
 *    B2 = B'0*A02 + B'1*A12 + B'2*A22 --> B'2 = (B2 - B'0*A02 - B'1*A12)*A22.-1
 * lower:
 *    B0 = B'0*A00                     --> B'0 = B'0*A00.-1
 *    B1 = B'0*A10 + B'1*A11           --> B'1 = (B1 - B'0*A10)*A11.-1
 *    B2 = B'0*A20 + B'1*A21 + B'2*A22 --> B'2 = (B2 - B'0*A20 - B'1*A21)*A22.-1
 */
static
void __solve_blk_ru_rlt(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                        int N, int S, int E, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;
  int transB = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;
    // off-diagonal
    if (flags & ARMAS_UPPER)
      __subblock(&A0, A, 0,  cI);
    else
      __subblock(&A0, A, cI, 0);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    // for B rows
    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;
      __subblock(&B0, B, cJ, 0);
      // block on diagonal columns
      __subblock(&B1, B, cJ, cI);
      // scale current block
      if (alpha != __ONE)
        __blk_scale(&B1, alpha, nI, nJ);
      // update block with old solutions
      __kernel_colwise_inner_no_scale(&B1, &B0, &A0, -1.0, transB,
                                      cI, nI, nJ, cache);
      // solve on diagonal 
      __solve_blk_recursive(&B1, &A1, 1.0, flags, nI, 0, nJ, cache);
    }
  }
}

/*
 *   RIGHT-UPPER-TRANSA                RIGHT-LOWER
 *                                                                   
 *                 A00 | A01 | A02                    A00 |  0  |  0 
 *                ----------------                   ----------------
 *    B0|B1|B2      0  | A11 | A12       B0|B1|B2     A10 | A11 |  0 
 *                ----------------                   ----------------
 *                  0  |  0  | A22                    A20 | A21 | A22
 *
 *  upper:
 *    B0 = B'0*A00 + B'1*A01 + B'2*A02 --> B'0 = (B0 - B'1*A01 - B'2*A02)*A00.-1
 *    B1 = B'1*A11 + B'2*A12           --> B'1 = (B1           - B'2*A12)*A11.-1
 *    B2 = B'2*A22                     --> B'2 = B2*A22.-1
 *  lower:
 *    B0 = B'0*A00 + B'1*A10 + B'2*A20 --> B'0 = (B0 - B'1*A10 - B'2*A20)*A00.-1
 *    B1 = B'1*A11 + B'2*A21           --> B'1 = (B1 - B'2*A21)*A11.-1
 *    B2 = B'2*A22                     --> B'2 = B2*A22.-1
 */
static
void __solve_blk_rut_rl(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                        int N, int S, int E, cache_t *cache)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;
  int transB = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;
    // off-diagonal block
    if (flags & ARMAS_UPPER)
      __subblock(&A0, A, cI, cI+nI);
    else
      __subblock(&A0, A, cI+nI, cI);
    // diagonal block
    __subblock(&A1, A, cI, cI);

    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;

      __subblock(&B0, B, cJ, cI+nI);
      __subblock(&B1, B, cJ, cI);
      // scale current block
      if (alpha != __ONE)
        __blk_scale(&B1, alpha, nI, nJ);
      // update solution
      __kernel_colwise_inner_no_scale(&B1, &B0, &A0, -1.0, transB,
                                      N-i, nI, nJ, cache); 
      // solve diagonal
      __solve_blk_recursive(&B1, &A1, 1.0, flags, nI, 0, nJ, cache);
    }
  }
}

void __solve_blocked(mdata_t *B, const mdata_t *A, DTYPE alpha,
                     int flags, int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  cache_t mcache;

  armas_cache_setup2(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
  case ARMAS_RIGHT|ARMAS_UPPER:
  case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
    __solve_blk_ru_rlt(B, A, alpha, flags, N, S, E, &mcache);
    break;
    
  case ARMAS_RIGHT|ARMAS_LOWER:
  case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
    __solve_blk_rut_rl(B, A, alpha, flags, N, S, E, &mcache);
    break;

  case ARMAS_UPPER:
  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_blk_lu_llt(B, A, alpha, flags, N, S, E, &mcache);
    break;

  case ARMAS_LOWER:
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_blk_lut_ll(B, A, alpha, flags, N, S, E, &mcache);
    break;
  }
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
