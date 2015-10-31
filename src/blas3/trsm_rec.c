
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
#if defined(__solve_recursive) && defined(__solve_blk_recursive)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__solve_left_unb) && defined(__solve_right_unb) && \
  defined(__kernel_colwise_inner_no_scale)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

// Recursive versions of TRSM

/*
 *   RIGHT-UPPER             RIGHT-LOWER
 *                                                     
 *            A00 | A01               A00 |  0  
 *   B0|B1 * -----------     B0|B1 * -----------
 *             0  | A11               A10 | A11 
 *
 */

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *                                                     
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A01*B0 + A11*B1  --> B1 = trsm(B'1 - A01*B0)
 *  lower:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A10*B0 + A11*B1  --> B1 = trsm(B'1 - A10*B0, A11)
 *
 *   Forward substitution.
 */
static
void __solve_left_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;

  if (N < MIN_MBLOCK_SIZE) {
    __solve_left_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  __solve_left_forward(__subblock(&b0, B, 0, S),
                       __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);

  // update B[0:N/2, S:E] with B[N/2:N, S:E]
  if (flags & ARMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __subblock(&b1, B, N/2, S);
  __kernel_colwise_inner_no_scale(&b1, &a0, &b0, -1.0, flags,
                                  N/2, E-S, N-N/2, cache);
  
  __solve_left_forward(__subblock(&b0, B, N/2, S),
                       __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
}


/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *                                                     
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0 + A01*B1  --> B0 = A00.-1*(B'0 - A01*B1)
 *    B'1 = A11*B1           --> B1 = A11.-1*B'1
 *  lower:
 *    B'0 = A00*B0 + A10*B1  --> B0 = trsm(B'0 - A10*B1, A00)
 *    B'1 = A11*B1           --> B1 = trsm(B'1, A11)
 *
 *   Backward substitution.
 */
static
void __solve_left_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;

  if (N < MIN_MBLOCK_SIZE) {
    __solve_left_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  // UPPER and LOWER TRANS are backward subsition, 
  __solve_left_backward(__subblock(&b0, B, N/2, S),
                        __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);

  // update B[0:N/2, S:E] with B[N/2:N, S:E]
  if (flags & ARMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __subblock(&b1, B, 0, S);
  __kernel_colwise_inner_no_scale(&b1, &a0, &b0, -1.0, flags,
                                  N-N/2, E-S, N/2, cache);
    
  __solve_left_backward(__subblock(&b0, B, 0, S),
                        __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
}


/*
 * Forward substitution for RIGHT-UPPER, RIGHT-LOWER-TRANSA
 */
static
void __solve_right_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  int ar, ac, ops;

  if (N < MIN_MBLOCK_SIZE) {
    __solve_right_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  __solve_right_forward(__subblock(&b0, B, S, 0),
                        __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);

  ar = flags & ARMAS_UPPER ? 0   : N/2;
  ac = flags & ARMAS_UPPER ? N/2 : 0;

  __subblock(&a0, A, ar, ac);
  __subblock(&b1, B, S, N/2);

  ops = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;
  __kernel_colwise_inner_no_scale(&b1, &b0, &a0, -1.0, ops,
                                  N/2, N-N/2, E-S, cache);
  
  __solve_right_forward(__subblock(&b0, B, S,   N/2),
                        __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
}


/*
 * Backward substitution for RIGHT-UPPER-TRANSA and RIGHT-LOWER
 */
static
void __solve_right_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  int ops;

  if (N < MIN_MBLOCK_SIZE) {
    __solve_right_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  __solve_right_backward(__subblock(&b0, B, S,   N/2),
                         __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);

  if (flags & ARMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
    }
  __subblock(&b1, B, S, 0);
  ops = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;
  __kernel_colwise_inner_no_scale(&b1, &b0, &a0, -1.0, ops,
                                  N-N/2, N/2, E-S, cache);
    
  __solve_right_backward(__subblock(&b0, B, S, 0),
                         __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
}

void __solve_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  
  switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
  case ARMAS_RIGHT|ARMAS_UPPER:
  case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
    __solve_right_forward(B, A, alpha, flags, N, S, E, cache);
    break;
    
  case ARMAS_RIGHT|ARMAS_LOWER:
  case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
    __solve_right_backward(B, A, alpha, flags, N, S, E, cache);
    break;

  case ARMAS_UPPER:
  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_left_backward(B, A, alpha, flags, N, S, E, cache);
    break;

  case ARMAS_LOWER:
  case ARMAS_UPPER|ARMAS_TRANSA:
  default:
    __solve_left_forward(B, A, alpha, flags, N, S, E, cache);
    break;
  }
}

void __solve_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  cache_t mcache;

  armas_cache_setup2(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
  case ARMAS_RIGHT|ARMAS_UPPER:
  case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
    __solve_right_forward(B, A, alpha, flags, N, S, E, &mcache);
    break;
    
  case ARMAS_RIGHT|ARMAS_LOWER:
  case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
    __solve_right_backward(B, A, alpha, flags, N, S, E, &mcache);
    break;

  case ARMAS_UPPER:
  case ARMAS_LOWER|ARMAS_TRANSA:
    __solve_left_backward(B, A, alpha, flags, N, S, E, &mcache);
    break;

  case ARMAS_LOWER:
  case ARMAS_UPPER|ARMAS_TRANSA:
    __solve_left_forward(B, A, alpha, flags, N, S, E, &mcache);
    break;
  }
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
