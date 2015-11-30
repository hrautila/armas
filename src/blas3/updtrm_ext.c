
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_update_trm) && defined(__update_trm_blk)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_colwise_inner_no_scale)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"
#include "scheduler.h"

/*
 * update diagonal block
 *
 *  l00           a00 a01   b00 b01 b02    u00 u01 u02
 *  l10 l11       a10 a11   b10 b11 b12        u11 u12
 *  l20 l21 l22   a20 a21                          u22
 *
 */
 static
 void __update_ext_trm_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                            DTYPE alpha, DTYPE beta,
                            int flags,  int P, int nC, int nR, cache_t *cache)
{
  register int i, incA, incB;
  mdata_t A0, B0, C0;

  incA = flags & ARMAS_TRANSA ? A->step : 1;
  incB = flags & ARMAS_TRANSB ? 1 : B->step;

  __subblock(&A0, A, 0, 0);
  __subblock(&B0, B, 0, 0);

  if (flags & ARMAS_UPPER) {
    // index by row
    int M = min(nC, nR);
    for (i = 0; i < M; i++) {   
      // scale the target row with beta
      __subblock(&C0,   C, i, i);

      // update one row of C  (nC-i columns, 1 row)
      __kernel_ext_colwise_inner_scale_c(&C0, &A0, &B0, alpha, beta, flags,
                                         P, 0, nC-i, 0, 1, cache); 
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  } else {
    // index by column
    int N = min(nC, nR);
    for (i = 0; i < N; i++) {
      __subblock(&C0,   C, i, i);
      // update one column of C  (1 column, nR-i rows)
      __kernel_ext_colwise_inner_scale_c(&C0, &A0, &B0, alpha, beta, flags,
                                         P, 0, 1, 0, nR-i, cache);
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  }
}

void __update_ext_trm_naive(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            DTYPE alpha, DTYPE beta, int flags, int P, int S, int L, 
                            int R, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  cache_t mcache;

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  armas_cache_setup2(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  __update_ext_trm_diag(C, A, B, alpha, beta, flags, P, L-S, E-R, &mcache);
}


/*
 * Generic triangular matrix update:
 *      C = beta*op(C) + alpha*A*B
 *      C = beta*op(C) + alpha*A*B.T
 *      C = beta*op(C) + alpha*A.T*B
 *      C = beta*op(C) + alpha*A.T*B.T
 *
 * Some conditions on parameters that define the updated block:
 * 1. S == R && E == L 
 *    matrix is triangular square matrix
 * 2. S == R && L >  E
 *    matrix is trapezoidial with upper trapezoidial part right of triangular part
 * 3. S == R && L <  E
 *    matrix is trapezoidial with lower trapezoidial part below triangular part
 * 4. S != R && S >  E
 *    update is only to upper trapezoidial part right of triangular block
 * 5. S != R && R >  L
 *    update is only to lower trapezoidial part below triangular block
 * 6. S != R
 *    inconsistent update block spefication, will not do anything
 *            
 */
static
void __update_ext_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          DTYPE alpha, DTYPE beta, int flags,
                          int P, int S, int L, int R, int E, cache_t *cache)
{
  register int i, nI, ar, ac, br, bc, N, M;
  mdata_t Cd, Ad, Bd;

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  if ( S != R && (S <= E || R <= L)) {
    // inconsistent update configuration
    return;
  }

  if (flags & ARMAS_UPPER) {
    // by rows; M is the last row; L-S is column count; implicitely S == R
    M = min(L, E);
    for (i = R; i < M; i += cache->NB) {
      nI = M - i < cache->NB ? M - i : cache->NB;
    
      // 1. update block on diagonal (square block)
      br = flags & ARMAS_TRANSB ? i : 0;
      bc = flags & ARMAS_TRANSB ? 0 : i;
      ar = flags & ARMAS_TRANSA ? 0 : i;
      ac = flags & ARMAS_TRANSA ? i : 0;

      //printf("i=%dm nI=%d, L-i=%d, L-i-nI=%d\n", i, nI, L-i, L-i-nI);
      __subblock(&Cd, C, i,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_ext_trm_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, nI, cache);

      // 2. update right of the diagonal block (rectangle, nI rows)
      br = flags & ARMAS_TRANSB ? i+nI : 0;
      bc = flags & ARMAS_TRANSB ? 0    : i+nI;
      ar = flags & ARMAS_TRANSA ? 0    : i;
      ac = flags & ARMAS_TRANSA ? i    : 0;

      __subblock(&Cd, C, i,  i+nI);
      __subblock(&Ad, A, ar, ac);
      __subblock(&Bd, B, br, bc);
      __kernel_ext_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                         P, 0, L-i-nI, 0, nI, cache);

    }
   } else {
    // by columns; N is the last column, E-R is row count;
    N = min(L, E);
    for (i = S; i < N; i += cache->NB) {
      nI = N - i < cache->NB ? N - i : cache->NB;
    
      // 1. update on diagonal (square block)
      br = flags & ARMAS_TRANSB ? i : 0;
      bc = flags & ARMAS_TRANSB ? 0 : i;
      ar = flags & ARMAS_TRANSA ? 0 : i;
      ac = flags & ARMAS_TRANSA ? i : 0;
      __subblock(&Cd, C, i, i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_ext_trm_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, nI, cache);

      // 2. update block below the diagonal block (rectangle, nI columns)
      br = flags & ARMAS_TRANSB ? i    : 0;
      bc = flags & ARMAS_TRANSB ? 0    : i;
      ar = flags & ARMAS_TRANSA ? 0    : i+nI;
      ac = flags & ARMAS_TRANSA ? i+nI : 0;
      __subblock(&Cd, C, i+nI,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __kernel_ext_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                         P, 0, nI, 0, E-i-nI, cache);
    }
  }
}

void __update_ext_trm_blocked(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              DTYPE alpha, DTYPE beta, int flags, int P, int S, int L, 
                              int R, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  cache_t mcache;

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  armas_cache_setup3(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  __update_ext_trm_blk(C, A, B, alpha, beta, flags, P, S, L, R, E, &mcache);
}


#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
