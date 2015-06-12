
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__kernel_symm_ext) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if EXT_PRECISION && defined(__kernel_ext_colblk_inner) && defined(__kernel_ext_panel_inner)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "scheduler.h"
#include "matrix.h"
#include "matcpy.h"

extern void __kernel_ext_panel_inner(mdata_t *C, mdata_t *dC,
                                     const mdata_t *A, const mdata_t *B,
                                     DTYPE alpha, int nJ, int nR, int nP, cache_t *cache);

// C += A*B; A is the diagonal block
static
void __mult_symm_ext_diag(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          DTYPE alpha, int flags, 
                          int nP, int nSL, int nRE, cache_t *cache)
{
  int unit = flags & ARMAS_UNIT ? 1 : 0;
  int nA, nB, nAC;

  if (nP == 0)
    return;

  nAC = flags & ARMAS_RIGHT ? nSL : nRE;
  
  if (flags & ARMAS_LOWER) {
    // upper part of source untouchable, copy diagonal block and fill upper part
    //colcpy_fill_up(Acpy->md, Acpy->step, A->md, A->step, nAC, nAC, unit);
    __CPTRIL_UFILL(cache->Acpy, A, nAC, nAC, unit);
  } else {
    // lower part of source untouchable, copy diagonal block and fill lower part
    //colcpy_fill_low(Acpy->md, Acpy->step, A->md, A->step, nAC, nAC, unit);
    __CPTRIU_LFILL(cache->Acpy, A, nAC, nAC, unit);
  }

  if (flags & ARMAS_RIGHT) {
    //__CPTRANS(Bcpy->md, Bcpy->step, B->md, B->step, nRE, nSL);
    __CPBLK_TRANS(cache->Bcpy, B, nRE, nSL);
  } else {
    //__CP(Bcpy->md, Bcpy->step, B->md, B->step, nRE, nSL);
    __CPBLK(cache->Bcpy, B, nRE, nSL);
  }

  if (flags & ARMAS_RIGHT) {
    __kernel_colblk_inner(C, cache->Bcpy, cache->Acpy, alpha, nAC, nRE, nP);
  } else {
    __kernel_colblk_inner(C, cache->Acpy, cache->Bcpy, alpha, nSL, nAC, nP);
  }
}



static
void __kernel_ext_symm_left(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            DTYPE alpha, DTYPE beta, int flags,
                            int P, int S, int L, int R, int E,
                            int KB, int NB, int MB)
{
  int i, j, nI, nJ, flags1, flags2;
  mdata_t A0, B0, C0, Acpy, Bcpy, Cpy, dC;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB/4], Bbuf[MAX_KB*MAX_NB/4] __attribute__((aligned(64)));
  DTYPE Cbuf[MAX_MB*MAX_NB/4], Dbuf[MAX_MB*MAX_NB/4] __attribute__((aligned(64)));

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  if (NB > MAX_NB/2 || NB <= 0) {
    NB = MAX_NB/2;
  }
  if (MB > MAX_MB/2 || MB <= 0) {
    MB = MAX_MB/2;
  }
  if (KB  > MAX_KB/2 || KB <= 0) {
    KB = MAX_KB/2;
  }

  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  // setup cache area
  Acpy = (mdata_t){Abuf, MAX_KB/2};
  Bcpy = (mdata_t){Bbuf, MAX_KB/2};
  Cpy  = (mdata_t){Cbuf, MAX_MB/2};
  dC   = (mdata_t){Dbuf, MAX_MB/2};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB, &Cpy, &dC};

  flags1 = 0;
  flags2 = 0;

  /*
   * P is A, B common dimension, e.g. P cols in A and P rows in B.
   *
   * [R,R] [E,E] define block on A diagonal that divides A in three blocks
   * if A is upper:
   *   A0 [0, R] [R, E]; B0 [0, S] [R, L] (R rows,cols in P); (A transposed)
   *   A1 [R, R] [E, E]; B1 [R, S] [E, L] (E-R rows,cols in P)
   *   A2 [R, E] [E, N]; B2 [E, S] [N, L] (N-E rows, cols in  P)
   * if A is LOWER:
   *   A0 [R, 0] [E, R]; B0 [0, S] [R, L]
   *   A1 [R, R] [E, E]; B1 [R, S] [E, L] (diagonal block, fill_up);
   *   A2 [E, R] [E, N]; B2 [E, S] [N, L] (A transpose)
   *    
   * C = A0*B0 + A1*B1 + A2*B2
   */
  flags1 |= flags & ARMAS_UPPER ? ARMAS_TRANSA : 0;
  flags2 |= flags & ARMAS_LOWER ? ARMAS_TRANSA : 0;

  for (i = R; i < E; i += MB) {
    nI = E - i < MB ? E - i : MB;

    // for all column of C, B ...
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;
      __subblock(&C0, C, i, j);

      // block of C upper left at [i,j], lower right at [i+nI, j+nj]
      __blk_scale_ext(&Cpy, &dC, &C0, beta, nI, nJ);

      // 1. off diagonal block in A; if UPPER then above [i,j]; if LOWER then left of [i,j]
      //    above|left diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? 0 : i), (flags&ARMAS_UPPER ? i : 0));
      __subblock(&B0, B, 0, j);

      __kernel_ext_panel_inner(&Cpy, &dC, &A0, &B0, alpha, flags1, i, nJ, nI, &cache);
      //__kernel_ext_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags1, i, nJ, nI, &cache);

      // 2. on-diagonal block in A;
      __subblock(&A0, A, i, i);
      __subblock(&B0, B, i, j);
      __mult_symm_ext_diag(&C0, &A0, &B0, alpha, flags, nI, nJ, nI, &cache);

      // 3. off-diagonal block in A; if UPPER then right of [i, i+nI];
      //    if LOWER then below [i+nI, i]

      // right|below of diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? i : i+nI), (flags&ARMAS_UPPER ? i+nI : i));
      __subblock(&B0, B, i+nI, j);
      __kernel_ext_panel_inner(&Cpy, &dC, &A0, &B0, alpha, flags2, P-i-nI, nJ, nI, &cache);
      //__kernel_ext_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags2, P-i-nI, nJ, nI, &cache); 

      // merge to original 
      __blk_merge_ext(&C0, &Cpy, &dC, nI, nJ);
    }

  }
}


static
void __kernel_ext_symm_right(mdata_t *C, const mdata_t *A, const mdata_t *B,
                             DTYPE alpha, DTYPE beta, int flags,
                             int P, int S, int L, int R, int E,
                             int KB, int NB, int MB)
{
  int flags1, flags2;
  register int nR, nC, ic, ir;
  mdata_t A0, B0, C0, Acpy, Bcpy, Cpy, dC;
  cache_t cache;

  DTYPE Abuf[MAX_KB*MAX_MB/4], Bbuf[MAX_KB*MAX_NB/4] __attribute__((aligned(64)));
  DTYPE Cbuf[MAX_MB*MAX_NB/4], Dbuf[MAX_MB*MAX_NB/4] __attribute__((aligned(64)));

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  if (NB > MAX_NB/2 || NB <= 0) {
    NB = MAX_NB/2;
  }
  if (MB > MAX_MB/2 || MB <= 0) {
    MB = MAX_MB/2;
  }
  if (KB  > MAX_KB/2 || KB <= 0) {
    KB = MAX_KB/2;
  }

  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  // setup cache area
  Acpy = (mdata_t){Abuf, MAX_KB/2};
  Bcpy = (mdata_t){Bbuf, MAX_KB/2};
  Cpy  = (mdata_t){Cbuf, MAX_MB/2};
  dC   = (mdata_t){Dbuf, MAX_MB/2};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB, &Cpy, &dC};

  flags1 = 0;
  flags2 = 0;

  /*
   * P is A, B common dimension, e.g. P cols in A and P rows in B.
   * 
   * C = B * A;
   * [S,S] [L,L] define block on A diagonal that divides A in three blocks
   * if A is upper:
   *   A0 [0, S] [S, S]; B0 [R, 0] [E, S] (R rows,cols in P); (A transposed)
   *   A1 [S, S] [L, L]; B1 [R, S] [E, L] (E-R rows,cols in P)
   *   A2 [S, L] [L, N]; B2 [R, L] [E, N] (N-E rows, cols in  P)
   * if A is LOWER:
   *   A0 [S, 0] [S, S]; B0 [R, 0] [E, S]
   *   A1 [S, S] [L, L]; B1 [R, S] [E, L] (diagonal block, fill_up);
   *   A2 [L, S] [N, L]; B2 [R, L] [E, N] (A transpose)
   *
   * C = A0*B0 + A1*B1 + A2*B2
   */

  flags1 = flags & ARMAS_TRANSB ? ARMAS_TRANSA : 0;
  flags2 = flags & ARMAS_TRANSB ? ARMAS_TRANSA : 0;
  
  flags1 |= flags & ARMAS_LOWER ? ARMAS_TRANSB : 0;
  flags2 |= flags & ARMAS_UPPER ? ARMAS_TRANSB : 0;

  for (ic = S; ic < L; ic += NB) {
    nC = L - ic < NB ? L - ic : NB;

    // for all rows of C, B ...
    for (ir = R; ir < E; ir += MB) {
      nR = E - ir < MB ? E - ir : MB;

      __subblock(&C0, C, ir, ic);
      __blk_scale_ext(&Cpy, &dC, &C0, beta, nI, nJ);

      // above|left diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? 0 : ic), (flags&ARMAS_UPPER ? ic : 0));
      __subblock(&B0, B, ir, 0);
      __kernel_ext_panel_inner(&Cpy, &dC, &A0, &B0, alpha, flags1, ic, nC, nR, &cache);
      //__kernel_ext_colwise_inner_no_scale(&C0, &B0, &A0, alpha, flags1, ic, nC, nR, &cache);

      // diagonal block
      __subblock(&A0, A, ic, ic);
      __subblock(&B0, B, ir, ic);
      __mult_ext_symm_diag(&C0, &A0, &B0, alpha, flags, nC, nC, nR, &cache); 

      // right|below of diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? ic : ic+nC), (flags&ARMAS_UPPER ? ic+nC : ic));
      __subblock(&B0, B, ir, ic+nC);
      __kernel_ext_panel_inner(&Cpy, &dC, &A0, &B0, alpha, flags2, P-ic-nC, nC, nR, &cache);
      //__kernel_ext_colwise_inner_no_scale(&C0, &B0, &A0, alpha, flags2, P-ic-nC, nC, nR, &cache);

      // merge to original 
      __blk_merge_ext(&C0, &Cpy, &dC, nI, nJ);
    }
  }

}


int __kernel_ext_symm(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       DTYPE alpha, DTYPE beta, int flags,
                       int P, int S, int L, int R, int E,
                       int KB, int NB, int MB)
{
  if (flags & ARMAS_RIGHT) {
    __kernel_ext_symm_right(C, A, B, alpha, beta, flags, P, S, L, R, E, KB, NB, MB);
  } else {
    __kernel_ext_symm_left(C, A, B, alpha, beta, flags, P, S, L, R, E, KB, NB, MB);
  }
  return 0;
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
