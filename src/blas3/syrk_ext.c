
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <pthread.h>
#include <string.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_update_sym) && defined(__rank_diag)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_ext_colwise_inner_scale_c)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

extern
void __update_ext_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          DTYPE alpha, DTYPE beta, int flags,
                          int P, int S, int L, int R, int E, cache_t *cache);

void
__rank_ext_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                DTYPE alpha, DTYPE beta,  int flags,  int P, int nC, cache_t *cache)
{
  register int i, incA, trans;
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {B->md, B->step};

  incA = flags & ARMAS_TRANSA ? A->step : 1;
  trans = flags & ARMAS_TRANSA ? ARMAS_TRANSA : ARMAS_TRANSB;

  if (flags & ARMAS_UPPER) {
    for (i = 0; i < nC; i++) {
      // update one row of C  (nC-i columns, 1 row)
      __kernel_ext_colwise_inner_scale_c(C, &A0, &B0, alpha, beta, trans,
                                         P, 0, nC-i, 0, 1, cache); 
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incA; 
    }
  } else {
    for (i = 0; i < nC; i++) {
      // update one row of C  (nC-i columns, 1 row)
      __kernel_ext_colwise_inner_scale_c(C, &A0, &B0, alpha, beta, trans,
                                         P, 0, i+1, 0, 1, cache);
      // move to next row of C
      C->md ++;
      // move A to next row
      A0.md += incA;
    }
  }
}

/*
 * Symmetric rank update
 *
 * upper
 *   C00 C01 C02    C00  C01 C02     A0  
 *    0  C11 C12 =   0   C11 C12  +  A1 * B0 B1 B2
 *    0   0  C22     0   0   C22     A2
 *
 * lower:
 *   C00  0   0    C00   0   0      A0  
 *   C10 C11  0 =  C10  C11  0   +  A1 * B0 B1 B2
 *   C20 C21 C22   C20  C21 C22     A2

 */
void __rank_ext_blk(mdata_t *C, const mdata_t *A, DTYPE alpha, DTYPE beta,
                    int flags,  int P, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  register int i, nI, nC;
  mdata_t Cd, Ad, Bd;
  cache_t mcache;

  if (E-S <= 0 || P <= 0) {
    return;
  }

  armas_cache_setup3(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  if (flags & ARMAS_TRANSA) {
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);

      // 1. update on diagonal
      __rank_ext_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &mcache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, 0, i);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, 0, S);
        nC = i;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Bd, A, 0, i+nI);
        nC = E - i - nI;
      }

      __kernel_ext_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSA,
                                         P, 0, nC, 0, nI, &mcache); 
    }
  } else {
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __rank_ext_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &mcache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, i, 0);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, A, i+nI, 0);
        nC = E - i - nI;
      }

      __kernel_ext_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSB,
                                         P, 0, nC, 0, nI, &mcache);
    }
  }
}



#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

