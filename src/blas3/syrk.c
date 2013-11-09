
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <pthread.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_update_sym) && defined(__rank_diag)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_colwise_inner_no_scale) && defined(__kernel_colwise_inner_scale_c)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

void
__rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
            DTYPE alpha, DTYPE beta,  int flags,  int P, int nC, cache_t *cache)
{
  register int i, incA, trans;
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {B->md, B->step};

  incA = flags & ARMAS_TRANSA ? A->step : 1;
  trans = flags & ARMAS_TRANSA ? ARMAS_TRANSA : ARMAS_TRANSB;

  if (flags & ARMAS_UPPER) {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      __vscale(C->md, C->step, beta, nC-i);
      // update one row of C  (nC-i columns, 1 row)
      __kernel_colwise_inner_no_scale(C, &A0, &B0, alpha, trans,
                                      P, nC-i, 1, cache); //KB, NB, MB, Acpy, Bcpy);
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incA; 
    }
  } else {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      __vscale(C->md, C->step, beta, i+1);
      // update one row of C  (nC-i columns, 1 row)
      __kernel_colwise_inner_no_scale(C, &A0, &B0, alpha, trans,
                                      P, i+1, 1, cache); //KB, NB, MB, Acpy, Bcpy);
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
static
void __rank_blk(mdata_t *C, const mdata_t *A,
                DTYPE alpha, DTYPE beta,
                int flags,  int P, int S, int E, int KB, int NB, int MB)
{
  register int i, j, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-S <= 0 || P <= 0)
    return;

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  if (flags & ARMAS_TRANSA) {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);

      // 1. update on diagonal
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &cache);

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

      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSA,
                                     P, 0, nC, 0, nI, &cache); 
    }
  } else {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &cache);

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

      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSB,
                                     P, 0, nC, 0, nI, &cache);
    }
  }
}



static
int __rank_threaded(int blk, int nblk,
                    __armas_dense_t *C, const __armas_dense_t *A, 
                    DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int K = flags & ARMAS_TRANSA ? A->rows : A->cols;
  __rank_blk((mdata_t *)C, (const mdata_t *)A, alpha, beta, flags, K, 0, C->rows,
             conf->kb, conf->nb, conf->mb);
  return 0;
}

// C := alpha*A*A.T + beta*C
// C := alpha*A.T*A + beta*C

int __armas_update_sym(__armas_dense_t *C,  const __armas_dense_t *A, 
                       DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, mb, ok, n;
  mdata_t *_C;
  const mdata_t *_A;

  if (C->rows == 0 || C->cols == 0)
    return -1;

  conf->error = 0;

  if (flags & ARMAS_TRANS)
    flags |= ARMAS_TRANSA;

  switch (flags & ARMAS_TRANSA) {
  case ARMAS_TRANSA:
    ok = C->rows == A->cols && C->rows == C->cols;
    break;
  default:
    ok = C->rows == A->rows && C->rows == C->cols;
    break;
  }
  if (!ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;

  nproc = armas_use_nproc(__armas_size(C), conf);

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // if only one thread, just do it
  if (nproc == 1) {
    __rank_blk(_C, _A, alpha, beta, flags, K, 0, C->rows,
               conf->kb, conf->nb, conf->mb);
    return 0;
  }
  return __rank_threaded(0, nproc, C, A, alpha, beta, flags, conf);
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

