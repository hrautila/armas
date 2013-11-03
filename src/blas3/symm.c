
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>

#include "dtype.h"
#include "internal.h"
#include "matrix.h"
#include "matcpy.h"


// C += A*B; A is the diagonal block
static
void __mult_symm_diag(mdata_t *C, const mdata_t *A, const mdata_t *B,
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
void __kernel_symm_left(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        DTYPE alpha, DTYPE beta, int flags,
                        int P, int S, int L, int R, int E,
                        int KB, int NB, int MB)
{
  int i, j, nI, nJ, flags1, flags2;
  mdata_t A0, B0, C0, Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  Acpy  = (mdata_t){Abuf, MAX_KB};
  Bcpy  = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

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
      __blk_scale(&C0, beta, nI, nJ);

      // 1. off diagonal block in A; if UPPER then above [i,j]; if LOWER then left of [i,j]
      //    above|left diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? 0 : i), (flags&ARMAS_UPPER ? i : 0));
      __subblock(&B0, B, 0, j);

      __kernel_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags1,
                                      i, nJ, nI, &cache);

      // 2. on-diagonal block in A;
      __subblock(&A0, A, i, i);
      __subblock(&B0, B, i, j);
      __mult_symm_diag(&C0, &A0, &B0, alpha, flags, nI, nJ, nI, &cache);

      // 3. off-diagonal block in A; if UPPER then right of [i, i+nI];
      //    if LOWER then below [i+nI, i]

      // right|below of diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? i : i+nI), (flags&ARMAS_UPPER ? i+nI : i));
      __subblock(&B0, B, i+nI, j);
      __kernel_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags2,
                                      P-i-nI, nJ, nI, &cache); 
    }
  }
}


static
void __kernel_symm_right(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        DTYPE alpha, DTYPE beta, int flags,
                        int P, int S, int L, int R, int E,
                        int KB, int NB, int MB)
{
  int flags1, flags2;
  register int nR, nC, ic, ir;
  mdata_t A0, B0, C0, Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }

  Acpy  = (mdata_t){Abuf, MAX_KB};
  Bcpy  = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

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
      __blk_scale(&C0, beta, nR, nC);

      // above|left diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? 0 : ic), (flags&ARMAS_UPPER ? ic : 0));
      __subblock(&B0, B, ir, 0);
      __kernel_colwise_inner_no_scale(&C0, &B0, &A0, alpha, flags1,
                                      ic, nC, nR, &cache);
      // diagonal block
      __subblock(&A0, A, ic, ic);
      __subblock(&B0, B, ir, ic);
      __mult_symm_diag(&C0, &A0, &B0, alpha, flags, nC, nC, nR, &cache); 

      // right|below of diagonal
      __subblock(&A0, A, (flags&ARMAS_UPPER ? ic : ic+nC), (flags&ARMAS_UPPER ? ic+nC : ic));
      __subblock(&B0, B, ir, ic+nC);
      __kernel_colwise_inner_no_scale(&C0, &B0, &A0, alpha, flags2,
                                      P-ic-nC, nC, nR, &cache);
    }
  }

}

static
void *__start_thread(void *arg) {
  kernel_param_t *kp = (kernel_param_t *)arg;
  if (kp->flags & ARMAS_RIGHT) {
    __kernel_symm_right(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                        kp->K, kp->S, kp->L, kp->R, kp->E,
                        kp->KB, kp->NB, kp->MB);
  } else {
    __kernel_symm_left(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                       kp->K, kp->S, kp->L, kp->R, kp->E,
                       kp->KB, kp->NB, kp->MB);
  }
  return arg;
}



static
int __mult_sym_threaded(int blknum, int nblk, int colwise, 
                          __armas_dense_t *C,
                          const __armas_dense_t *A, const __armas_dense_t *B,
                          DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{

  mdata_t *_C;
  const mdata_t *_A, *_B;
  int ie, ir, K, err;
  pthread_t th;
  kernel_param_t kp;

  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  K = A->cols;

  if (colwise) {
      ir = __block_index4(blknum, nblk, C->cols);
      ie = __block_index4(blknum+1, nblk, C->cols);
  } else {
      ir = __block_index4(blknum, nblk, C->rows);
      ie = __block_index4(blknum+1, nblk, C->rows);
  }

  __DEBUG(printf("blk=%d nblk=%d, ir=%d, ie=%d\n", blknum, nblk, ir, ie));

  // last block immediately
  if (blknum == nblk-1) {
    if (flags & ARMAS_RIGHT) {
      if (colwise) {
        __kernel_symm_right(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                            conf->kb, conf->nb, conf->mb);
      } else {
        __kernel_symm_right(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                            conf->kb, conf->nb, conf->mb);
      }
    } else {
      if (colwise) {
        __kernel_symm_left(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                           conf->kb, conf->nb, conf->mb);
      } else {
        __kernel_symm_left(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                           conf->kb, conf->nb, conf->mb);
      }
    }
    return 0;
  }


  // set up call parameters for thread
  if (colwise) {
    __kernel_params(&kp, _C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                    conf->kb, conf->nb, conf->mb, conf->optflags);
  } else {
    __kernel_params(&kp, _C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                    conf->kb, conf->nb, conf->mb, conf->optflags);
  }

  // create new thread to compute this block
  err = pthread_create(&th, NULL, __start_thread, &kp);
  if (err) {
    conf->error = -err;
    return -1;
  }
  // recursively invoke next block
  err = __mult_sym_threaded(blknum+1, nblk, colwise, C, A, B, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  return err;
}



int __armas_mult_sym(__armas_dense_t *C, const __armas_dense_t *A, const __armas_dense_t *B,
                      DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, ok, empty;
  mdata_t *_C;
  const mdata_t *_A, *_B;

  if (C->rows == 0 || C->cols == 0)
    return 0;

  conf->error = 0;

  // check consistency
  switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
  case ARMAS_RIGHT:
    ok = C->rows == B->rows && C->cols == A->cols && B->cols == A->rows && A->rows == A->cols;
    empty = A->cols == 0 || B->rows == 0;
    break;
  case ARMAS_LEFT:
  default:
    ok = C->rows == A->rows && C->cols == B->cols && A->cols == B->rows && A->rows == A->cols;
    empty = A->rows == 0 || B->cols == 0;
    break;
  }
  if (empty)
    return 0;
  if (! ok) {
    conf->error = 1;
    return -1;
  }

  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  K = A->cols;
  nproc = armas_use_nproc(__armas_size(C), conf);
  if (nproc == 1) {
    if (flags & ARMAS_RIGHT) {
      __kernel_symm_right(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                          conf->kb, conf->nb, conf->mb);
    } else {
      __kernel_symm_left(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                         conf->kb, conf->nb, conf->mb);
    }
    return 0;
  }

  int colwise = 20*nproc < C->cols;
  return __mult_sym_threaded(0, nproc, colwise,
                             C, A, B, alpha, beta, flags, conf);
}


// Local Variables:
// indent-tabs-mode: nil
// End:
