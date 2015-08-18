
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_update2_sym) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_colwise_inner_scale_c) && defined(__rank_diag)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"


// SYR2K;
//   C = alpha*A*B.T + alpha*B*A.T + beta*C  
//   C = alpha*A.T*B + alpha*B.T*A + beta*C  if flags & ARMAS_TRANS
/*
 * Symmetric rank2 update
 *
 * upper
 *   C00 C01 C02    C00  C01 C02     A0               B0           
 *    0  C11 C12 =   0   C11 C12  +  A1 * B0 B1 B2 +  B1 * A0 A1 A2
 *    0   0  C22     0   0   C22     A2               B2           
 *
 * lower:
 *   C00  0   0    C00   0   0      A0               B0           
 *   C10 C11  0 =  C10  C11  0   +  A1 * B0 B1 B2 +  B1 * A0 A1 A2
 *   C20 C21 C22   C20  C21 C22     A2               B2           
 *
 */
void __rank2_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                 DTYPE alpha, DTYPE beta, int flags,
                 int P, int S, int E,  int KB, int NB, int MB)
{
  register int i, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-S <= 0 || P <= 0)
    return;

  if (NB > E-S) {
    NB = E-S;
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

  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  // setup cache area
  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB, (mdata_t *)0, (mdata_t *)0};

  if (flags & ARMAS_TRANSA) {
    //   C = alpha*A.T*B + alpha*B.T*A + beta*C 
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, 0, i);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, B, 0, S);
        nC = E - i - nI;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Bd, B, 0, i+nI);
        nC = i;
      }
      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSA,
                                     P, 0, nC, 0, nI, &cache); 

      // 2nd part
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Bd, &Ad, alpha, 1.0, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Bd, B, 0, i);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Ad, A, 0, S);
        nC = E - i - nI;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Ad, A, 0, i+nI);
        nC = i;
      }
      __kernel_colwise_inner_scale_c(&Cd, &Bd, &Ad, alpha, 1.0, ARMAS_TRANSA,
                                     P, 0, nC, 0, nI, &cache);

    }
  } else {
    //   C = alpha*A*B.T + alpha*B*A.T + beta*C  
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __subblock(&Bd, B, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &cache); 

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, i, 0);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, B, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, B, i+nI, 0);
        nC = E - i - nI;
      }

      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSB,
                                     P, 0, nC, 0, nI, &cache);

      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, B, i, 0);
      __subblock(&Bd, A, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, 1.0, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, B, i, 0);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, A, i+nI, 0);
        nC = E - i - nI;
      }

      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, 1.0, ARMAS_TRANSB,
                                     P, 0, nC, 0, nI, &cache);

    }
  }
}


static
int __rank2_threaded(int blk, int nblk, __armas_dense_t *C,
                     const __armas_dense_t *A, const __armas_dense_t *B,
                     DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  __rank2_blk((mdata_t *)C, (const mdata_t *)A, (const mdata_t *)B, alpha, beta,
              flags, K, 0, C->rows, conf->kb, conf->nb, conf->mb);
  return 0;
}



/**
 * @brief Symmetric matrix rank-2k update
 *
 * Computes
 * > C = beta*C + alpha*A*B.T + alpha*B*A.T\n
 * > C = beta*C + alpha*A.T*B + alpha*B.T*A   if TRANSA
 *
 * Matrix C has elements stored in the  upper (lower) triangular part
 * if flag bit ARMAS_UPPER (ARMAS_LOWER) is set.
 * If matrix is upper (lower) then the strictly lower (upper) part is not referenced.
 *
 * @param[in,out] C result matrix
 * @param[in] A first operand matrix
 * @param[in] B second operand matrix
 * @param[in] alpha scalar constant
 * @param[in] beta scalar constant
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval 0 Operation succeeded
 * @retval -1 Failed, conf->error set to actual error code.
 *
 * @ingroup blas3
 */
int __armas_update2_sym(__armas_dense_t *C,
                        const __armas_dense_t *A, const __armas_dense_t *B,
                        DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  long nproc;
  int K;
  mdata_t *_C;
  const mdata_t *_A, *_B;

  if (__armas_size(C) == 0 || __armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();

  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  nproc = armas_use_nproc(__armas_size(C), conf);

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // if only one thread, just do it
  if (nproc == 1) {
    __rank2_blk(_C, _A, _B, alpha, beta, flags, K, 0, C->rows,
                conf->kb, conf->nb, conf->mb);
    return 0;
  }

  return __rank2_threaded(0, nproc, C, A, B, alpha, beta, flags, conf);
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

