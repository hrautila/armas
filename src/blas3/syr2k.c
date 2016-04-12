
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank-2k update

//! \cond
#include <stdio.h>
#include <pthread.h>
#include <string.h>
//! \endcond
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

//! \cond
#include "internal.h"
#include "matrix.h"
#include "scheduler.h"
//! \endcond

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
                 int P, int S, int E,  int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  register int i, nI, nC;
  mdata_t Cd, Ad, Bd;
  cache_t mcache;

  if (E-S <= 0 || P <= 0) {
    return;
  }

  armas_cache_setup2(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  if (flags & ARMAS_TRANSA) {
    //   C = alpha*A.T*B + alpha*B.T*A + beta*C 
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &mcache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, 0, i);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, B, 0, S);
        nC = i;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Bd, B, 0, i+nI);
        nC = E - i - nI;
      }
      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, ARMAS_TRANSA,
                                     P, 0, nC, 0, nI, &mcache); 

      // 2nd part
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Bd, &Ad, alpha, 1.0, flags, P, nI, &mcache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Bd, B, 0, i);
      if (flags & ARMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Ad, A, 0, S);
        nC = i;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Ad, A, 0, i+nI);
        nC = E - i - nI;
      }
      __kernel_colwise_inner_scale_c(&Cd, &Bd, &Ad, alpha, 1.0, ARMAS_TRANSA,
                                     P, 0, nC, 0, nI, &mcache);

    }
  } else {
    //   C = alpha*A*B.T + alpha*B*A.T + beta*C  
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __subblock(&Bd, B, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &mcache); 

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
                                     P, 0, nC, 0, nI, &mcache);

      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, B, i, 0);
      __subblock(&Bd, A, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, 1.0, flags, P, nI, &mcache);

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
                                     P, 0, nC, 0, nI, &mcache);

    }
  }
}

#if defined(ENABLE_THREADS)

extern
void __update_trm_blocked(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          DTYPE alpha, DTYPE beta, int flags, int P, int S, int L, 
                          int R, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);

static
void *__compute_recursive(void *arg)
{
  kernel_param_t *kp = ((block_args_t *)arg)->kp;
  armas_cbuf_t *cbuf  = ((block_args_t *)arg)->cbuf;

  int flags = kp->flags & ARMAS_TRANS ? ARMAS_TRANSA : ARMAS_TRANSB;
  flags |= kp->flags & ARMAS_UPPER ? ARMAS_UPPER : ARMAS_LOWER;
  
  // C = beta*C + alpha*A*B.T or C = beta*C + alpha*A.T*B
  __update_trm_blocked(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, flags, 
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
  // C = C + alpha*B*A.T or C = C + alpha*B.T*A
  __update_trm_blocked(&kp->C, &kp->B, &kp->A, kp->alpha, __ONE, flags, 
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
  return (void *)0;
}

static 
int __rank2_recursive(int blk, int nblk, __armas_dense_t *C,
                      const __armas_dense_t *A, const __armas_dense_t *B,
                      DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int rs, re, cs, ce, err;
  mdata_t *_C, C0, A0, B0;
  const mdata_t *_A, *_B; 
  pthread_t th;
  kernel_param_t kp;
  armas_cbuf_t cbuf;
  block_args_t args = (block_args_t){&kp, &cbuf};

  if (flags & ARMAS_UPPER) {
    rs = __block_index4(blk, nblk, C->rows);
    re = __block_index4(blk+1, nblk, C->rows);
    cs = rs;
    ce = C->cols;
  } else {
    cs = __block_index4(blk, nblk, C->cols);
    ce = __block_index4(blk+1, nblk, C->cols);
    rs = cs;
    re = C->rows;
  }

  _C = (mdata_t *)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  int K = flags & ARMAS_TRANS ? A->rows : A->cols;

  // shift the start point to top-left corner [rs,cs] of this block. Block size
  //   UPPER:  [rows/nblk-rs, cols-ce] 
  //   LOWER:  [cols/nblk-cs, rows-rs] 
  __subblock(&C0, _C, rs, cs);
  __subblock(&A0, _A, (flags & ARMAS_TRANS ? 0 : rs), (flags & ARMAS_TRANS ? rs : 0));
  __subblock(&B0, _B, (flags & ARMAS_TRANS ? 0 : rs), (flags & ARMAS_TRANS ? rs : 0));

  if (blk == nblk-1) {
    armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
    
    __rank2_blk(&C0, &A0, &B0, alpha, beta, flags, K, 0, re-rs,
                conf->kb, conf->nb, conf->mb, &cbuf);

    armas_cbuf_release(&cbuf);
    return 0;
  }

  __kernel_params(&kp, &C0, &A0, &B0, alpha, beta, flags, K, 0, ce-cs,
                  0, re-rs, conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
  err = pthread_create(&th, NULL, __compute_recursive, &args);
  if (err) {
    conf->error = -err;
    armas_cbuf_release(&cbuf);
    return -1;
  }
  // recursively invoke next block
  err = __rank2_recursive(blk+1, nblk, C, A, B, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  armas_cbuf_release(&cbuf);
  return 0;

}

static
void *__compute_block2(void *arg, armas_cbuf_t *cbuf)
{
  kernel_param_t *kp = (kernel_param_t *)arg;

  int flags = kp->flags & ARMAS_TRANS ? ARMAS_TRANSA : ARMAS_TRANSB;
  flags |= kp->flags & ARMAS_UPPER ? ARMAS_UPPER : ARMAS_LOWER;
  
  if (kp->optflags & ARMAS_OBLAS_BLOCKED) {
    // C = beta*C + alpha*A*B.T || C = beta*C + alpha*A.T*B
    __update_trm_blocked(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, flags, 
                         kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
    // C = C + alpha*B*A.T || C = C + alpha*B.T*A
    __update_trm_blocked(&kp->C, &kp->B, &kp->A, kp->alpha, __ONE, flags, 
                         kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
  } else {
    // tiled
    //printf("..tiled: R,S=%d,%d, E,L=%d,%d\n", kp->R, kp->S,  kp->E, kp->L);
    if (kp->S == kp->R && kp->L == kp->E) {
      // diagonal block; 
      __update_trm_blocked(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, flags, 
                           kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);

      __update_trm_blocked(&kp->C, &kp->B, &kp->A, kp->alpha, __ONE, flags, 
                           kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
    } else {
      // square block;
      __kernel_inner(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, flags,
                     kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);

      __kernel_inner(&kp->C, &kp->B, &kp->A, kp->alpha, __ONE, flags,
                     kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);

    }
  }
  return (void *)0;
}

static 
int __rank2_schedule(int nblk, __armas_dense_t *C,
                     const __armas_dense_t *A, const __armas_dense_t *B,
                     DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int rN, cN, i, j, iR, iE, jS, jL, k, nT, K;
  blas_task_t *tasks;
  mdata_t *_C;
  const mdata_t *_A, *_B;
  armas_counter_t ready;

  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  K = flags & ARMAS_TRANS ? A->rows : A->cols;

  // number of tasks
  nT = nblk; rN = cN = 0;
  if (conf->optflags & ARMAS_OBLAS_TILED) {
    nT = blocking(C->rows, C->cols, conf->wb, &rN, &cN);
    // substract the off-diagonal tile count
    nT -= ((rN - 1)*cN)/2;
  }

  tasks = (blas_task_t *)calloc(nT, sizeof(blas_task_t));
  if (! tasks) {
    conf->error = ARMAS_EMEMORY;
    return -1;
  }
  armas_counter_init(&ready, nT);
  k = 0; 

  if (conf->optflags & ARMAS_OBLAS_BLOCKED) {
    // compute in nblk blocks (row stripes for upper/column stripes for lower)
    iR = 0; iE = C->rows;
    jS = 0; jL = C->cols;
    for (j = 0; j < nblk; j++) {
      if (flags & ARMAS_UPPER) {
        iR = __block_index4(j,   nblk, C->rows);
        iE = __block_index4(j+1, nblk, C->rows);
        jS = iR;
      } else {
        jS = __block_index4(j,   nblk, C->cols);
        jL = __block_index4(j+1, nblk, C->cols);
        iR = jS;
      }
      __kernel_params(&tasks[k].kp, _C, _A, _B, alpha, beta, flags, K, jS, jL, iR, iE,
                      conf->kb, conf->nb, conf->mb, conf->optflags);
      // init task and schedule
      armas_task2_init(&tasks[k].t, k, __compute_block2, &tasks[k].kp, &ready);
      armas_schedule(&tasks[k].t);
      k++;
    }
  } else {
    // compute in tiles of wb x wb; C is square matrix --> cN == rN
    for (j = 0; j < cN; j++) {
      jS = block_index(j,   cN, conf->wb, C->cols);
      jL = block_index(j+1, cN, conf->wb, C->cols);
      for (i = j; i < rN; i++) {
        iR = block_index(i,   rN, conf->wb, C->rows);
        iE = block_index(i+1, rN, conf->wb, C->rows);
        if (flags & ARMAS_UPPER) {
          __kernel_params(&tasks[k].kp, _C, _A, _B, alpha, beta, flags,
                          K, iR, iE, jS, jL, conf->kb, conf->nb, conf->mb, conf->optflags);
        } else {
          __kernel_params(&tasks[k].kp, _C, _A, _B, alpha, beta, flags,
                          K, jS, jL, iR, iE, conf->kb, conf->nb, conf->mb, conf->optflags);
        }

        // init task and schedule
        armas_task2_init(&tasks[k].t, k, __compute_block2, &tasks[k].kp, &ready);
        armas_schedule(&tasks[k].t);
        k++;
      }
    }
  }
  assert(k == nT);
  // wait for tasks to finish
  armas_counter_wait(&ready);
  // 1. check that task worker count is zero on all tasks
  int refcnt = 0;
  for (i = 0; i < nT; i++) {
    refcnt += tasks[i].t.wcnt;
  }
  assert(refcnt == 0);
  // release task memory
  free(tasks);
  return 0;
}
#endif // ENABLE_THREADS

/**
 * @brief Symmetric matrix rank-2k update
 *
 * Computes
 * - \f$ C = beta \times C + alpha \times A B^T + alpha \times B A^T \f$
 * - \f$ C = beta \times C + alpha \times A^T B + alpha \times B^T A \f$ if *ARMAS_TRANSA* set
 *
 * Matrix C has elements stored in the  upper (lower) triangular part
 * if flag bit *ARMAS_UPPER* (*ARMAS_LOWER*) is set.
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
 * @retval 0  Operation succeeded
 * @retval <0 Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int __armas_update2_sym(__armas_dense_t *C,
                        const __armas_dense_t *A, const __armas_dense_t *B,
                        DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  if (__armas_size(C) == 0 || __armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();

  int ok = A->rows == B->rows && A->cols == B->cols && C->rows == C->cols;
  switch (flags & ARMAS_TRANS) {
  case ARMAS_TRANS:
    ok = ok && C->rows == A->cols;
    break;
  default:
    ok = ok && C->rows == A->rows;
    break;
  }
  if (!ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  
#if defined(ENABLE_THREADS)
  long nproc = armas_use_nproc(__armas_size(C), conf);
  if (conf->optflags & (ARMAS_OBLAS_BLOCKED|ARMAS_OBLAS_TILED)) {
    return __rank2_schedule(nproc, C, A, B, alpha, beta, flags, conf);
  }
  return __rank2_recursive(0, nproc, C, A, B, alpha, beta, flags, conf);
#else
  mdata_t *_C = (mdata_t*)C;
  const mdata_t *_A = (const mdata_t *)A;
  const mdata_t *_B = (const mdata_t *)B;
  armas_cbuf_t *cbuf = armas_cbuf_get(conf);

  int K = flags & ARMAS_TRANS ? A->rows : A->cols;

  // if only one thread, just do it
  __rank2_blk(_C, _A, _B, alpha, beta, flags, K, 0, C->rows,
              conf->kb, conf->nb, conf->mb, cbuf);
  return 0;
#endif
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

