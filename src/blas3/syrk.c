
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
#if defined(__kernel_colwise_inner_no_scale) && defined(__kernel_colwise_inner_scale_c)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

extern
void __update_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      DTYPE alpha, DTYPE beta, int flags,
                      int P, int S, int L, int R, int E, cache_t *cache);

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
                int flags,  int P, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  register int i, nI, nC;
  mdata_t Cd, Ad, Bd;
  cache_t mcache;

  if (E-S <= 0 || P <= 0) {
    return;
  }

  armas_cache_setup2(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));

  if (flags & ARMAS_TRANSA) {
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);

      // 1. update on diagonal
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &mcache);

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
                                     P, 0, nC, 0, nI, &mcache); 
    }
  } else {
    for (i = S; i < E; i += mcache.NB) {
      nI = E - i < mcache.NB ? E - i : mcache.NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &mcache);

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
void *__compute_threaded(void *arg)
{
  kernel_param_t *kp = ((block_args_t *)arg)->kp;
  armas_cbuf_t *cbuf  = ((block_args_t *)arg)->cbuf;

  int flags = kp->flags & ARMAS_TRANSA ? ARMAS_TRANSA : ARMAS_TRANSB;
  flags |= kp->flags & ARMAS_UPPER ? ARMAS_UPPER : ARMAS_LOWER;
  
  __update_trm_blocked(&kp->C, &kp->A, &kp->A, kp->alpha, kp->beta, flags, 
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->MB, kp->NB, kp->KB, cbuf);
  return (void *)0;
}

static
int __rank_threaded(int blk, int nblk,
                    __armas_dense_t *C, const __armas_dense_t *A, 
                    DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int rs, re, cs, ce, err;
  mdata_t *_C, C0, A0;
  const mdata_t *_A; 
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

  int K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // shift the start point to top-left corner [rs,cs] of this block. Block size
  //   UPPER:  [rows/nblk-rs, cols-ce] 
  //   LOWER:  [cols/nblk-cs, rows-rs] 
  __subblock(&C0, _C, rs, cs);
  __subblock(&A0, _A, (flags & ARMAS_TRANSA ? 0 : rs), (flags & ARMAS_TRANSA ? rs : 0));

  if (blk == nblk-1) {
    armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
    
  __rank_blk(&C0, &A0, alpha, beta, flags, K, 0, re-rs,
             conf->kb, conf->nb, conf->mb, &cbuf);

    armas_cbuf_release(&cbuf);
    return 0;
  }

  __kernel_params(&kp, &C0, &A0, (mdata_t *)0, alpha, beta, flags, K, 0, ce-cs,
                  0, re-rs, conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
  err = pthread_create(&th, NULL, __compute_threaded, &args);
  if (err) {
    conf->error = -err;
    armas_cbuf_release(&cbuf);
    return -1;
  }
  // recursively invoke next block
  err = __rank_threaded(blk+1, nblk, C, A, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  armas_cbuf_release(&cbuf);
  return 0;
}

#endif

// C := alpha*A*A.T + beta*C
// C := alpha*A.T*A + beta*C

/**
 * @brief Symmetric matrix rank-k update
 *
 * Computes
 * > C = beta*C + alpha*A*A.T\n
 * > C = beta*C + alpha*A.T*A   if TRANSA
 *
 * Matrix C is upper (lower) triangular if flag bit ARMAS_UPPER (ARMAS_LOWER)
 * is set. If matrix is upper (lower) then
 * the strictly lower (upper) part is not referenced.
 *
 * @param[in,out] C symmetric result matrix
 * @param[in] A first operand matrix
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
int __armas_update_sym(__armas_dense_t *C,  const __armas_dense_t *A, 
                       DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int ok;

  if (__armas_size(A) == 0 || __armas_size(C) == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();

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

#if defined(ENABLE_THREADS)
  long nproc = armas_use_nproc(__armas_size(C), conf);
  return __rank_threaded(0, nproc, C, A, alpha, beta, flags, conf);

#else
  mdata_t *_C = (mdata_t*)C;
  const mdata_t *_A = (const mdata_t *)A;
  armas_cbuf_t *cbuf = armas_cbuf_get(conf);

  int K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // if only one thread, just do it
  __rank_blk(_C, _A, alpha, beta, flags, K, 0, C->rows,
             conf->kb, conf->nb, conf->mb, cbuf);
  return 0;
#endif
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

