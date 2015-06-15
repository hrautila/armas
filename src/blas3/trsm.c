
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
// this file provides following type independet functions
#if defined(__armas_solve_trm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__solve_right_unb) && defined(__solve_left_unb) && \
    defined(__solve_recursive) && defined(__solve_blocked)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "scheduler.h"

#if EXT_PRECISION && defined(__solve_ext)
extern int 
__solve_ext(mdata_t *B, const mdata_t *A, DTYPE alpha,
            int flags, int N, int S, int E, int KB, int NB, int MB, int optflags);

#endif

// ------------------------------------------------------------------------------

static
void *__compute_block(void *arg) {
  kernel_param_t *kp = (kernel_param_t *)arg;

#if defined(__solve_ext)
    // if extended precision enabled and requested
  IF_EXPR2(kp->optflags&ARMAS_OEXTPREC, arg,
           __solve_ext(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, kp->optflags));
#endif

  switch (kp->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
  case ARMAS_SNAIVE:
    if (kp->flags & ARMAS_RIGHT) {
      __solve_right_unb(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    } else {
      __solve_left_unb(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    }
    break;

  case ARMAS_RECURSIVE:
    __solve_recursive(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                      kp->KB, kp->NB, kp->MB);
    break;

  default:
    __solve_blocked(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                    kp->KB, kp->NB, kp->MB); 
    break;
  }
  return arg;
}

// ------------------------------------------------------------------------------
// recursive threading 

static
int __solve_trm_threaded(int blknum, int nblk, 
                         __armas_dense_t *B, const __armas_dense_t *A, 
                         DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ir, ie, err;
  mdata_t *_B;
  const mdata_t *_A;
  pthread_t th;
  kernel_param_t kp;

  _B = (mdata_t*)B;
  _A = (const mdata_t *)A;

  if (flags & ARMAS_RIGHT) {
    ir = __block_index4(blknum, nblk, B->rows);
    ie = __block_index4(blknum+1, nblk, B->rows);
  } else {
    ir = __block_index4(blknum, nblk, B->cols);
    ie = __block_index4(blknum+1, nblk, B->cols);
  }

  if (blknum == nblk-1) {

#if defined(__solve_ext)
    // if extended precision enabled and requested
    IF_EXPR(conf->optflags&ARMAS_OEXTPREC,
            __solve_ext(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, conf->optflags));
#endif

    // normal precision here
    switch (conf->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
    case ARMAS_SNAIVE:
      if (flags & ARMAS_RIGHT) {
        __solve_right_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      } else {
        __solve_left_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      }
      break;

    case ARMAS_RECURSIVE:
      __solve_recursive(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb);
      break;

    default:
      __solve_blocked(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb);
      break;
    }
    return 0;
  }

  // C = nil, beta = alpha, L, R = 0
  __kernel_params(&kp, _B, _A, 0, alpha, alpha, flags, A->cols, ir, 0, 0, ie,
                  conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  err = pthread_create(&th, NULL, __compute_block, &kp);
  if (err) {
    conf->error = -err;
    return -1;
  }
  // recursively invoke next block
  err = __solve_trm_threaded(blknum+1, nblk, B, A, alpha, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  return err;
}



// ------------------------------------------------------------------------------
// blocked scheduling to worker threads.

static
int __solve_trm_schedule(int nblk, 
                         __armas_dense_t *B, const __armas_dense_t *A, 
                         DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ir, ie, err, nT, k, j;
  mdata_t *_B;
  const mdata_t *_A;
  blas_task_t *tasks;
  armas_counter_t ready;

  _B = (mdata_t*)B;
  _A = (const mdata_t *)A;

  nT = nblk;
  tasks = (blas_task_t *)calloc(nT, sizeof(blas_task_t));
  if (! tasks) {
    conf->error = ARMAS_EMEMORY;
    return -1;
  }
  armas_counter_init(&ready, nT);
  k = 0; 

  for (j = 0; j < nblk; j++) {
    if (flags & ARMAS_RIGHT) {
      ir = __block_index4(j, nblk, B->rows);
      ie = __block_index4(j+1, nblk, B->rows);
    } else {
      ir = __block_index4(j, nblk, B->cols);
      ie = __block_index4(j+1, nblk, B->cols);
    }

    // C = B, not used: beta = alpha, L, R = 0
    __kernel_params(&tasks[k].kp, _B, _A, 0, alpha, alpha, flags, A->cols, ir, 0, 0, ie,
                  conf->kb, conf->nb, conf->mb, conf->optflags);
    // init task
    armas_task_init(&tasks[k].t, k, __compute_block, &tasks[k].kp, &ready);
    // schedule
    armas_schedule(&tasks[k].t);
    k++;
  }

  // wait for tasks to finish
  armas_counter_wait(&ready);
  // 1. check that task worker count is zero on all tasks
  int refcnt = 0;
  for (j = 0; j < nT; j++) {
    refcnt += tasks[j].t.wcnt;
  }
  assert(refcnt == 0);
  // release task memory
  free(tasks);
}


/**
 * @brief Triangular solve with multiple right hand sides
 *
 * If flag bit LEFT is set then computes 
 * > B = alpha*A.-1*B\n
 * > B = alpha*A.-T*B  if TRANSA or TRANS
 *
 * If flag bit RIGHT is set then computes
 * > B = alpha*B*A.-1\n
 * > B = alpha*B*A.-T  if TRANSA or TRANS
 *
 * The matrix A is upper (lower) triangular matrix if ARMAS_UPPER (ARMAS_LOWER) is
 * set. If matrix A is upper (lower) then the strictly lower (upper) part is not
 * referenced. Flag bit UNIT indicates that matrix A is unit diagonal and the diagonal
 * elements are not accessed.
 *
 * @param[in,out] B  Result matrix
 * @param[in]   A Triangular operand matrix
 * @param[in]   alpha scalar multiplier
 * @param[in]   flags option bits
 * @param[in,out] conf environment configuration
 *
 * @retval 0 Succeeded
 * @retval -1 Failed, conf->error set to error code.
 *
 * @ingroup blas3
 */
int __armas_solve_trm(__armas_dense_t *B, const __armas_dense_t *A, 
                      DTYPE alpha, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, ok, opts;
  mdata_t *_B;
  const mdata_t *_A;

  if (__armas_size(B) == 0 || __armas_size(A) == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();
  
  // check consistency
  switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
  case ARMAS_RIGHT:
    ok = B->cols == A->rows && A->cols == A->rows;
    break;
  case ARMAS_LEFT:
  default:
    ok = A->cols == A->rows && A->cols == B->rows;
    break;
  }
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  _B = (mdata_t*)B;
  _A = (const mdata_t *)A;

  // tiled scheduling not supported
  opts = conf->optflags;
  if (opts & ARMAS_BLAS_TILED) {
    opts |= ARMAS_BLAS_BLOCKED;
    opts ^= ARMAS_BLAS_TILED;
  }
  nproc = armas_nblocks(__armas_size(B), conf->wb, conf->maxproc, opts);
  // nproc = armas_use_nproc(__armas_size(B), conf);
  //printf("__armas_solve_trm2: nproc=%d, opts=x%x\n", (int)nproc, conf->optflags);

  if (nproc == 1) {
    ie = flags & ARMAS_RIGHT ? B->rows : B->cols;

#if defined(__solve_ext)
    // if extended precision enabled and requested
    IF_EXPR2(conf->optflags&ARMAS_OEXTPREC, 0, 
             __solve_ext(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb, conf->optflags));
#endif

    // normal precision here
    switch (conf->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
    case ARMAS_SNAIVE:
      //printf("__armas_solve_trm2 naive ...\n");
      if (flags & ARMAS_RIGHT) {
        __solve_right_unb(_B, _A, alpha, flags, A->cols, 0, ie);
      } else {
        __solve_left_unb(_B, _A, alpha, flags, A->cols, 0, ie);
      }
      return 0;
    case ARMAS_RECURSIVE:
      //printf("__armas_solve_trm2 recursive ...\n");
      __solve_recursive(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb);
      return 0;
    default:
      __solve_blocked(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb);
      return 0;
    }
  }

  if (conf->optflags & (ARMAS_BLAS_BLOCKED|ARMAS_BLAS_TILED)) {
    return __solve_trm_schedule(nproc, B, A, alpha, flags, conf);
  }
  return __solve_trm_threaded(0, nproc, B, A, alpha, flags, conf);
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
