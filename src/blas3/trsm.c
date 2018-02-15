
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Triangular matrix solve

//! \cond
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_solve_trm) 
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

//! \cond
#include "internal.h"
#include "matrix.h"
#include "scheduler.h"

#if EXT_PRECISION && defined(__solve_ext)
extern int 
__solve_ext(mdata_t *B, const mdata_t *A, DTYPE alpha,
            int flags, int N, int S, int E, int KB, int NB, int MB, int optflags, armas_cbuf_t *cbuf);

#define HAVE_EXT_PRECISION 1
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"
//! \endcond

// ------------------------------------------------------------------------------

#if defined(ENABLE_THREADS)

static
void *__compute_recursive(void *arg) 
{
  kernel_param_t *kp = ((block_args_t *)arg)->kp;
  armas_cbuf_t *cbuf  = ((block_args_t *)arg)->cbuf;

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
    __solve_ext(&kp->C, &kp->A, kp->alpha, kp->flags,
                kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, kp->optflags, cbuf);
    return (void *)0;
  }

  switch (kp->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
  case ARMAS_ONAIVE:
    if (kp->flags & ARMAS_RIGHT) {
      __solve_right_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    } else {
      __solve_left_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    }
    break;

  case ARMAS_ORECURSIVE:
    __solve_recursive(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                       kp->KB, kp->NB, kp->MB, cbuf);
    break;

  default:
    __solve_blocked(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                     kp->KB, kp->NB, kp->MB, cbuf); 
    break;
  }
  return arg;
}

void *__compute_block2(void *arg, armas_cbuf_t *cbuf) 
{
  kernel_param_t *kp = (kernel_param_t *)arg;

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
    __solve_ext(&kp->C, &kp->A, kp->alpha, kp->flags,
                kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, kp->optflags, cbuf);
    return (void *)0;
  }

  switch (kp->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
  case ARMAS_ONAIVE:
    if (kp->flags & ARMAS_RIGHT) {
      __solve_right_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    } else {
      __solve_left_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    }
    break;

  case ARMAS_ORECURSIVE:
    __solve_recursive(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                       kp->KB, kp->NB, kp->MB, cbuf);
    break;

  default:
    __solve_blocked(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E,
                     kp->KB, kp->NB, kp->MB, cbuf); 
    break;
  }
  return arg;
}

// ------------------------------------------------------------------------------
// recursive threading 

static
int __solve_trm_threaded(int blknum, int nblk, 
                         armas_x_dense_t *B, const armas_x_dense_t *A, 
                         DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ir, ie, err;
  mdata_t *_B;
  const mdata_t *_A;
  pthread_t th;
  kernel_param_t kp;
  armas_cbuf_t cbuf;
  block_args_t args = (block_args_t){&kp, &cbuf};

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
    armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
    // if extended precision enabled and requested
    if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
      __solve_ext(_B, _A, alpha, flags, A->cols,
                  ir, ie, conf->kb, conf->nb, conf->mb, conf->optflags, &cbuf);
      armas_cbuf_release(&cbuf);
      return 0;
    }

    // otherwise; normal precision here
    switch (conf->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
    case ARMAS_ONAIVE:
      if (flags & ARMAS_RIGHT) {
        __solve_right_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      } else {
        __solve_left_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      }
      break;

    case ARMAS_ORECURSIVE:
      __solve_recursive(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, &cbuf);
      break;

    default:
      __solve_blocked(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, &cbuf);
      break;
    }
    armas_cbuf_release(&cbuf);
    return 0;
  }

  // C = nil, beta = alpha, L, R = 0
  __kernel_params(&kp, _B, _A, 0, alpha, alpha, flags, A->cols, ir, 0, 0, ie,
                  conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
  err = pthread_create(&th, NULL, __compute_recursive, &args);
  if (err) {
    conf->error = -err;
    armas_cbuf_release(&cbuf);
    return -1;
  }
  // recursively invoke next block
  err = __solve_trm_threaded(blknum+1, nblk, B, A, alpha, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  armas_cbuf_release(&cbuf);
  return err;
}



// ------------------------------------------------------------------------------
// blocked scheduling to worker threads.

static
int __solve_trm_schedule(int nblk, 
                         armas_x_dense_t *B, const armas_x_dense_t *A, 
                         DTYPE alpha, int flags, armas_conf_t *conf)
{
  int ir, ie, nT, k, j;
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
    armas_task2_init(&tasks[k].t, k, __compute_block2, &tasks[k].kp, &ready);
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
  return 0;
}

#endif  // ENABLE_THREADS

/**
 * @brief Triangular solve with multiple right hand sides
 *
 * If flag bit *ARMAS_LEFT* is set then computes 
 *    - \f$ B = alpha \times A^{-1} B \f$
 *    - \f$ B = alpha \times A^{-T} B \f$ if *ARMAS_TRANS* set
 *
 * If flag bit *ARMAS_RIGHT* is set then computes
 *    - \f$ B = alpha \times B A^{-1} \f$
 *    - \f$ B = alpha \times B A^{-T} \f$ if *ARMAS_TRANS* set 
 *
 * The matrix A is upper (lower) triangular matrix if *ARMAS_UPPER* (*ARMAS_LOWER*) is
 * set. If matrix A is upper (lower) then the strictly lower (upper) part is not
 * referenced. Flag bit *ARMAS_UNIT* indicates that matrix A is unit diagonal and the diagonal
 * elements are not accessed.
 *
 * If configuation option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out] B  Result matrix
 * @param[in]   alpha scalar multiplier
 * @param[in]   A Triangular operand matrix
 * @param[in]   flags option bits
 * @param[in,out] conf environment configuration
 *
 * @retval 0 Succeeded
 * @retval <0 Failed, *conf.error* set to error code.
 *
 * @ingroup blas3
 */
int armas_x_solve_trm(armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, 
                      int flags, armas_conf_t *conf)
{
  int ok;

  if (armas_x_size(B) == 0 || armas_x_size(A) == 0)
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

#if defined(ENABLE_THREADS)
  // tiled scheduling not supported
  int opts = conf->optflags;
  if (opts & ARMAS_OBLAS_TILED) {
    opts |= ARMAS_OBLAS_BLOCKED;
    opts ^= ARMAS_OBLAS_TILED;
  }

  long nproc = armas_nblocks(armas_x_size(B), conf->wb, conf->maxproc, opts);
  if (conf->optflags & (ARMAS_OBLAS_BLOCKED|ARMAS_OBLAS_TILED) && nproc > 1) {
    return __solve_trm_schedule(nproc, B, A, alpha, flags, conf);
  }
  return __solve_trm_threaded(0, nproc, B, A, alpha, flags, conf);

#else
  mdata_t *_B = (mdata_t*)B;
  const mdata_t *_A = (const mdata_t *)A;
  int ie = flags & ARMAS_RIGHT ? B->rows : B->cols;
  armas_cbuf_t *cbuf = conf->cbuf ? conf->cbuf : armas_cbuf_default();

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (conf->optflags&ARMAS_OEXTPREC)) {
      // compiler dead code pruning removes following if HAVE_EXT_PRECISION == 0
    __solve_ext(_B, _A, alpha, flags, A->cols, 0, ie,
                conf->kb, conf->nb, conf->mb, conf->optflags, cbuf);
    return 0;
  }

  // otherwise; normal precision here
  switch (conf->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
  case ARMAS_ONAIVE:
    if (flags & ARMAS_RIGHT) {
      __solve_right_unb(_B, _A, alpha, flags, A->cols, 0, ie);
    } else {
      __solve_left_unb(_B, _A, alpha, flags, A->cols, 0, ie);
    }
    break;
  case ARMAS_ORECURSIVE:
    __solve_recursive(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb, cbuf);
    break;
  default:
    __solve_blocked(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb, cbuf);
    break;
  }
  return 0;
#endif
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
