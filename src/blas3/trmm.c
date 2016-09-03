
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Triangular matrix multiplication

//! \cond
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mult_trm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__trmm_unb) && defined(__trmm_recursive) && defined(__trmm_blk)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
//#include "mvec_nosimd.h"
#include "scheduler.h"

#if EXT_PRECISION && defined(__trmm_ext_blk)
extern void __trmm_ext_blk(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                           int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);
#define HAVE_EXT_PRECISION 1
#else
#define HAVE_EXT_PRECISION 0
#endif

#include "cond.h"
//! \endcond

// Functions here implement various versions of TRMM operation.
#if defined(ENABLE_THREADS)

// ------------------------------------------------------------------------------

static
void *__compute_recursive(void *arg) 
{
  kernel_param_t *kp = ((block_args_t *)arg)->kp;
  armas_cbuf_t *cbuf  = ((block_args_t *)arg)->cbuf;

  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
      __trmm_ext_blk(&kp->C, &kp->A, kp->alpha, kp->flags, 
                     kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
      return (void *)0;
  }

  switch (kp->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
  case ARMAS_ONAIVE:
    __trmm_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    break;
  case ARMAS_ORECURSIVE:
    __trmm_recursive(&kp->C, &kp->A, kp->alpha, kp->flags,
                     kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
    break;
  default:
    __trmm_blk(&kp->C, &kp->A, kp->alpha, kp->flags,
               kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
  }
  return (void *)0;
}

static
void *__compute_block2(void *arg, armas_cbuf_t *cbuf) 
{
  kernel_param_t *kp = (kernel_param_t *)arg;

  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
      __trmm_ext_blk(&kp->C, &kp->A, kp->alpha, kp->flags, 
                     kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
      return (void *)0;
  }

  switch (kp->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
  case ARMAS_ONAIVE:
    __trmm_unb(&kp->C, &kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
    break;
  case ARMAS_ORECURSIVE:
    __trmm_recursive(&kp->C, &kp->A, kp->alpha, kp->flags,
                      kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
    break;
  default:
    __trmm_blk(&kp->C, &kp->A, kp->alpha, kp->flags,
                kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
  }
  return arg;
}


// ------------------------------------------------------------------------------
// recursive thread scheduling

static
int __mult_trm_threaded(int blknum, int nblk, 
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

    // if extended precision enabled; 
    if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
      // compiler dead code pruning removes following if HAVE_EXT_PRECISION == 0
      __trmm_ext_blk(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, &cbuf);
      armas_cbuf_release(&cbuf);
      return 0;
    }

    // normal precision
    switch (conf->optflags & (ARMAS_ONAIVE|ARMAS_ORECURSIVE)) {
    case ARMAS_ONAIVE:
      __trmm_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      break;
    case ARMAS_ORECURSIVE:
      __trmm_recursive(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, &cbuf);
      break;
    default:
      __trmm_blk(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb, &cbuf);
      break;
    }
    armas_cbuf_release(&cbuf);

    return 0;
  }

  // C = B, not used: beta = alpha, L, R = 0
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
  err = __mult_trm_threaded(blknum+1, nblk, B, A, alpha, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  armas_cbuf_release(&cbuf);
  return err;
}


// ------------------------------------------------------------------------------
// blocked scheduling to worker threads.

static
int __mult_trm_schedule(int nblk, 
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

#endif // ENABLE_THREADS

/**
 * @brief Triangular matrix-matrix multiply
 *
 * If flag bit *ARMAS_LEFT* is set then computes 
 *    - \f$ B = alpha \times A B \f$
 *    - \f$ B = alpha \times A^T B  \f$ if *ARMAS_TRANS* set
 *
 * If flag bit *ARMAS_RIGHT* is set then computes
 *    - \f$ B = alpha \times B A \f$
 *    - \f$ B = alpha \times B A^T \f$ if *ARMAS_TRANS*  set
 *
 * The matrix A is upper (lower) triangular matrix if *ARMAS_UPPER* (*ARMAS_LOWER*) is
 * set. If matrix A is upper (lowert) then the strictly lower (upper) part is not
 * referenced. Flag bit *ARMAS_UNIT* indicates that matrix A is unit diagonal and the diagonal
 * entries are not accessed.
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in,out] B  Result matrix
 * @param[in]   A Triangular operand matrix
 * @param[in]   alpha scalar multiplier
 * @param[in]   flags option bits
 * @param[in,out] conf environment configuration
 *
 * @retval 0  Succeeded
 * @retval <0 Failed, conf.error set to error code.
 *
 * @ingroup blas3
 */
int armas_x_mult_trm(armas_x_dense_t *B, const armas_x_dense_t *A, 
                      DTYPE alpha, int flags, armas_conf_t *conf)
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
    ok = B->rows == A->cols && A->cols == A->rows;
    break;
  }
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

#if defined(ENABLE_THREADS)
  int opts = conf->optflags;
  if (opts & ARMAS_OBLAS_TILED) {
    opts ^= ~ARMAS_OBLAS_TILED;
    opts |= ARMAS_OBLAS_BLOCKED;
  }
  long nproc = armas_nblocks(armas_x_size(B), conf->wb, conf->maxproc, opts);

  if (conf->optflags & (ARMAS_OBLAS_BLOCKED|ARMAS_OBLAS_TILED) && nproc > 1) {
    return __mult_trm_schedule(nproc, B, A, alpha, flags, conf);
  }
  return __mult_trm_threaded(0, nproc, B, A, alpha, flags, conf);

#else
  int ie = flags & ARMAS_RIGHT ? B->rows : B->cols;
  mdata_t *_B = (mdata_t*)B;
  const mdata_t *_A = (const mdata_t *)A;
  armas_cbuf_t *cbuf = conf->cbuf ? conf->cbuf : armas_cbuf_default();

  
  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    __trmm_ext_blk(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb, cbuf);
    return 0;
  }

  // normal precision here; extended precision not enabled neither requested
  switch (conf->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
  case ARMAS_SNAIVE:
    __trmm_unb(_B, _A, alpha, flags, A->cols, 0, ie);
    break;
  case ARMAS_RECURSIVE:
    __trmm_recursive(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb);
    break;
  default:
    __trmm_blk(_B, _A, alpha, flags, A->cols, 0, ie, conf->kb, conf->nb, conf->mb);
    break;
  }
  return 0;
#endif
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
