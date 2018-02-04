
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

//! \cond
#include <stdlib.h>
#include <stdint.h>
//#include <pthread.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mult) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_inner)
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

#if EXT_PRECISION && defined(__kernel_inner_ext)
extern int __kernel_inner_ext(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              DTYPE alpha, DTYPE beta, int flags,
                              int P, int S, int L, int R, int E, 
                              int KB, int NB, int MB, armas_cbuf_t *cbuf);
#define HAVE_EXT_PRECISION 1
#else
#define HAVE_EXT_PRECISION 0
#endif

// include conditional code macros; 
//#include "cond.h"

// code blocks for threaded execution
#if defined(ENABLE_THREADS)

static
void *__compute_recursive(void *arg) 
{
  kernel_param_t *kp = ((block_args_t *)arg)->kp;
  armas_cbuf_t *cbuf  = ((block_args_t *)arg)->cbuf;

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
    // compiler 'dead code' pruning removes this block if extended precision is not defined
    __kernel_inner_ext(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, kp->flags,
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
    return (void *)0;
  }

  // normal precision here; extended not enabled and not requested
  __kernel_inner(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, kp->flags,
                 kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
  return (void *)0;
}

static
void *__compute_block2(void *arg, armas_cbuf_t *cbuf) 
{
  kernel_param_t *kp = (kernel_param_t *)arg;

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (kp->optflags & ARMAS_OEXTPREC)) {
    // compiler 'dead code' pruning removes this block if extended precision is not defined
    __kernel_inner_ext(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, kp->flags,
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
    return (void *)0;
  }

  // normal precision here; extended not enabled and not requested
  __kernel_inner(&kp->C, &kp->A, &kp->B, kp->alpha, kp->beta, kp->flags,
                 kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB, cbuf);
  return 0;
}

// compute recursively in nblk threads
static
int __mult_threaded(int blknum, int nblk, int colwise, 
                     armas_x_dense_t *C,
                     const armas_x_dense_t *A,
                     const armas_x_dense_t *B,
                     DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  mdata_t *_C;
  const mdata_t *_A, *_B;
  int ie, ir, K, err;
  pthread_t th;
  kernel_param_t kp;
  armas_cbuf_t cbuf;
  block_args_t args = (block_args_t){&kp, &cbuf};
  
  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // last block immediately
  if (blknum == nblk-1) {
    armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
    if (colwise) {
      ir = __block_index4(blknum, nblk, C->cols);
      ie = __block_index4(blknum+1, nblk, C->cols);
      // if extended precision enabled and requested
      if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
        __kernel_inner_ext(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                           conf->kb, conf->nb, conf->mb, &cbuf);
        return 0;
      }
      // normal precision here
      __kernel_inner(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                     conf->kb, conf->nb, conf->mb, &cbuf);
    } else {
      ir = __block_index4(blknum,   nblk, C->rows);
      ie = __block_index4(blknum+1, nblk, C->rows);
      // if extended precision enabled and requested
      if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
        __kernel_inner_ext(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                           conf->kb, conf->nb, conf->mb, &cbuf);
        return 0;
      }
      // normal precision here
      __kernel_inner(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                     conf->kb, conf->nb, conf->mb, &cbuf);
    }
    armas_cbuf_release(&cbuf);
    return 0;
  }

  // initialize kernel parameters for this block
  if (colwise) {
      ir = __block_index4(blknum,   nblk, C->cols);
      ie = __block_index4(blknum+1, nblk, C->cols);
      __kernel_params(&kp, _C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                      conf->kb, conf->nb, conf->mb, conf->optflags);
  } else {
      ir = __block_index4(blknum,   nblk, C->rows);
      ie = __block_index4(blknum+1, nblk, C->rows);
      __kernel_params(&kp, _C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                      conf->kb, conf->nb, conf->mb, conf->optflags);
  }

  // create new thread to compute this block
  armas_cbuf_init(&cbuf, conf->cmem, conf->l1mem);
  err = pthread_create(&th, NULL, __compute_recursive, &args);
  if (err) {
    conf->error = -err;
    armas_cbuf_release(&cbuf);
    return -1;
  }

#if 0
  // set thread affinity, blknum = 0 ... nblk-1, first cpu reserved to main thread,
  // level k thread affinity set k+2'th cpu in set.
  cpuid = armas_nth_cpu(&armas_sched_default()->cpus, blknum+2);
  if (cpuid != -1) {
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);
    pthread_setaffinity_np(th, sizeof(cpuset), &cpuset);
  }
#endif

  // recursively invoke next block
  err = __mult_threaded(blknum+1, nblk, colwise, C, A, B, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  // release cache memory block
  armas_cbuf_release(&cbuf);
  return err;
}

// --------------------------------------------------------------------------------------
// new scheduler

static
int __mult_schedule(int nblk, int colwise, armas_x_dense_t *C,
                     const armas_x_dense_t *A,
                     const armas_x_dense_t *B,
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

  K = flags & (ARMAS_TRANSA|ARMAS_CONJA) ? A->rows : A->cols;

  if (conf->optflags & ARMAS_OBLAS_BLOCKED) {
    nT = nblk;
  } else {
    nT = blocking(C->rows, C->cols, conf->wb, &rN, &cN);
  }
  tasks = (blas_task_t *)calloc(nT, sizeof(blas_task_t));
  if (! tasks) {
    conf->error = ARMAS_EMEMORY;
    return -1;
  }
  armas_counter_init(&ready, nT);
  k = 0; 

  if (conf->optflags & ARMAS_OBLAS_BLOCKED) {
    iR = 0; iE = C->rows;
    jS = 0; jL = C->cols;
    for (j = 0; j < nblk; j++) {
      if (colwise) {
        jS = __block_index4(j,   nblk, C->cols);
        jL = __block_index4(j+1, nblk, C->cols);
      } else {
        iR = __block_index4(j,   nblk, C->rows);
        iE = __block_index4(j+1, nblk, C->rows);
      }
      // set parameters
      __kernel_params(&tasks[k].kp, _C, _A, _B, alpha, beta, flags, K, jS, jL, iR, iE,
                      conf->kb, conf->nb, conf->mb, conf->optflags);
      // init task
      armas_task2_init(&tasks[k].t, k, __compute_block2, &tasks[k].kp, &ready);
      // schedule
      armas_schedule(&tasks[k].t);
      k++;
    }
  } else {
    for (j = 0; j < cN; j++) {
      jS = block_index(j,   cN, conf->wb, C->cols);
      jL = block_index(j+1, cN, conf->wb, C->cols);
      for (i = 0; i < rN; i++) {
        iR = block_index(i,   rN, conf->wb, C->rows);
        iE = block_index(i+1, rN, conf->wb, C->rows);
        // set parameters
        __kernel_params(&tasks[k].kp, _C, _A, _B, alpha, beta, flags, K, jS, jL, iR, iE,
                        conf->kb, conf->nb, conf->mb, conf->optflags);
        // init task
        armas_task2_init(&tasks[k].t, k, __compute_block2, &tasks[k].kp, &ready);
        // schedule
        armas_schedule(&tasks[k].t);
        k++;
      }
    }
  }
  
  // wait for tasks to finish
  armas_counter_wait(&ready);
  // 1. check that task worker count is zero on all tasks
  int refcnt;
  for (k = 0; k < 50; k++) {
    refcnt = 0;
    for (i = 0; i < nT; i++) {
      refcnt += tasks[i].t.wcnt;
    }
    if (refcnt == 0)
      break;
  }
  assert(refcnt == 0);
  // release task memory
  free(tasks);
  return 0;
}

#endif   // ENABLE_THREADS

// ----------------------------------------------------------------------------------------
// exported public functions


/**
 * @brief General matrix-matrix multiplication
 *
 * Computes
 *   - \f$ C = alpha \times A B + beta \times C \f$
 *   - \f$ C = alpha \times A^T B + beta \times C \f$  if _ARMAS_TRANSA_ is set
 *   - \f$ C = alpha \times A B^T + beta \times C \f$  if _ARMAS_TRANSB_ is set
 *   - \f$ C = alpha \times A^T B^T + beta \times C \f$ if _ARMAS_TRANSA_ and _ARMAS_TRANSB_ are set
 *
 * Uses \f$|A|\f$ if flag ARMAS_ABSA set and \f$|B|\f$ if flag ARMAS_ABSB is set.
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
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
 * @retval -1 Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int armas_x_mult(DTYPE beta, armas_x_dense_t *C,
                 DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
                 int flags, armas_conf_t *conf)
{
  int ok;

  if (armas_x_size(A) == 0 || armas_x_size(B) == 0 || armas_x_size(C) == 0)
    return  0;

  if (!conf)
    conf = armas_conf_default();

  // check consistency
  switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB|ARMAS_CTRANSA|ARMAS_CTRANSB)) {
  case ARMAS_TRANSA|ARMAS_TRANSB:
  case ARMAS_TRANSA|ARMAS_CTRANSB:
  case ARMAS_CTRANSA|ARMAS_CTRANSB:
  case ARMAS_CTRANSA|ARMAS_TRANSB:
    ok = C->rows == A->cols && C->cols == B->rows && A->rows == B->cols;
    break;
  case ARMAS_TRANSA:
  case ARMAS_CTRANSA:
    ok = C->rows == A->cols && C->cols == B->cols && A->rows == B->rows;
    break;
  case ARMAS_TRANSB:
  case ARMAS_CTRANSB:
    ok = C->rows == A->rows && C->cols == B->rows && A->cols == B->cols;
    break;
  default:
    ok = C->rows == A->rows && C->cols == B->cols && A->cols == B->rows;
    break;
  }
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

#if defined(ENABLE_THREADS)
  long nproc = armas_use_nproc(armas_x_size(C), conf);
  int colwise = C->rows <= C->cols;
  if (conf->optflags & (ARMAS_OBLAS_BLOCKED|ARMAS_OBLAS_TILED) && nproc > 1) {
    return __mult_schedule(nproc, colwise, C, A, B, alpha, beta, flags, conf);
  }
  // default is recursive scheduling of threads
  return __mult_threaded(0, nproc, colwise, C, A, B, alpha, beta, flags, conf);

#else
  // no threading
  mdata_t *_C = (mdata_t*)C;
  const mdata_t *_A = (const mdata_t *)A;
  const mdata_t *_B = (const mdata_t *)B;
  int K = flags & (ARMAS_TRANSA|ARMAS_CTRANSA) ? A->rows : A->cols;

  armas_cbuf_t *cbuf = armas_cbuf_get(conf);

  // if extended precision enabled and requested
  if (HAVE_EXT_PRECISION && (conf->optflags & ARMAS_OEXTPREC)) {
    __kernel_inner_ext(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                       conf->kb, conf->nb, conf->mb, cbuf);
    return 0;
  }
  
  // otherwise, normal precision here
  __kernel_inner(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                  conf->kb, conf->nb, conf->mb, cbuf);
  return 0;
#endif
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// indent-tabs-mode: nil
// End:
