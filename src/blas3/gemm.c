
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * @defgroup blas3 BLAS level 3 functions
 */
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mult) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_inner)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"


static
void *__start_thread(void *arg) {
  kernel_param_t *kp = (kernel_param_t *)arg;
  __kernel_inner(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                 kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB);
  return arg;
}

// compute recursively in nblk threads
static
int __mult_threaded(int blknum, int nblk, int colwise, 
                     __armas_dense_t *C,
                     const __armas_dense_t *A,
                     const __armas_dense_t *B,
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

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // last block immediately
  if (blknum == nblk-1) {
    if (colwise) {
      ir = __block_index4(blknum, nblk, C->cols);
      ie = __block_index4(blknum+1, nblk, C->cols);
      __kernel_inner(_C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                     conf->kb, conf->nb, conf->mb);
    } else {
      ir = __block_index4(blknum, nblk, C->rows);
      ie = __block_index4(blknum, nblk, C->rows);
      __kernel_inner(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, ir, ie,
                     conf->kb, conf->nb, conf->mb);
    }
    return 0;
  }

  // initialize kernel parameters for this block
  if (colwise) {
      ir = __block_index4(blknum, nblk, C->cols);
      ie = __block_index4(blknum+1, nblk, C->cols);
      __kernel_params(&kp, _C, _A, _B, alpha, beta, flags, K, ir, ie, 0, C->rows,
                      conf->kb, conf->nb, conf->mb, conf->optflags);
  } else {
      ir = __block_index4(blknum, nblk, C->rows);
      ie = __block_index4(blknum, nblk, C->rows);
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
  err = __mult_threaded(blknum+1, nblk, colwise, C, A, B, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  return err;
}

// ----------------------------------------------------------------------------------------
// exported public functions


/**
 * @brief General matrix-matrix multiplication
 *
 * Computes
 * > C = alpha*A*B + beta*C\n
 * > C = alpha*A.T*B + beta*C   if TRANSA\n
 * > C = alpha*A*B.T + beta*C   if TRANSB\n
 * > C = alpha*A.T*B.T + beta*C if TRANSA and TRANSB
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
int __armas_mult(__armas_dense_t *C, const __armas_dense_t *A, const __armas_dense_t *B,
                 DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, mb, ok, n;
  mdata_t *_C;
  const mdata_t *_A, *_B;

  if (C->rows == 0 || C->cols == 0 || A->cols == 0 || B->rows == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();

  conf->error = 0;

  // check consistency
  switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
  case ARMAS_TRANSA|ARMAS_TRANSB:
    ok = C->rows == A->cols && C->cols == B->rows && A->rows == B->cols;
    break;
  case ARMAS_TRANSA:
    ok = C->rows == A->cols && C->cols == B->cols && A->rows == B->rows;
    break;
  case ARMAS_TRANSB:
    ok = C->rows == A->rows && C->cols == B->rows && A->cols == B->cols;
    break;
  default:
    ok = C->rows == A->rows && C->cols == B->cols && A->cols == B->rows;
    break;
  }
  if (! ok) {
    conf->error = 1;
    return -1;
  }

  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  nproc = armas_use_nproc(__armas_size(C), conf);

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // if only one thread, just do it
  if (nproc == 1) {
    __kernel_inner(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                   conf->kb, conf->nb, conf->mb);
    return 0;
  }

  int colwise = 20*nproc < C->cols;
  return __mult_threaded(0, nproc, colwise, C, A, B, alpha, beta, flags, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
