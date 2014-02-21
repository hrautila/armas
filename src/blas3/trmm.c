
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mult_trm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__trmm_unb) && defined(__trmm_recursive) && defined(__trmm_blk)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"


// Functions here implement various versions of TRMM operation.


static
void *__start_thread(void *arg) {
  kernel_param_t *kp = (kernel_param_t *)arg;
    switch (kp->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
    case ARMAS_SNAIVE:
      __trmm_unb(kp->C, kp->A, kp->alpha, kp->flags, kp->K, kp->S, kp->E);
      break;
    case ARMAS_RECURSIVE:
      __trmm_recursive(kp->C, kp->A, kp->alpha, kp->flags,
                       kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB);
      break;
    default:
      __trmm_blk(kp->C, kp->A, kp->alpha, kp->flags,
                 kp->K, kp->S, kp->E, kp->KB, kp->NB, kp->MB);
    }
  return arg;
}

static
int __mult_trm_threaded(int blknum, int nblk, 
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
    switch (conf->optflags & (ARMAS_SNAIVE|ARMAS_RECURSIVE)) {
    case ARMAS_SNAIVE:
      __trmm_unb(_B, _A, alpha, flags, A->cols, ir, ie);
      break;
    case ARMAS_RECURSIVE:
      __trmm_recursive(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb);
      break;
    default:
      __trmm_blk(_B, _A, alpha, flags, A->cols, ir, ie, conf->kb, conf->nb, conf->mb);
      break;
    }
    return 0;
  }

  // C = B, not used: beta = alpha, L, R = 0
  __kernel_params(&kp, _B, _A, 0, alpha, alpha, flags, A->cols, ir, 0, 0, ie,
                  conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  err = pthread_create(&th, NULL, __start_thread, &kp);
  if (err) {
    conf->error = -err;
    return -1;
  }
  // recursively invoke next block
  err = __mult_trm_threaded(blknum+1, nblk, B, A, alpha, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  return err;
}

/**
 * @brief Triangular matrix-matrix multiply
 *
 * If flag bit LEFT is set then computes 
 * > B = alpha*A*B\n
 * > B = alpha*A.T*B  if TRANSA or TRANS
 *
 * If flag bit RIGHT is set then computes
 * > B = alpha*B*A\n
 * > B = alpha*B*A.T  if TRANSA or TRANS
 *
 * The matrix A is upper (lower) triangular matrix if ARMAS_UPPER (ARMAS_LOWER) is
 * set. If matrix A is upper (lowert) then the strictly lower (upper) part is not
 * referenced. Flag bit UNIT indicates that matrix A is unit diagonal and the diagonal
 * entries are not accessed.
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
int __armas_mult_trm(__armas_dense_t *B, const __armas_dense_t *A, 
                      DTYPE alpha, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, ok, empty, blk;
  mdata_t *_B;
  const mdata_t *_A;

  if (!conf)
    conf = armas_conf_default();

  // check consistency
  switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
  case ARMAS_RIGHT:
    ok = B->cols == A->rows && A->cols == A->rows;
    empty = B->rows == 0 || A->cols == 0;
    break;
  case ARMAS_LEFT:
  default:
    ok = B->rows == A->cols && A->cols == A->rows;
    empty = A->rows == 0 || B->cols == 0;
    break;
  }
  if (empty)
    return 0;
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  _B = (mdata_t*)B;
  _A = (const mdata_t *)A;

  nproc = armas_use_nproc(__armas_size(B), conf);
  if (nproc == 1) {
    ie = flags & ARMAS_RIGHT ? B->rows : B->cols;
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
  }

  return __mult_trm_threaded(0, nproc, B, A, alpha, flags, conf);
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
