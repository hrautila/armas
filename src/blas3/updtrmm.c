
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
// this file provides following type independent functions
#if defined(__armas_update_trm) && defined(__update_trm_blk)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__kernel_colwise_inner_no_scale)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "mvec_nosimd.h"

/*
 * update diagonal block
 *
 *  l00           a00 a01   b00 b01 b02    u00 u01 u02
 *  l10 l11       a10 a11   b10 b11 b12        u11 u12
 *  l20 l21 l22   a20 a21                          u22
 *
 */
 static
 void __update_trm_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                        DTYPE alpha, DTYPE beta,
                        int flags,  int P, int nC, int nR, cache_t *cache)
{
  register int i, incA, incB, transA, transB;
  mdata_t A0, B0, C0;

  incA = flags & ARMAS_TRANSA ? A->step : 1;
  incB = flags & ARMAS_TRANSB ? 1 : B->step;

  __subblock(&A0, A, 0, 0);
  __subblock(&B0, B, 0, 0);

  if (flags & ARMAS_UPPER) {
    // index by row
    int M = min(nC, nR);
    for (i = 0; i < M; i++) {   
      // scale the target row with beta
      __subblock(&C0, C, i, i);
      __vscale(C0.md, C0.step, beta, nC-i);

      // update one row of C  (nC-i columns, 1 row)
      __kernel_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags,
                                      P, nC-i, 1, cache); 
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  } else {
    // index by column
    int N = min(nC, nR);
    for (i = 0; i < N; i++) {
      __subblock(&C0, C, i, i);
      // scale the target column with beta
      __vscale(C0.md, 1, beta, nR-i);
      // update one column of C  (1 column, nR-i rows)
      __kernel_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags,
                                      P, 1, nR-i, cache);
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  }
}

static
void __update_trm_naive(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        DTYPE alpha, DTYPE beta, int flags,
                        int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <=0 || P <= 0) {
    return;
  }

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  __update_trm_diag(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
}

 static
 void __update_upper_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                               DTYPE alpha, DTYPE beta,
                               int flags,  int P, int N, int M, cache_t *cache)
{
  mdata_t c0, a0, b0;

  // can upper triangular (M == N) or upper trapezoidial (N > M)
  int nb = min(N, M);

  if (nb < MIN_MBLOCK_SIZE) {
    __update_trm_diag(C, A, B, alpha, beta, flags, P, N, M, cache);
    return;
  }

  // upper LEFT diagonal (square block)
  __subblock(&c0, C, 0, 0);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, 0, 0);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  } else {
    __update_upper_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  }

  // upper RIGHT square (nb/2 rows, nb-nb/2 cols)
  __subblock(&c0, C, 0, nb/2);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, (flags & ARMAS_TRANSB ? nb/2 : 0), (flags & ARMAS_TRANSB ? 0 : nb/2));
  __kernel_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb-nb/2, 0, nb/2, cache);

  // lower RIGHT diagonal
  __subblock(&c0, C, nb/2, nb/2);
  __subblock(&a0, A, (flags & ARMAS_TRANSA ? 0 : nb/2), (flags & ARMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, (flags & ARMAS_TRANSB ? nb/2 : 0), (flags & ARMAS_TRANSB ? 0 : nb/2));
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  } else {
    __update_upper_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  }
  if (M >= N)
    return;
  
  // right trapezoidal part 
  __subblock(&c0, C, 0, nb);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, (flags & ARMAS_TRANSB ? nb : 0), (flags & ARMAS_TRANSB ? 0 : nb));
  __kernel_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, N-nb, 0, nb, cache);
}

 static
 void __update_lower_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                               DTYPE alpha, DTYPE beta,
                               int flags,  int P, int N, int M, cache_t *cache)
{
  mdata_t c0, a0, b0;

  // can be lower triangular (M == N) or lower trapezoidial (M > N)
  int nb = min(M, N);

  //printf("__update_lower_rec: M=%d, N=%d, nb=%d\n", M, N, nb);
  if (nb < MIN_MBLOCK_SIZE) {
    __update_trm_diag(C, A, B, alpha, beta, flags, P, N, M, cache);
    return;
  }

  // upper LEFT diagonal
  __subblock(&c0, C, 0, 0);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, 0, 0);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  } else {
    __update_lower_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  }

  // lower LEFT square (nb-nb/2 rows, nb/2 cols)
  __subblock(&c0, C, nb/2, 0);
  __subblock(&a0, A, (flags & ARMAS_TRANSA ? 0 : nb/2), (flags & ARMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, 0,    0);
  __kernel_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb/2, 0, nb-nb/2, cache);

  // lower RIGHT diagonal
  __subblock(&c0, C, nb/2, nb/2);
  __subblock(&a0, A, (flags & ARMAS_TRANSA ? 0 : nb/2), (flags & ARMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, (flags & ARMAS_TRANSB ? nb/2 : 0), (flags & ARMAS_TRANSB ? 0 : nb/2));
  //__subblock(&a0, A, nb/2, 0);
  //__subblock(&b0, B, 0,  nb/2);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  } else {
    __update_lower_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  }
  if (M <= N)
    return;

  printf("__update_lower_rec: M > N: M-nb=%d rows \n", M-nb);
  // lower trapezoidial part
  __subblock(&c0, C, nb, 0);
  __subblock(&a0, A, (flags & ARMAS_TRANSA ? 0 : nb), (flags & ARMAS_TRANSA ? nb : 0));
  __subblock(&b0, B, 0,  0);
  __kernel_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb, 0, M-nb, cache);
}

static
void __update_trm_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            DTYPE alpha, DTYPE beta, int flags,
                            int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  //printf("__update_trm_rec: S=%d, L=%d, R=%d, E=%d\n", S, L, R, E);

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  if (flags & ARMAS_UPPER) {
    __update_upper_recursive(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
  } else {
    __update_lower_recursive(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
  }
}

/*
 * Generic triangular matrix update:
 *      C = beta*op(C) + alpha*A*B
 *      C = beta*op(C) + alpha*A*B.T
 *      C = beta*op(C) + alpha*A.T*B
 *      C = beta*op(C) + alpha*A.T*B.T
 *
 * Some conditions on parameters that define the updated block:
 * 1. S == R && E == L 
 *    matrix is triangular square matrix
 * 2. S == R && L >  E
 *    matrix is trapezoidial with upper trapezoidial part right of triangular part
 * 3. S == R && L <  E
 *    matrix is trapezoidial with lower trapezoidial part below triangular part
 * 4. S != R && S >  E
 *    update is only to upper trapezoidial part right of triangular block
 * 5. S != R && R >  L
 *    update is only to lower trapezoidial part below triangular block
 * 6. S != R
 *    inconsistent update block spefication, will not do anything
 *            
 */
void __update_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      DTYPE alpha, DTYPE beta, int flags,
                      int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  register int i, j, nI, ar, ac, br, bc, N, M;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  //printf("__update_trm_blk: S=%d, L=%d, R=%d, E=%d\n", S, L, R, E);

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  if ( S != R && (S <= E || R <= L)) {
    // inconsistent update configuration
    return;
  }
  if (flags & ARMAS_UPPER) {
    // by rows; M is the last row; L-S is column count; implicitely S == R
    M = min(L, E);
    for (i = R; i < M; i += NB) {
      nI = M - i < NB ? M - i : NB;
    
      // 1. update block on diagonal (square block)
      br = flags & ARMAS_TRANSB ? i : 0;
      bc = flags & ARMAS_TRANSB ? 0 : i;
      ar = flags & ARMAS_TRANSA ? 0 : i;
      ac = flags & ARMAS_TRANSA ? i : 0;

      //printf("i=%dm nI=%d, L-i=%d, L-i-nI=%d\n", i, nI, L-i, L-i-nI);
      __subblock(&Cd, C, i,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_upper_recursive(&Cd, &Ad, &Bd, alpha, beta, flags, P,
                               nI, nI, &cache);

      // 2. update right of the diagonal block (rectangle, nI rows)
      br = flags & ARMAS_TRANSB ? i+nI : 0;
      bc = flags & ARMAS_TRANSB ? 0    : i+nI;
      ar = flags & ARMAS_TRANSA ? 0    : i;
      ac = flags & ARMAS_TRANSA ? i    : 0;

      __subblock(&Cd, C, i,  i+nI);
      __subblock(&Ad, A, ar, ac);
      __subblock(&Bd, B, br, bc);
      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                     P, 0, L-i-nI, 0, nI, &cache);

    }
   } else {
    // by columns; N is the last column, E-R is row count;
    N = min(L, E);
    for (i = S; i < N; i += NB) {
      nI = N - i < NB ? N - i : NB;
    
      // 1. update on diagonal (square block)
      br = flags & ARMAS_TRANSB ? i : 0;
      bc = flags & ARMAS_TRANSB ? 0 : i;
      ar = flags & ARMAS_TRANSA ? 0 : i;
      ac = flags & ARMAS_TRANSA ? i : 0;
      __subblock(&Cd, C, i, i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_lower_recursive(&Cd, &Ad, &Bd, alpha, beta, flags,
                               P, nI, nI, &cache);

      // 2. update block below the diagonal block (rectangle, nI columns)
      br = flags & ARMAS_TRANSB ? i    : 0;
      bc = flags & ARMAS_TRANSB ? 0    : i;
      ar = flags & ARMAS_TRANSA ? 0    : i+nI;
      ac = flags & ARMAS_TRANSA ? i+nI : 0;
      __subblock(&Cd, C, i+nI,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __kernel_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                     P, 0, nI, 0, E-i-nI, &cache);
    }
  }
}

static
void *__start_thread(void *arg) {
  kernel_param_t *kp = (kernel_param_t *)arg;

  switch (kp->optflags) {
  case ARMAS_SNAIVE:
    __update_trm_naive(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                       kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB);
    break;
  case ARMAS_RECURSIVE:
    __update_trm_recursive(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                           kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB);
    break;
  default:
    __update_trm_blk(kp->C, kp->A, kp->B, kp->alpha, kp->beta, kp->flags,
                     kp->K, kp->S, kp->L, kp->R, kp->E, kp->KB, kp->NB, kp->MB);
  }
  return arg;
}

static
int __update_trm_threaded(int blk, int nblk, __armas_dense_t *C,
                          const __armas_dense_t *A, const __armas_dense_t *B,
                          DTYPE alpha, DTYPE beta, int flags,
                          armas_conf_t *conf)
{
  int rs, re, cs, ce, err;
  mdata_t *_C, C0, A0, B0;
  const mdata_t *_A, *_B;
  pthread_t th;
  kernel_param_t kp;

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

  int K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // shift the start point to top-left corner [rs,cs] of this block. Block size
  //   UPPER:  [rows/nblk-rs, cols-ce] 
  //   LOWER:  [cols/nblk-cs, rows-rs] 
  __subblock(&C0, _C, rs, cs);
  __subblock(&A0, _A, (flags & ARMAS_TRANSA ? 0 : rs), (flags & ARMAS_TRANSA ? rs : 0));
  __subblock(&B0, _B, (flags & ARMAS_TRANSB ? cs : 0), (flags & ARMAS_TRANSB ? 0 : cs));

  if (blk == nblk-1) {
    switch (conf->optflags) {
    case ARMAS_SNAIVE:
      __update_trm_naive(&C0, &A0, &B0, alpha, beta, flags,
                         K, 0, ce-cs, 0, re-rs, conf->kb, conf->nb, conf->mb);
      break;
    case ARMAS_RECURSIVE:
      __update_trm_recursive(&C0, &A0, &B0, alpha, beta, flags,
                             K, 0, ce-cs, 0, re-rs, conf->kb, conf->nb, conf->mb);
      break;
    default:
      __update_trm_blk(&C0, &A0, &B0, alpha, beta, flags,
                       K, 0, ce-cs, 0, re-rs, conf->kb, conf->nb, conf->mb);
    }
    return 0;
  }

  // setup thread parameters
  __kernel_params(&kp, &C0, &A0, &B0,
                  alpha, beta, flags, K, 0, ce-cs, 0, re-rs,
                  conf->kb, conf->nb, conf->mb, conf->optflags);

  // create new thread to compute this block
  err = pthread_create(&th, NULL, __start_thread, &kp);
  if (err) {
    conf->error = -err;
    return -1;
  }
  // recursively invoke next block
  err = __update_trm_threaded(blk+1, nblk, C, A, B, alpha, beta, flags, conf);
  // wait for this block to finish
  pthread_join(th, NULL);
  return err;
}


/**
 * @brief Triangular or trapezoidial matrix rank-k update
 *
 * Computes
 * > C = beta*C + alpha*A*B\n
 * > C = beta*C + alpha*A.T*B   if TRANSA\n
 * > C = beta*C + alpha*A*B.T   if TRANSB\n
 * > C = beta*C + alpha*A.T*B.T if TRANSA and TRANSB
 *
 * Matrix C is upper (lower) triangular or trapezoidial if flag bit
 * ARMAS_UPPER (ARMAS_LOWER) is set. If matrix is upper (lower) then
 * the strictly lower (upper) part is not referenced.
 *
 * @param[in,out] C triangular/trapezoidial result matrix
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
int __armas_update_trm(__armas_dense_t *C,
                       const __armas_dense_t *A, const __armas_dense_t *B,
                       DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  long nproc;
  int K, ir, ie, mb, ok, n;
  mdata_t *_C;
  const mdata_t *_A, *_B;

  if (__armas_size(C) == 0 || __armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;

  if (!conf)
    conf = armas_conf_default();
  
  switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
  case ARMAS_TRANSA|ARMAS_TRANSB:
    ok = A->rows == B->cols && C->cols == B->rows;
    break;
  case ARMAS_TRANSA:
    ok = A->rows == B->rows && C->cols == B->cols;
    break;
  case ARMAS_TRANSB:
    ok = A->cols == B->cols && C->cols == B->rows;
    break;
  default:
    ok = A->cols == B->rows && C->cols == B->cols;
    break;
  }
  if (!ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  _C = (mdata_t*)C;
  _A = (const mdata_t *)A;
  _B = (const mdata_t *)B;

  nproc = armas_use_nproc(__armas_size(C), conf);

  K = flags & ARMAS_TRANSA ? A->rows : A->cols;

  // if only one thread, just do it
  if (nproc == 1) {
    switch (conf->optflags) {
    case ARMAS_SNAIVE:
      __update_trm_naive(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                         conf->kb, conf->nb, conf->mb);
      break;
    case ARMAS_RECURSIVE:
      __update_trm_recursive(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                             conf->kb, conf->nb, conf->mb);
      break;
    default:
      __update_trm_blk(_C, _A, _B, alpha, beta, flags, K, 0, C->cols, 0, C->rows,
                       conf->kb, conf->nb, conf->mb);
    }
    return 0;
  }
  return __update_trm_threaded(0, nproc, C, A, B, alpha, beta, flags, conf);
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
