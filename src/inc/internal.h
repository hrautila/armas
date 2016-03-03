

// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef __ARMAS_INTERNAL_H
#define __ARMAS_INTERNAL_H

#include <string.h>
#include <armas/armas.h>

// maximum sizes

#ifndef MAX_KB
#define MAX_KB 192
#endif

#ifndef MAX_NB
#define MAX_NB 128
#endif

#ifndef MAX_MB
#define MAX_MB 128
#endif

// no recursion for vectors shorter than this
#ifndef MIN_MVEC_SIZE
#define MIN_MVEC_SIZE 256
#endif

#ifndef MIN_MBLOCK_SIZE
#define MIN_MBLOCK_SIZE 256
#endif

enum armas_partition {
  ARMAS_PTOP = 0,
  ARMAS_PBOTTOM = 1,
  ARMAS_PLEFT = 2,
  ARMAS_PRIGHT = 3,
  ARMAS_PTOPLEFT = 4,
  ARMAS_PBOTTOMRIGHT = 5
};


typedef struct mdata {
  DTYPE *md;
  int step;
} mdata_t;

typedef struct mvec {
  DTYPE *md;
  int inc;
} mvec_t;


typedef struct cache_s {
  mdata_t Acpy;  // kb*mb
  mdata_t Bcpy;  // kb*nb
  size_t KB;
  size_t NB;
  size_t MB;
  size_t rb;
  // for extended precision versions
  mdata_t C0;    // kb*mb; kb >= nb;
  mdata_t dC;    // kb*mb; kb >= nb
  armas_cbuf_t *cbuf;
} cache_t;

// parameter block for kernel function invocation
typedef struct kernel_param {
  mdata_t C;
  mdata_t A;
  mdata_t B;
  DTYPE alpha;
  DTYPE beta;
  int flags;
  int K;
  int S;
  int L;
  int R;
  int E;
  int KB;
  int NB;
  int MB;
  int optflags;
} kernel_param_t;

static inline
kernel_param_t * __kernel_params(kernel_param_t *p,
                                 mdata_t *C, const mdata_t *A, const mdata_t *B,
                                 DTYPE alpha, DTYPE beta, int flags,
                                 int K, int S, int L, int R, int E,
                                 int KB, int NB, int MB, int opts)
{
  p->C = C ? *C : (mdata_t){(DTYPE *)0, 0}; 
  p->A = A ? *A : (mdata_t){(DTYPE *)0, 0}; 
  p->B = B ? *B : (mdata_t){(DTYPE *)0, 0}; 
  p->alpha = alpha;
  p->beta = beta;
  p->flags = flags;
  p->K = K;
  p->S = S; p->L = L;
  p->R = R; p->E = E;
  p->KB = KB; p->NB = NB; p->MB = MB;
  p->optflags = opts;
  return p;
}


// thread parameters for recursive thread invocations
typedef struct block_args_s 
{
  kernel_param_t *kp;
  armas_cbuf_t *cbuf;
} block_args_t;


static inline
int min(int a, int b) {
  return a < b ? a : b;
}

static inline
int max(int a, int b) {
  return a < b ? b : a;
}


// Calculate how many row/column blocks are needed with blocking size WB.
static inline
int blocking(int M, int N, int WB, int *nM, int *nN)
{
  *nM = M/WB;
  *nN = N/WB;
  if (M % WB > WB/10) {
    *nM += 1;
  }
  if (N % WB > WB/10) {
    *nN += 1;
  }
  return (*nM)*(*nN);
}

// compute start of k'th out of nblk block when block size wb and total is K
// requires: K/wb == nblk or K/wb == nblk-1
static inline
int block_index(int k, int nblk, int wb, int K)
{
  return k == nblk ? K : k*wb;
}

// compute start of i'th block out of r blocks in sz elements
static inline
int __block_index4(int i, int n, int sz) {
    if (i == n) {
        return sz;
    }
    return i*sz/n - ((i*sz/n) & 0x3);
}

static inline
int __block_index2(int i, int n, int sz) {
    if (i == n) {
        return sz;
    }
    return i*sz/n - ((i*sz/n) & 0x1);
}

// block is empty if data pointer is null or step is zero
static inline
int __empty_blk(const mdata_t *A)
{
  return ! A->md || A->step == 0;
}

// make A subblock of B, starting from B[r,c] 
static inline
mdata_t *__subblock(mdata_t *A, const mdata_t *B, int r, int c)
{
  A->md = &B->md[r + c*B->step];
  A->step = B->step;
  return A;
}

static inline
mdata_t *__subblock_t(mdata_t *A, const mdata_t *B, int r, int c, int trans)
{
  A->md = trans ? &B->md[c+r*B->step] : &B->md[r + c*B->step];
  A->step = B->step;
  return A;
}

// make X subvector of Y, starting at Y[n]
static inline
mvec_t *__subvector(mvec_t *X, const mvec_t *Y, int n)
{
  X->md = &Y->md[n*Y->inc];
  X->inc = Y->inc;
  return X;
}

static inline
mvec_t *__rowvec(mvec_t *X, const mdata_t *A, int r, int c)
{
  X->md = &A->md[r + c*A->step];
  X->inc = A->step;
  return X;
}

static inline
mvec_t *__colvec(mvec_t *X, const mdata_t *A, int r, int c)
{
  X->md = &A->md[r + c*A->step];
  X->inc = 1;
  return X;
}

// return A[r, c];
static inline
DTYPE __get(const mdata_t *A, int r, int c)
{
  return A->md[r+c*A->step];
}

// set A[r, c] = v;
static inline
void __set(mdata_t *A, int r, int c, DTYPE v)
{
  A->md[r+c*A->step] = v;
}

// A[r, c] += v;
static inline
void __add(mdata_t *A, int r, int c, DTYPE v)
{
  A->md[r+c*A->step] += v;
}

// A[k]
static inline
DTYPE __get_at(const mvec_t *A, int k)
{
    return A->md[k*A->inc];
}

// A[k] = v
static inline
void __set_at(mvec_t *A, int k, DTYPE v)
{
    A->md[k*A->inc] = v;
}

// A[k] += v
static inline
void __add_at(mvec_t *A, int k, DTYPE v)
{
    A->md[k*A->inc] += v;
}

static inline
void __vec_scale(mvec_t *X, int n, DTYPE val)
{
  register int i;
  for (i = 0; i < n; i++) {
    X->md[i*X->inc] *= val;
  }
}

#ifdef DEBUG
#define __DEBUG(a) do { a; } while (0)
#else
#define __DEBUG(a)
#endif

static inline
void __blk_merge_ext(mdata_t *C, mdata_t *C0, mdata_t *dC, int nR, int nC)
{
    int i, j;
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nR-1; i += 2) {
      C->md[(i+0)+j*C->step] = C0->md[(i+0)+j*C0->step] + dC->md[(i+0)+j*dC->step];
      C->md[(i+1)+j*C->step] = C0->md[(i+1)+j*C0->step] + dC->md[(i+1)+j*dC->step];
    }
    if (i != nR) {
      C->md[(i+0)+j*C->step] = C0->md[(i+0)+j*C0->step] + dC->md[(i+0)+j*dC->step];
    }
  }
}

static inline
void clear_blk(mdata_t *A, int nR, int nC)
{
  int j;
  for (j = 0; j < nC; j++) {
    memset(&A->md[j*A->step], 0, nR*sizeof(DTYPE));
  }
}

static inline
void ext_merge(mdata_t *A, mdata_t *B, int nR, int nC)
{
  int i, j;
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      A->md[i+j*A->step] += B->md[i+j*B->step];
    }
  }
}

extern void armas_cbuf_split2(armas_cbuf_t *cbuf, void **aptr, 
                              void **bptr, size_t *mb, size_t *nb, size_t *kb, size_t p);
extern void armas_cache_setup2(cache_t *cache, armas_cbuf_t *cbuf, 
                               size_t mb, size_t nb, size_t kb, size_t p);
extern void armas_cbuf_split3(armas_cbuf_t *cbuf, void **aptr, void **bptr,
                              void **cptr, size_t *mb, size_t *nb, size_t *kb, size_t p);
extern void armas_cache_setup3(cache_t *cache, armas_cbuf_t *cbuf, 
                               size_t mb, size_t nb, size_t kb, size_t p);

// forward declarations of some element type dependent functions
//extern void __SCALE(DTYPE *X, int ldX, const DTYPE beta, int M, int N);
//extern void __tile_add(DTYPE *X, int ldX, const DTYPE beta, int M, int N);
//extern void __PRINT_TILE(const DTYPE *X, int ldX, int M, int N, const char *s, const char *fmt);
extern void __blk_scale(mdata_t *X, const DTYPE beta, int M, int N);
extern void __blk_add(mdata_t *X, const DTYPE beta, int M, int N);
extern void __blk_print(const mdata_t *X, int M, int N, const char *s, const char *fmt);
extern void __vec_print(const mvec_t *X, int N, const char *s, const char *fmt);

extern
void __blk_scale_ext(mdata_t *C0, mdata_t *dC, const mdata_t *A, DTYPE beta, int nR, int nC);

extern
void __kernel_colblk_inner(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                           DTYPE alpha, int nJ, int nR, int nP, int rb);

extern
void __kernel_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                     DTYPE alpha, int flags,
                                     int P, int nSL, int nRE, cache_t *cache);

extern
void __kernel_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                    DTYPE alpha, DTYPE beta, int flags,
                                    int P, int S, int L, int R, int E, cache_t *cache);

extern
void __kernel_inner(mdata_t *C, const mdata_t *A, const mdata_t *B,
                    DTYPE alpha, DTYPE beta, int flags,
                    int P, int S, int L, int R, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);

extern
void __rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                 DTYPE alpha, DTYPE beta, int flags,  int P, int nC, cache_t *cache);

extern
void __trmm_unb(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int N, int S, int E);

extern 
void __trmm_blk(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);

extern
void __trmm_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);

extern
void __trmm_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache);


extern
void __solve_right_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E);

extern
void __solve_left_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E);

extern
void __solve_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);

extern
void __solve_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache);

extern
void __solve_blocked(mdata_t *B, const mdata_t *A, DTYPE alpha,
                     int flags, int N, int S, int E, int KB, int NB, int MB, armas_cbuf_t *cbuf);


extern
void __update_ger_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                            DTYPE alpha, int flags, int N, int M);

extern
void __update_trmv_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int flags, int N, int M);

extern
void __update_trmv_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N, int M);

extern
void __gemv_recursive(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                      DTYPE alpha, DTYPE beta, int flags, int S, int L, int R, int E);

extern
void __gemv(mvec_t *Y, const mdata_t *A, const mvec_t *X,
            DTYPE alpha, int flags, int M, int N);

extern
int __gemv_ext_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                   DTYPE alpha, DTYPE beta, int flags, int nX, int nY);

extern
void __gemv_update_ext_unb(mvec_t *Y, mvec_t *dY, const mdata_t *A, const mvec_t *X,
                           const mvec_t *dX, int sign, int flags, int nX, int nY);


extern 
void __kernel_ext_colblk_inner(mdata_t *Cblk, mdata_t *dC, const mdata_t *Ablk, 
                               const mdata_t *Bblk, DTYPE alpha, int nJ, int nR, int nP, int rb);

extern
void __kernel_ext_panel_inner(mdata_t *C, mdata_t *dC, const mdata_t *A, const mdata_t *B,
                              DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache);

extern
void __kernel_ext_panel_inner_dA(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *dA, const mdata_t *B,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache);

extern 
void __kernel_ext_panel_inner_dB(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *B, const mdata_t *dB,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache);

extern
void __kernel_ext_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                         DTYPE alpha, int flags, int P, int nSL, int nRE,
                                         cache_t *cache);

extern
void __kernel_ext_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                        DTYPE alpha, DTYPE beta, int flags, int P, int S, int L,
                                        int R, int E, cache_t *cache);

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
