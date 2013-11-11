

// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef __ARMAS_INTERNAL_H
#define __ARMAS_INTERNAL_H

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
#define MIN_MVEC_SIZE 48
#endif

#ifndef MIN_MBLOCK_SIZE
#define MIN_MBLOCK_SIZE 48
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

typedef struct cache_buffer {
  mdata_t *Acpy;
  mdata_t *Bcpy;
  int KB;
  int NB;
  int MB;
} cache_t;

// parameter block for kernel function invocation
typedef struct kernel_param {
  mdata_t *C;
  const mdata_t *A;
  const mdata_t *B;
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
  p->C = C; p->A = A; p->B = B;
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

static inline
int min(int a, int b) {
  return a < b ? a : b;
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

#ifdef DEBUG
#define __DEBUG(a) do { a; } while (0)
#else
#define __DEBUG(a)
#endif

// forward declarations of some element type dependent functions
//extern void __SCALE(DTYPE *X, int ldX, const DTYPE beta, int M, int N);
//extern void __tile_add(DTYPE *X, int ldX, const DTYPE beta, int M, int N);
//extern void __PRINT_TILE(const DTYPE *X, int ldX, int M, int N, const char *s, const char *fmt);
extern void __blk_scale(mdata_t *X, const DTYPE beta, int M, int N);
extern void __blk_add(mdata_t *X, const DTYPE beta, int M, int N);
extern void __blk_print(const mdata_t *X, int M, int N, const char *s, const char *fmt);
extern void __vec_print(const mvec_t *X, int N, const char *s, const char *fmt);

extern
void __kernel_colblk_inner(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                           DTYPE alpha, int nJ, int nR, int nP);

extern
void __kernel_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                     DTYPE alpha, int flags,
                                     int P, int nSL, int nRE, cache_t *cache);

extern
void __kernel_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                    DTYPE alpha, DTYPE beta, int flags,
                                    int P, int S, int L, int R, int E, cache_t *cache);

extern
void __rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                 DTYPE alpha, DTYPE beta, int flags,  int P, int nC, cache_t *cache);

extern
void __trmm_unb(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int N, int S, int E);

extern 
void __trmm_blk(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                int N, int S, int E, int KB, int NB, int MB);

extern
void __trmm_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E, int KB, int NB, int MB);

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
                       int flags, int N, int S, int E, int KB, int NB, int MB);

extern
void __solve_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache);

extern
void __solve_blocked(mdata_t *B, const mdata_t *A, DTYPE alpha,
                     int flags, int N, int S, int E, int KB, int NB, int MB);


extern
void __update_ger_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                            DTYPE alpha, int flags, int N, int M);

extern
void __update_trmv_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int flags, int N, int M);

extern
void __update_trmv_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N, int M);

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
