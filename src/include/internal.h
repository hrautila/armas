
// Copyright (c) Harri Rautila, 2012-2019

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef ARMAS_INTERNAL_H
#define ARMAS_INTERNAL_H

#include <assert.h>
#include <string.h>
#include "armas.h"

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

enum armas_partition {
  ARMAS_PTOP = 0,
  ARMAS_PBOTTOM = 1,
  ARMAS_PLEFT = 2,
  ARMAS_PRIGHT = 3,
  ARMAS_PTOPLEFT = 4,
  ARMAS_PBOTTOMRIGHT = 5
};

typedef struct cache_s {
  DTYPE *Acpy;  // kb*mb
  DTYPE *Bcpy;  // kb*nb
  size_t ab_step;
  size_t KB;
  size_t NB;
  size_t MB;
  size_t rb;
  // for extended precision versions
  DTYPE *C0;    // kb*mb; kb >= nb;
  DTYPE *dC;    // kb*mb; kb >= nb
  size_t c_step;
  armas_cbuf_t *cbuf;
} cache_t;


static inline
int min(int a, int b) {
  return a < b ? a : b;
}

static inline
int max(int a, int b) {
  return a < b ? b : a;
}

#ifdef __OPTIMIZE__
/* if not compiled with -O0 */
#define fail_on_error(x)    x
#define require(x)
#else
/* if compiled with -O0 */
#define require(x)   assert(x)
#define fail_on_error(x)                        \
do {                                            \
    if ((x) < 0) {                              \
        abort();                                \
    }                                           \
} while(0)
#endif  /* __OPTIMIZE__ */
#ifdef DEBUG
#define A_DEBUG(a) do { a; } while (0)
#else
#define A_DEBUG(a)
#endif

extern void armas_cbuf_split2(
    armas_cbuf_t *cbuf, void **aptr, void **bptr, size_t *mb, size_t *nb, size_t *kb, size_t p);
extern void armas_cache_setup2(
    cache_t *cache, armas_cbuf_t *cbuf, size_t mb, size_t nb, size_t kb, size_t p);
extern void armas_cbuf_split3(
    armas_cbuf_t *cbuf, void **aptr, void **bptr, void **cptr, size_t *mb, size_t *nb, size_t *kb, size_t p);
extern void armas_cache_setup3(
    cache_t *cache, armas_cbuf_t *cbuf, size_t mb, size_t nb, size_t kb, size_t p);
extern void armas_cache_setup(cache_t *cache, armas_cbuf_t *cbuf, int three, size_t p);

extern DTYPE armas_x_dot_unsafe(const armas_x_dense_t *x, const armas_x_dense_t *y);

extern void armas_x_adot_unsafe(
    DTYPE *value, DTYPE alpha, const armas_x_dense_t *x, const armas_x_dense_t *y);

extern void armas_x_axpby_unsafe(
    DTYPE beta, armas_x_dense_t *y, DTYPE alpha, const armas_x_dense_t *x);

extern int armas_x_scale_unsafe(armas_x_dense_t *x, DTYPE alpha);

extern void armas_x_mvupdate_unb(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y, int flags);

extern void armas_x_mvupdate_rec(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y, int flags);

extern void armas_x_mvupdate_unsafe(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y, int flags);

extern void armas_x_mvmult_trm_unsafe(
    armas_x_dense_t *X, DTYPE alpha, const armas_x_dense_t *A, int flags);

extern void armas_x_mvmult_unsafe(
    DTYPE beta, armas_x_dense_t *y,
    DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *x, int flags);

extern void armas_x_mvmult_sym_unsafe(
    DTYPE beta, armas_x_dense_t *y,
    DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *x, int flags);


extern void armas_x_mvupdate_trm_unb(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *x, const armas_x_dense_t *y, int flags);

extern void armas_x_mvupdate_trm_rec(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *x, const armas_x_dense_t *y, int flags);

extern int armas_x_mult_kernel(
    DTYPE beta, armas_x_dense_t *C, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *B, int flags, cache_t *cache);

extern int armas_x_mult_kernel_nc(
    armas_x_dense_t *C, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *B, int flags, cache_t *cache);

extern void armas_x_mult_sym_unsafe(
    DTYPE beta, armas_x_dense_t *C, DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
    int flags, int K, cache_t *cache);

extern void armas_x_mult_kernel_inner(
    armas_x_dense_t *Cblk, DTYPE alpha, const armas_x_dense_t *Ablk, const armas_x_dense_t *Bblk, int rb);

extern void armas_x_trmm_unb(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A,  int flags);

extern void armas_x_trmm_recursive(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *mcache);

extern void armas_x_trmm_blk(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *mcache);

extern void armas_x_mult_trm_unsafe(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *cache);

extern void armas_x_solve_unb(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags);

extern void armas_x_solve_recursive(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *mcache);

extern void armas_x_solve_blocked(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *mcache);

extern void armas_x_solve_trm_unsafe(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *cache);

// extended precision
extern int armas_x_ext_mult_kernel(
    DTYPE beta, armas_x_dense_t *C, DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
    int flags,  cache_t *cache);

extern int armas_x_ext_mult_kernel_nc(
    armas_x_dense_t *C, DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
    int flags, cache_t *cache);

extern void armas_x_ext_panel_dB_unsafe(
    armas_x_dense_t *C, armas_x_dense_t *dC, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *B, const armas_x_dense_t *dB, int flags, cache_t *cache);

extern void armas_x_ext_panel_dA_unsafe(
    armas_x_dense_t *C, armas_x_dense_t *dC, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *dA, const armas_x_dense_t *B, int flags, cache_t *cache);

extern void armas_x_ext_panel_unsafe(
    armas_x_dense_t *C, armas_x_dense_t *dC, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *B, int flags, cache_t *cache);

extern void armas_x_ext_mult_inner(
    armas_x_dense_t *C, armas_x_dense_t *dC, DTYPE alpha,
    const armas_x_dense_t *A, const armas_x_dense_t *B, int rb);

extern void armas_x_ext_scale_unsafe(
    armas_x_dense_t *C0, armas_x_dense_t *dC, DTYPE beta, const armas_x_dense_t *A);

extern DTYPE armas_x_ext_dot_unsafe(
    const armas_x_dense_t *X,  const armas_x_dense_t *Y);

extern void armas_x_ext_dot2_unsafe(
    DTYPE *h, DTYPE *l, const armas_x_dense_t *X, const armas_x_dense_t *Y);

extern void armas_x_ext_adot_unsafe(
    DTYPE *h, DTYPE *l, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y);

extern void armas_x_ext_adot_dx_unsafe(
    DTYPE *h, DTYPE *l, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *dX, const armas_x_dense_t *Y);

extern int armas_x_ext_axpy_unsafe(
    armas_x_dense_t *Y, DTYPE alpha, const armas_x_dense_t *X);

extern int armas_x_ext_axpby_unsafe(
    DTYPE beta, armas_x_dense_t *y, DTYPE alpha, const armas_x_dense_t *x);

extern int armas_x_ext_axpby_dx_unsafe(
    DTYPE beta, armas_x_dense_t *y, DTYPE alpha, DTYPE dx, const armas_x_dense_t *x);

extern ABSTYPE armas_x_ext_asum_unsafe(const armas_x_dense_t *y);

extern DTYPE armas_x_ext_sum_unsafe(const armas_x_dense_t *y);

extern int armas_x_ext_mvmult_unsafe(
    DTYPE beta, armas_x_dense_t *y, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *x, int flags);

extern void armas_x_ext_mvmult_dx_unsafe(
    DTYPE beta, armas_x_dense_t *y, armas_x_dense_t *dy, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *x, int flags);

extern int armas_x_ext_mvmult_trm_unsafe(
    armas_x_dense_t *x, DTYPE alpha, const armas_x_dense_t *A, int flags);

extern int armas_x_ext_mvmult_sym_unsafe(
    DTYPE beta, armas_x_dense_t *y, DTYPE alpha, const armas_x_dense_t *A,
    const armas_x_dense_t *x, int flags);

extern int armas_x_ext_mvsolve_trm_unsafe(
    armas_x_dense_t *x, armas_x_dense_t *dx, DTYPE alpha, const armas_x_dense_t *A, int flags);

extern int armas_x_ext_mvupdate_unsafe(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y);

extern int armas_x_ext_mvupdate_trm_unsafe(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y, int flags);

extern int armas_x_ext_mvupdate2_sym_unsafe(
    DTYPE beta, armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *X, const armas_x_dense_t *Y, int flags);

extern void armas_x_ext_mult_trm_unsafe(
    armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *mcache);

extern void armas_x_ext_solve_trm_unb_unsafe(
  armas_x_dense_t *B, armas_x_dense_t *dB, DTYPE alpha, const armas_x_dense_t *A, int flags);

extern void armas_x_ext_solve_trm_blk_unsafe(
  armas_x_dense_t *B, armas_x_dense_t *dB, DTYPE alpha, const armas_x_dense_t *A, int flags, cache_t *cache);

extern int armas_x_ext_solve_trm_unsafe(
  armas_x_dense_t *B, armas_x_dense_t *dB, DTYPE alpha, const armas_x_dense_t *A, int flags, armas_cbuf_t *cbuf);

#endif
