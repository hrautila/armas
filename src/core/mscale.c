
// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__blk_scale) && defined(__blk_add) && defined(__blk_print) && defined(__vec_print)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------



#include "internal.h"


// Scale a tile of M rows by N columns with leading index ldX.
static
void __SCALE(DTYPE *X, int ldX, const DTYPE beta, int M, int N)
{
  register int i, j;
  if (beta == 1.0) {
    return;
  }

  // set to zero
  if (beta == 0.0) {
    for (j = 0; j < N-3; j += 4) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
        X[i+(j+1)*ldX] = 0.0;
        X[i+(j+2)*ldX] = 0.0;
        X[i+(j+3)*ldX] = 0.0;
      }
    }
    if (j == N) 
      return;
    for (; j < N; j++) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
      }
    }
    return;
  }
  // scale here
  for (j = 0; j < N-3; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
      X[i+(j+1)*ldX] *= beta;
      X[i+(j+2)*ldX] *= beta;
      X[i+(j+3)*ldX] *= beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
    }
  }
}

void __blk_scale(mdata_t *X, DTYPE const beta, int M, int N)
{
  __SCALE(X->md, X->step, beta, M, N);
}

// SHIFT (= A + I*beta) a tile of M rows by N columns with leading index ldX.
static
void __tile_add(DTYPE *X, int ldX, const DTYPE beta, int M, int N)
{
  register int i, j;
  if (beta == 0.0) {
    return;
  }

  // shift here
  for (j = 0; j < N-3; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] += beta;
      X[i+(j+1)*ldX] += beta;
      X[i+(j+2)*ldX] += beta;
      X[i+(j+3)*ldX] += beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] += beta;
    }
  }
}

void __blk_add(mdata_t *X, const DTYPE beta, int M, int N)
{
  __tile_add(X->md, X->step, beta, M, N);
}

static
void __tile_print(const DTYPE *d, int ldD, int nR, int nC, const char *s, const char *efmt)
{
  int i, j;

  if (s)
    printf("%s\n", s);
  if (!efmt)
    efmt = __DATA_FORMAT;

  for (i = 0; i < nR; i++) {
    printf("[");
    for (j = 0; j < nC; j++) {
      if (j > 0)
        printf(", ");
      printf(efmt, __PRINTABLE(d[j*ldD+i]));
    }
    printf("]\n");
  }
  printf("\n");
}

void __blk_print(const mdata_t *A, int nR, int nC, const char *s, const char *efmt)
{
  __tile_print(A->md, A->step, nR, nC, s, efmt);
}
  
void __vec_print(const mvec_t *X, int N, const char *s, const char *efmt)
{
  register int i;
  if (s)
    printf("%s\n", s);
  if (!efmt)
    efmt = __DATA_FORMAT;
  printf("[");
  for (i = 0; i < N; i++) {
    if (i > 0)
      printf(", ");
    printf(efmt, __PRINTABLE(X->md[i*X->inc]));
  }
  printf("]\n");
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
