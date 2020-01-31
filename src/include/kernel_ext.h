
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _ARMAS_KERNEL_EXT_H
#define _ARMAS_KERNEL_EXT_H 1

#include "eft.h"

// matrix-matrix multiplication primitives
#if defined(FLOAT32)
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
  #if defined(__x86_64__)
    #if defined(__AVX__)
      #include "x86_64/mult_ext_avx_f32.h"
    #else
      #include "nosimd/mult_ext_nosimd.h"
    #endif
  #else
      #include "nosimd/mult_ext.h"
  #endif

#else
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
  #if defined(__AVX__)
    #include "x86_64/mult_ext_avx_f64.h"
  #else
    #include "nosimd/mult_ext.h"
  #endif

#endif


// update 4 columns of C;
static inline
void __CMULT4EXT(armas_x_dense_t *C, armas_x_dense_t *dC, const armas_x_dense_t *A,
                 const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *c2, *c3, *d0, *d1, *d2, *d3;
  const DTYPE *a0;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  const DTYPE *b1 = &B->elems[(col+1)*B->step];
  const DTYPE *b2 = &B->elems[(col+2)*B->step];
  const DTYPE *b3 = &B->elems[(col+3)*B->step];
  for (i = 0; i < nI; i++) {
    d0 = &dC->elems[i+(col+0)*dC->step];
    d1 = &dC->elems[i+(col+1)*dC->step];
    d2 = &dC->elems[i+(col+2)*dC->step];
    d3 = &dC->elems[i+(col+3)*dC->step];
    c0 = &C->elems[i+(col+0)*C->step];
    c1 = &C->elems[i+(col+1)*C->step];
    c2 = &C->elems[i+(col+2)*C->step];
    c3 = &C->elems[i+(col+3)*C->step];
    a0 = &A->elems[(i+0)*A->step];
    __mult1c2_ext(c0, c1, d0, d1, a0, b0, b1, alpha, nP);
    __mult1c2_ext(c2, c3, d2, d3, a0, b2, b3, alpha, nP);
  }
}


//  update two columns of C
static inline
void __CMULT2EXT(armas_x_dense_t *C, armas_x_dense_t *dC, const armas_x_dense_t *A,
                 const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *d0, *d1, *a0;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  const DTYPE *b1 = &B->elems[(col+1)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->elems[i+(col+0)*C->step];
    c1 = &C->elems[i+(col+1)*C->step];
    d0 = &dC->elems[i+(col+0)*dC->step];
    d1 = &dC->elems[i+(col+1)*dC->step];
    a0 = &A->elems[(i+0)*A->step];
    __mult1c2_ext(c0, c1, d0, d1, a0, b0, b1, alpha, nP);
  }
}

// update one column of C;
static inline
void __CMULT1EXT(armas_x_dense_t *C, armas_x_dense_t *dC, const armas_x_dense_t *A,
                 const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *d0;
  const DTYPE *a0;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->elems[i+(col+0)*C->step];
    d0 = &dC->elems[i+(col+0)*dC->step];
    a0 = &A->elems[i*A->step];
    __mult1c1_ext(c0, d0, a0, b0, alpha, nP);
  }
}

static inline
void armas_x_merge2_unsafe(armas_x_dense_t *C, const armas_x_dense_t *C0, const armas_x_dense_t *dC)
{
    int i, j;
  for (j = 0; j < C->cols; j++) {
    for (i = 0; i < C->rows-1; i += 2) {
      C->elems[(i+0)+j*C->step] = C0->elems[(i+0)+j*C0->step] + dC->elems[(i+0)+j*dC->step];
      C->elems[(i+1)+j*C->step] = C0->elems[(i+1)+j*C0->step] + dC->elems[(i+1)+j*dC->step];
    }
    if (i != C->rows) {
      C->elems[(i+0)+j*C->step] = C0->elems[(i+0)+j*C0->step] + dC->elems[(i+0)+j*dC->step];
    }
  }
}

static inline
void armas_x_merge_unsafe(armas_x_dense_t *A, const armas_x_dense_t *B)
{
  int i, j;
  for (j = 0; j < A->cols; j++) {
    for (i = 0; i < A->rows; i++) {
      A->elems[i+j*A->step] += B->elems[i+j*B->step];
    }
  }
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
