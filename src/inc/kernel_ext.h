
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _ARMAS_KERNEL_EXT_H
#define _ARMAS_KERNEL_EXT_H 1

#include "eft.h"

// matrix-matrix multiplication primitives

static
void __blk_scale_ext(mdata_t *C0, mdata_t *dC, const mdata_t *A, DTYPE beta, int nR, int nC)
{
  int i, j;
  if (beta == __ZERO) {
    for (j = 0; j < nC; j++) {
      for (i = 0; i < nR; i++) {
          C0->md[i+j*C0->step] = __ZERO;
          dC->md[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  if (beta == __ONE) {
    for (j = 0; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        C0->md[i+j*C0->step] = A->md[i+j*A->step];
        dC->md[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nR-1; i += 2) {
      twoprod(&C0->md[(i+0)+j*C0->step],
              &dC->md[(i+0)+j*dC->step], beta, A->md[(i+0)+j*A->step]);
      twoprod(&C0->md[(i+1)+j*C0->step],
              &dC->md[(i+1)+j*dC->step], beta, A->md[(i+1)+j*A->step]);
    }
    if (i != nR) {
      twoprod(&C0->md[(i+0)+j*C0->step],
              &dC->md[(i+0)+j*dC->step], beta, A->md[(i+0)+j*A->step]);
    }
  }
}

static
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

static
void clear_blk(mdata_t *A, int nR, int nC)
{
  int j;
  for (j = 0; j < nC; j++) {
    memset(&A->md[j*A->step], 0, nR*sizeof(DTYPE));
  }
}

static
void ext_merge(mdata_t *A, mdata_t *B, int nR, int nC)
{
  int i, j;
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      A->md[i+j*A->step] += B->md[i+j*B->step];
    }
  }
}


// update 4 columns of C;
static inline
void __CMULT4EXT(mdata_t *C, mdata_t *dC, const mdata_t *A,
                 const mdata_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *c2, *c3, *d0, *d1, *d2, *d3;
  const DTYPE *a0, *a1;
  const DTYPE *b0 = &B->md[(col+0)*B->step];
  const DTYPE *b1 = &B->md[(col+1)*B->step];
  const DTYPE *b2 = &B->md[(col+2)*B->step];
  const DTYPE *b3 = &B->md[(col+3)*B->step];
  for (i = 0; i < nI; i++) {
    d0 = &dC->md[i+(col+0)*dC->step];
    d1 = &dC->md[i+(col+1)*dC->step];
    d2 = &dC->md[i+(col+2)*dC->step];
    d3 = &dC->md[i+(col+3)*dC->step];
    c0 = &C->md[i+(col+0)*C->step];
    c1 = &C->md[i+(col+1)*C->step];
    c2 = &C->md[i+(col+2)*C->step];
    c3 = &C->md[i+(col+3)*C->step];
    a0 = &A->md[(i+0)*A->step];      
    __mult1c2_ext(c0, c1, d0, d1, a0, b0, b1, alpha, nP);
    __mult1c2_ext(c2, c3, d2, d3, a0, b2, b3, alpha, nP);
  }
}


//  update two columns of C
static inline
void __CMULT2EXT(mdata_t *C, mdata_t *dC, const mdata_t *A,
                 const mdata_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *d0, *d1, *a0, *a1;
  const DTYPE *b0 = &B->md[(col+0)*B->step];
  const DTYPE *b1 = &B->md[(col+1)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->md[i+(col+0)*C->step];
    c1 = &C->md[i+(col+1)*C->step];
    d0 = &dC->md[i+(col+0)*dC->step];
    d1 = &dC->md[i+(col+1)*dC->step];
    a0 = &A->md[(i+0)*A->step];      
    a1 = &A->md[(i+1)*A->step];      
    __mult1c2_ext(c0, c1, d0, d1, a0, b0, b1, alpha, nP);
  }
}

// update one column of C;
static inline
void __CMULT1EXT(mdata_t *C, mdata_t *dC, const mdata_t *A,
                 const mdata_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *d0;
  const DTYPE *a0;
  const DTYPE *b0 = &B->md[(col+0)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->md[i+(col+0)*C->step];
    d0 = &dC->md[i+(col+0)*dC->step];
    a0 = &A->md[i*A->step];
    __mult1c1_ext(c0, d0, a0, b0, alpha, nP);
  }
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
