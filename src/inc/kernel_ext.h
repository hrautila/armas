
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _ARMAS_KERNEL_EXT_H
#define _ARMAS_KERNEL_EXT_H 1

#include "eft.h"


#if 0

#if ! defined(FLOAT64)
#include "mult_ext_nosimd.h"
#else
#if defined(__AVX__)
#include "mult_ext_avx_f64.h"
#else
#include "mult_ext_nosimd.h"
#endif  // defined(__AVX__)
#endif  // defined(FLOAT64)

#endif  // 0

// matrix-matrix multiplication primitives


// update 4 columns of C;
static inline
void __CMULT4EXT(mdata_t *C, mdata_t *dC, const mdata_t *A,
                 const mdata_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *c2, *c3, *d0, *d1, *d2, *d3;
  const DTYPE *a0;
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
  DTYPE *c0, *c1, *d0, *d1, *a0;
  const DTYPE *b0 = &B->md[(col+0)*B->step];
  const DTYPE *b1 = &B->md[(col+1)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->md[i+(col+0)*C->step];
    c1 = &C->md[i+(col+1)*C->step];
    d0 = &dC->md[i+(col+0)*dC->step];
    d1 = &dC->md[i+(col+1)*dC->step];
    a0 = &A->md[(i+0)*A->step];      
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
