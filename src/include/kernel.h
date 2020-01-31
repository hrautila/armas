
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef _ARMAS_KERNEL_H
#define _ARMAS_KERNEL_H 1

#if defined(COMPLEX128)
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */
  #if defined(__AVX__)
    #include "x86_64/mult_avx_c128.h"
  #elif defined(__SSE__)
    #include "x86_64/mult_sse_c128.h"
  #else
    #include "nosimd/mult_nosimd.h"
  #endif

#elif defined(COMPLEX64)
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */

  #if defined(__AVX__)
    #include "x86_64/mult_avx_c64.h"
  #elif defined(__SSE__)
    #include "x86_64/mult_sse_c64.h"
  #else
    #include "nosimd/mult_nosimd.h"
  #endif

#elif defined(FLOAT32)
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
  #if defined(__x86_64__)
    #if defined(__AVX__)
      #include "x86_64/mult_avx_f32.h"
    #elif defined(__SSE__)
      #include "x86_64/mult_sse_f32.h"
    #else
      #include "nosimd/mult_nosimd.h"
    #endif
  #elif defined(__arm__)
    #if defined(__ARM_NEON)
      #if defined(__ARM_FEATURE_FMA)
        #include "arm/mult_armneon_fma_f32.h"
      #else
        #include "arm/mult_armneon_f32.h"
      #endif
    #else
      #include "nosimd/mult_nosimd.h"
    #endif
  #else
      #include "nosimd/mult_nosimd.h"
  #endif

#else
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
  #if defined(__FMA__)
    #include "x86_64/mult_fma_f64.h"
  #elif defined(__AVX__)
    #include "x86_64/mult_avx_f64.h"
  #elif defined(__SSE__)
    #include "x86_64/mult_sse_f64.h"
  #else
    #include "nosimd/mult_nosimd.h"
  #endif

#endif


// matrix-matrix multiplication primitives
#if HAVE_6COL == 1
// update 6 columns of C;
static inline
void __CMULT6(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *c2, *c3, *c4, *c5;
  const DTYPE *a0, *a1;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  const DTYPE *b1 = &B->elems[(col+1)*B->step];
  const DTYPE *b2 = &B->elems[(col+2)*B->step];
  const DTYPE *b3 = &B->elems[(col+3)*B->step];
  const DTYPE *b4 = &B->elems[(col+4)*B->step];
  const DTYPE *b5 = &B->elems[(col+5)*B->step];
  for (i = 0; i < nI-1; i += 2) {
    c0 = &C->elems[i+(col+0)*C->step];
    c1 = &C->elems[i+(col+1)*C->step];
    c2 = &C->elems[i+(col+2)*C->step];
    c3 = &C->elems[i+(col+3)*C->step];
    c4 = &C->elems[i+(col+4)*C->step];
    c5 = &C->elems[i+(col+5)*C->step];
    a0 = &A->elems[(i+0)*A->step];
    a1 = &A->elems[(i+1)*A->step];
    __mult2c6(c0, c1, c2, c3, c4, c5, a0, a1, b0, b1, b2, b3, b4, b5, alpha, nP);
  }
  if (i == nI)
    return;
  c0 = &C->elems[i+(col+0)*C->step];
  c1 = &C->elems[i+(col+1)*C->step];
  c2 = &C->elems[i+(col+2)*C->step];
  c3 = &C->elems[i+(col+3)*C->step];
  c4 = &C->elems[i+(col+4)*C->step];
  c5 = &C->elems[i+(col+5)*C->step];
  a0 = &A->elems[(i+0)*A->step];
  __mult1c6(c0, c1, c2, c3, c4, c5, a0, b0, b1, b2, b3, b4, b5, alpha, nP);
}
#endif

// update 4 columns of C;
static inline
void __CMULT4(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *c2, *c3;
  const DTYPE *a0, *a1;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  const DTYPE *b1 = &B->elems[(col+1)*B->step];
  const DTYPE *b2 = &B->elems[(col+2)*B->step];
  const DTYPE *b3 = &B->elems[(col+3)*B->step];
  for (i = 0; i < nI-1; i += 2) {
    c0 = &C->elems[i+(col+0)*C->step];
    c1 = &C->elems[i+(col+1)*C->step];
    c2 = &C->elems[i+(col+2)*C->step];
    c3 = &C->elems[i+(col+3)*C->step];
    a0 = &A->elems[(i+0)*A->step];
    a1 = &A->elems[(i+1)*A->step];
    __mult2c4(c0, c1, c2, c3, a0, a1, b0, b1, b2, b3, alpha, nP);
  }
  if (i == nI)
    return;
  c0 = &C->elems[i+(col+0)*C->step];
  c1 = &C->elems[i+(col+1)*C->step];
  c2 = &C->elems[i+(col+2)*C->step];
  c3 = &C->elems[i+(col+3)*C->step];
  a0 = &A->elems[(i+0)*A->step];
  __mult1c4(c0, c1, c2, c3, a0, b0, b1, b2, b3, alpha, nP);
}


//  update two columns of C
static inline
void __CMULT2(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0, *c1, *a0, *a1;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  const DTYPE *b1 = &B->elems[(col+1)*B->step];
  for (i = 0; i < nI-1; i += 2) {
    c0 = &C->elems[i+(col+0)*C->step];
    c1 = &C->elems[i+(col+1)*C->step];
    a0 = &A->elems[(i+0)*A->step];
    a1 = &A->elems[(i+1)*A->step];
    __mult2c2(c0, c1, a0, a1, b0, b1, alpha, nP);
  }
  if (i == nI)
    return;
  c0 = &C->elems[i+(col+0)*C->step];
  c1 = &C->elems[i+(col+1)*C->step];
  a0 = &A->elems[(i+0)*A->step];
  __mult1c2(c0, c1, a0, b0, b1, alpha, nP);
}

// update one column of C;
static inline
void __CMULT1(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int col, int nI, int nP)
{
  register int i;
  DTYPE *c0;
  const DTYPE *a0;
  const DTYPE *b0 = &B->elems[(col+0)*B->step];
  for (i = 0; i < nI; i++) {
    c0 = &C->elems[i+(col+0)*C->step];
    a0 = &A->elems[i*A->step];
    __mult1c1(c0, a0, b0, alpha, nP);
  }
}


// update 4 rows of C;
static inline
void __RMULT4(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int row, int nJ, int nP)
{
  register int k;
  DTYPE *c0, *c1;
  const DTYPE *b0, *b1;
  const DTYPE *a0 = &A->elems[(row+0)*A->step];
  const DTYPE *a1 = &A->elems[(row+1)*A->step];
  const DTYPE *a2 = &A->elems[(row+2)*A->step];
  const DTYPE *a3 = &A->elems[(row+3)*A->step];
  for (k = 0; k < nJ-1; k += 2) {
    c0 = &C->elems[row+(k+0)*C->step];
    c1 = &C->elems[row+(k+1)*C->step];
    b0 = &B->elems[(k+0)*B->step];
    b1 = &B->elems[(k+1)*B->step];
    __mult4c2(c0, c1, a0, a1, a2, a3, b0, b1, alpha, nP);
  }
  if (k == nJ)
    return;
  c0 = &C->elems[row+(k+0)*C->step];
  b0 = &B->elems[(k+0)*B->step];
  __mult4c1(c0, a0, a1, a2, a3, b0, alpha, nP);
}

// update 2 rows of C;
static inline
void __RMULT2(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int row, int nJ, int nP)
{
  register int k;
  DTYPE *c0, *c1;
  const DTYPE *b0, *b1;
  const DTYPE *a0 = &A->elems[(row+0)*A->step];
  const DTYPE *a1 = &A->elems[(row+1)*A->step];
  for (k = 0; k < nJ-1; k += 2) {
    c0 = &C->elems[row+(k+0)*C->step];
    c1 = &C->elems[row+(k+1)*C->step];
    b0 = &B->elems[(k+0)*B->step];
    b1 = &B->elems[(k+1)*B->step];
    __mult2c2(c0, c1, a0, a1, b0, b1, alpha, nP);
  }
  if (k == nJ)
    return;
  c0 = &C->elems[row+(k+0)*C->step];
  b0 = &B->elems[(k+0)*B->step];
  __mult2c1(c0, a0, a1, b0, alpha, nP);
}

// update 1row of C;
static inline
void __RMULT1(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *B, DTYPE alpha, int row, int nJ, int nP)
{
  register int k;
  DTYPE *c0;
  const DTYPE *b0;
  const DTYPE *a0 = &A->elems[(row+0)*A->step];
  for (k = 0; k < nJ; k += 1) {
    c0 = &C->elems[row+(k+0)*C->step];
    b0 = &B->elems[(k+0)*B->step];
    __mult1c1(c0, a0, b0, alpha, nP);
  }
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
