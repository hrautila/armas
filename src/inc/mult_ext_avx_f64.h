
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _MULT_EXT_AVX_F64_H
#define _MULT_EXT_AVX_F64_H 1

#include "eft.h"

// macro version of error free summation
#define TWOSUM(x, y, a, b, z, w)                \
  do {						\
    x = _mm256_add_pd(a, b);			\
    z = _mm256_sub_pd(x, a);			\
    w = _mm256_sub_pd(x, z);			\
    z = _mm256_sub_pd(b, z);			\
    w = _mm256_sub_pd(a, w);			\
    y = _mm256_add_pd(w, z);			\
  } while (0)

// macro version of error free product (without FMA)
#define TWOPROD(p, q, a, b, a1, a2, b1, b2, z, F)	\
  do {							\
    z  = _mm256_mul_pd(F, a);				\
    a1 = _mm256_sub_pd(z, a);				\
    a1 = _mm256_sub_pd(z, a1);				\
    a2 = _mm256_sub_pd(a, a1);				\
    z  = _mm256_mul_pd(F, b);				\
    b1 = _mm256_sub_pd(z, b);				\
    b1 = _mm256_sub_pd(z, b1);				\
    b2 = _mm256_sub_pd(b, b1);				\
    p  = _mm256_mul_pd(a, b);				\
    z  = _mm256_mul_pd(a1, b1);				\
    q  = _mm256_sub_pd(p, z);				\
    z  = _mm256_mul_pd(a2, b1);				\
    q  = _mm256_sub_pd(q, z);				\
    z  = _mm256_mul_pd(a1, b2);				\
    q  = _mm256_sub_pd(q, z);				\
    z  = _mm256_mul_pd(a2, b2);				\
    q  = _mm256_sub_pd(z, q);				\
  } while (0)

// error free horizontal summation of vector 4 doubles (TODO: fix this, error free summation
static
void hsum_m256d(double *sum, double *err, __m256d S, __m256d C)
{
  register __m256d A1, B1, R, Q, P, S0;

  Q = _mm256_permute2f128_pd(C, C, 1);    // Q = high(C), low(C)
  C = _mm256_add_pd(Q, C);                // low(C) == low(C) + high(C)
  Q = _mm256_permute_pd(C, 1);
  C = _mm256_add_pd(Q, C);                // C[0] == sum C

  P = _mm256_permute2f128_pd(S, S, 1);    // P = high(S),low(S)
  S0 = S;                                 // cheating optimizer; extra assignment 
  TWOSUM(S, Q, S0, P, /**/ A1, B1);       // low(S + Q) = sum of low(S), high(S)
  R = _mm256_permute_pd(Q, 1);
  Q = _mm256_add_pd(Q, R);                // Q[0] = R[0] + R[1]

  P = _mm256_permute_pd(S, 1);
  S0 = S;                                 // cheating optimizer; extra assignment 
  TWOSUM(S, R, S0, P, /**/ A1, B1);       // S[0] = sum S
  
  *err += C[0] + Q[0] + R[0];
  *sum += S[0];
}


// update single element of C; (mult1x1x4)
static
void __mult1c1_ext(DTYPE *c, DTYPE *d, const DTYPE *a, const DTYPE *b, DTYPE alpha, int nR)
{
  register int k;
  register __m256d A, A0, A1, B, B0, B1;
  register __m256d S, C, P, Q, R, Z, F, S0;

  F = _mm256_set1_pd(__FACTOR);
  // sum registers
  S = _mm256_set1_pd(0.0);
  C = _mm256_set1_pd(0.0);
  
  for (k = 0; k < nR-3; k += 4) {
    A  = _mm256_loadu_pd(&a[k]);
    B  = _mm256_loadu_pd(&b[k]);
    // 17 flops / 2 flops with FMA
    TWOPROD(P, Q, A, B, /**/ A0, A1, B0, B1, Z, F);
    S0 = S;
    // 6 flops
    TWOSUM(S, R, P, S0, /**/ A1, B1);
    // 2 flops
    R = _mm256_add_pd(Q, R); 
    C = _mm256_add_pd(C, R); 
  }
  if (k == nR)
    goto update;
  
  Z = _mm256_set1_pd(0.0);
  A = _mm256_loadu_pd(&a[k]);
  B = _mm256_loadu_pd(&b[k]);
  switch(nR-k) {
  case 3:
    A = _mm256_blend_pd(A, Z, 0x8);
    B = _mm256_blend_pd(B, Z, 0x8);
    break;
  case 2:
    A = _mm256_blend_pd(A, Z, 0xC);
    B = _mm256_blend_pd(B, Z, 0xC);
    break;
  case 1:
    A = _mm256_blend_pd(A, Z, 0xE);
    B = _mm256_blend_pd(B, Z, 0xE);
    break;
  }
  TWOPROD(P, Q, A, B, /**/ A0, A1, B0, B1, Z, F);
  S0 = S;
  TWOSUM(S, R, P, S0, /**/ A1, B1);
  // 
  R = _mm256_add_pd(Q, R); 
  C = _mm256_add_pd(C, R); 
  
 update:
  A = _mm256_set1_pd(alpha);
  TWOPROD(P, Q, A, S, /**/ A0, A1, B0, B1, Z, F);
  // here P + Q = alpha*sum a[:]*b[:] 
  C = _mm256_mul_pd(A, C);
  C = _mm256_add_pd(C, Q);
  // sum horizontally
  hsum_m256d(c, d, P, C);
}


// update 1x2 block of C (mult2x1x4)
static inline
void __mult1c2_ext(DTYPE *c0, DTYPE *c1, DTYPE *d0, DTYPE *d1, 
                   const DTYPE *a,
                   const DTYPE *b0, const DTYPE *b1, DTYPE alpha, int nR)
{
  register int k;
  register __m256d A, A0, A1, B0, B1, Bx, Bz;
  register __m256d S0, S1, C0, C1, P0, P1, Q0, Q1, R0, R1, Z, F, Sx;

  F = _mm256_set1_pd(__FACTOR);
  // sum registers
  S0 = _mm256_set1_pd(0.0);
  S1 = _mm256_set1_pd(0.0);
  C0 = _mm256_set1_pd(0.0);
  C1 = _mm256_set1_pd(0.0);
  
  for (k = 0; k < nR-3; k += 4) {
    A   = _mm256_loadu_pd(&a[k]);
    B0  = _mm256_loadu_pd(&b0[k]);
    B1  = _mm256_loadu_pd(&b1[k]);
    TWOPROD(P0, Q0, A, B0, /**/ A0, A1, Bx, Bz, Z, F);  // 17 flops / 2 flops with FMA
    Sx = S0;
    TWOSUM(S0, R0, P0, Sx, /**/ Bz, Bx);                // 6 flops
    R0 = _mm256_add_pd(Q0, R0);                         // 2 flops
    C0 = _mm256_add_pd(C0, R0); 
    TWOPROD(P1, Q1, A, B1, /**/ A0, A1, Bx, Bz, Z, F);
    Sx = S1;
    TWOSUM(S1, R1, P1, Sx, /**/ Bz, Bx);
    R1 = _mm256_add_pd(Q1, R1); 
    C1 = _mm256_add_pd(C1, R1); 
  }
  if (k == nR)
    goto update;
  
  Z  = _mm256_set1_pd(0.0);
  A  = _mm256_loadu_pd(&a[k]);
  B0 = _mm256_loadu_pd(&b0[k]);
  B1 = _mm256_loadu_pd(&b1[k]);
  switch(nR-k) {
  case 3:
    A  = _mm256_blend_pd(A, Z, 0x8);
    B0 = _mm256_blend_pd(B0, Z, 0x8);
    B1 = _mm256_blend_pd(B1, Z, 0x8);
    break;
  case 2:
    A  = _mm256_blend_pd(A, Z, 0xC);
    B0 = _mm256_blend_pd(B0, Z, 0xC);
    B1 = _mm256_blend_pd(B1, Z, 0xC);
    break;
  case 1:
    A  = _mm256_blend_pd(A, Z, 0xE);
    B0 = _mm256_blend_pd(B0, Z, 0xE);
    B1 = _mm256_blend_pd(B1, Z, 0xE);
    break;
  }
  TWOPROD(P0, Q0, A, B0, /**/ A0, A1, Bx, Bz, Z, F);
  Sx = S0;
  TWOSUM(S0, R0, P0, Sx, /**/ Bx, Bz);
  R0 = _mm256_add_pd(Q0, R0); 
  C0 = _mm256_add_pd(C0, R0); 

  TWOPROD(P1, Q1, A, B1, /**/ A0, A1, Bx, Bz, Z, F);
  Sx = S1;
  TWOSUM(S1, R1, P1, Sx, /**/ Bx, Bz);
  R1 = _mm256_add_pd(Q1, R1); 
  C1 = _mm256_add_pd(C1, R1); 
  
 update:
  A = _mm256_set1_pd(alpha);
  TWOPROD(P0, Q0, A, S0, /**/ A0, A1, Bx, Bz, Z, F);
  // here P + Q = alpha*sum a[:]*b[:] 
  C0 = _mm256_mul_pd(A, C0);
  C0 = _mm256_add_pd(C0, Q0);
  // sum horizontally
  hsum_m256d(c0, d0, P0, C0);

  TWOPROD(P1, Q1, A, S1, /**/ A0, A1, Bx, Bz, Z, F);
  // here P + Q = alpha*sum a[:]*b[:] 
  C1 = _mm256_mul_pd(A, C1);
  C1 = _mm256_add_pd(C1, Q1);
  // sum horizontally
  hsum_m256d(c1, d1, P1, C1);
}


#endif  // _MULT_EXT_AVX_F64_H

// Local Variables:
// indent-tabs-mode: nil
// End:

