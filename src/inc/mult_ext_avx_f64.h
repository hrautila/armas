
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _MULT_EXT_AVX_F64_H
#define _MULT_EXT_AVX_F64_H 1

#include "eft.h"

// error free horizontal summation of vector 4 doubles
static
void hsum_m256d(double *sum, double *err, __m256d S, __m256d C)
{
  __m256d R, Q, P; //, S0; //, A1, B1;
  double res;

  Q = _mm256_permute2f128_pd(C, C, 1);    // Q = high(C), low(C)
  C = _mm256_add_pd(Q, C);                // low(C) == low(C) + high(C)
  Q = _mm256_permute_pd(C, 1);
  C = _mm256_add_pd(Q, C);                // C[0] == sum C

  P = _mm256_permute2f128_pd(S, S, 1);    // P = high(S),low(S)
  twosum_f64x4(&S, &Q, S, P);
  R = _mm256_permute_pd(Q, 1);
  Q = _mm256_add_pd(Q, R);                // Q[0] = R[0] + R[1]

  P = _mm256_permute_pd(S, 1);
  twosum_f64x4(&S, &R, S, P);
  
  twosum(sum, &res, *sum, S[0]);
  *err += C[0] + Q[0] + R[0] + res;
}


// update single element of C; (mult1x1x4)
static
void __mult1c1_ext(DTYPE *c, DTYPE *d, const DTYPE *a, const DTYPE *b, DTYPE alpha, int nR)
{
  register int k;
  __m256d P, Q, S, R;
  register __m256d C, A, B, Z;

  // sum registers
  S = _mm256_set1_pd(0.0);
  C = _mm256_set1_pd(0.0);
  
  for (k = 0; k < nR-3; k += 4) {
    A  = _mm256_loadu_pd(&a[k]);
    B  = _mm256_loadu_pd(&b[k]);
    
    twoprod_f64x4(&P, &Q, A, B); // 17 flops / 2 flops with FMA
    twosum_f64x4(&S, &R, P, S);  // 6 flops
    C += (R + Q);                // 2 flops
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
  twoprod_f64x4(&P, &Q, A, B);
  twosum_f64x4(&S, &R, P, S);
  C += (R + Q);
  
 update:
  A = _mm256_set1_pd(alpha);
  twoprod_f64x4(&P, &Q, A, S);
  // here P + Q = alpha*sum a[:]*b[:] 
  C *= A;
  C += Q;
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
  register __m256d A, B0, B1, C0, C1, Z;
  __m256d P0, Q0, S0, R0, P1, Q1, S1, R1;

  // sum registers
  S0 = _mm256_set1_pd(0.0);
  S1 = _mm256_set1_pd(0.0);
  C0 = _mm256_set1_pd(0.0);
  C1 = _mm256_set1_pd(0.0);
  
  for (k = 0; k < nR-3; k += 4) {
    A   = _mm256_loadu_pd(&a[k]);
    B0  = _mm256_loadu_pd(&b0[k]);
    B1  = _mm256_loadu_pd(&b1[k]);

    twoprod_f64x4(&P0, &Q0, A, B0);
    twosum_f64x4(&S0, &R0, P0, S0);
    C0 += (R0 + Q0);

    twoprod_f64x4(&P1, &Q1, A, B1);
    twosum_f64x4(&S1, &R1, P1, S1);
    C1 += (R1 + Q1);
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
  twoprod_f64x4(&P0, &Q0, A, B0);
  twosum_f64x4(&S0, &R0, P0, S0);
  C0 += (R0 + Q0); 

  twoprod_f64x4(&P1, &Q1, A, B1);
  twosum_f64x4(&S1, &R1, P1, S1);
  C1 += (R1 + Q1);
  
 update:
  A = _mm256_set1_pd(alpha);
  twoprod_f64x4(&P0, &Q0, A, S0);
  // here P + Q = alpha*sum a[:]*b[:] 
  C0 *= A;
  C0 += Q0;
  // sum horizontally
  hsum_m256d(c0, d0, P0, C0);

  twoprod_f64x4(&P1, &Q1, A, S1);
  // here P + Q = alpha*sum a[:]*b[:] 
  C1 *= A;
  C1 += Q1;
  // sum horizontally
  hsum_m256d(c1, d1, P1, C1);
}


#endif  // _MULT_EXT_AVX_F64_H

// Local Variables:
// indent-tabs-mode: nil
// End:

