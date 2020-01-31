
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _MULT_EXT_AVX_F32_H
#define _MULT_EXT_AVX_F32_H 1

#include "eft.h"

#ifndef __HB32
#define __HB32 (1 << 31)

static uint32_t __masks_ps[8][8] __attribute__((aligned(64))) = {
  {__HB32, __HB32, __HB32, __HB32, __HB32, __HB32, __HB32, __HB32},
  {     0, __HB32, __HB32, __HB32, __HB32, __HB32, __HB32, __HB32},
  {     0,      0, __HB32, __HB32, __HB32, __HB32, __HB32, __HB32},
  {     0,      0,      0, __HB32, __HB32, __HB32, __HB32, __HB32},
  {     0,      0,      0,      0, __HB32, __HB32, __HB32, __HB32},
  {     0,      0,      0,      0,      0, __HB32, __HB32, __HB32},
  {     0,      0,      0,      0,      0,      0, __HB32, __HB32},
  {     0,      0,      0,      0,      0,      0,      0, __HB32}};

#endif

// error free horizontal summation of vector 8 singles
static
void hsum_m256(float *sum, float *err, __m256 S, __m256 C)
{
  __m256 R, Q, P; //, S0; //, A1, B1;
  float res, c0;

  c0 = hsum_f32x8(C);

  P = hflip_f32x8(S);        // half flip, switch low and high 128bit parts
  twosum_f32x8(&S, &Q, S, P);

  // add up residual terms in low 128bit of Q
  R = pflip_f32x8(Q);   
  Q += R;
  R = qflip_f32x8(Q);    
  Q += R;
  c0 += Q[0];

  P = pflip_f32x8(S);
  twosum_f32x8(&S, &Q, S, P);
  // 2 low floats have summation this far

  R = qflip_f32x8(Q);
  Q += R;
  c0 += Q[0];
  
  P = qflip_f32x8(S);
  twosum_f32x8(&S, &Q, S, P);

  twosum(sum, &res, *sum, S[0]);
  *err += c0 + Q[0] + res;
}


// update single element of C; (mult1x1x4)
static
void __mult1c1_ext(DTYPE *c, DTYPE *d, const DTYPE *a, const DTYPE *b, DTYPE alpha, int nR)
{
  register int k;
  __m256 P, Q, S, R;
  register __m256 C, A, B, Z, M;

  // sum registers
  S = _mm256_set1_ps(0.0);
  C = _mm256_set1_ps(0.0);
  
  for (k = 0; k < nR-7; k += 8) {
    A  = _mm256_load_ps(&a[k]);
    B  = _mm256_load_ps(&b[k]);
    
    twoprod_f32x8(&P, &Q, A, B); // 17 flops / 2 flops with FMA
    twosum_f32x8(&S, &R, P, S);  // 6 flops
    C += (R + Q);
  }
  if (k == nR)
    goto update;
  
  Z = _mm256_set1_ps(0.0);
  A = _mm256_load_ps(&a[k]);
  B = _mm256_load_ps(&b[k]);
  M  = _mm256_load_ps((float *)&__masks_ps[nR-k][0]);
  A = _mm256_blendv_ps(A, Z, M);
  B = _mm256_blendv_ps(B, Z, M);
#if 0
  switch(nR-k) {
  case 7:
    A = _mm256_blend_ps(A, Z, 0x80);
    B = _mm256_blend_ps(B, Z, 0x80);
    break;
  case 6:
    A = _mm256_blend_ps(A, Z, 0xC0);
    B = _mm256_blend_ps(B, Z, 0xC0);
    break;
  case 5:
    A = _mm256_blend_ps(A, Z, 0xE0);
    B = _mm256_blend_ps(B, Z, 0xE0);
    break;
  case 4:
    A = _mm256_blend_ps(A, Z, 0xF0);
    B = _mm256_blend_ps(B, Z, 0xF0);
    break;
  case 3:
    A = _mm256_blend_ps(A, Z, 0xF8);
    B = _mm256_blend_ps(B, Z, 0xF8);
    break;
  case 2:
    A = _mm256_blend_ps(A, Z, 0xFC);
    B = _mm256_blend_ps(B, Z, 0xFC);
    break;
  case 1:
    A = _mm256_blend_ps(A, Z, 0xFE);
    B = _mm256_blend_ps(B, Z, 0xFE);
    break;
  }
#endif
  twoprod_f32x8(&P, &Q, A, B);
  twosum_f32x8(&S, &R, P, S);
  C += (R + Q);
  
 update:
  A = _mm256_set1_ps(alpha);
  twoprod_f32x8(&P, &Q, A, S);
  // here P + Q = alpha*sum a[:]*b[:] 
  C *= A;
  C += Q;
  // sum horizontally
  hsum_m256(c, d, P, C);
}


// update 1x2 block of C (mult2x1x4)
static inline
void __mult1c2_ext(DTYPE *c0, DTYPE *c1, DTYPE *d0, DTYPE *d1, 
                   const DTYPE *a,
                   const DTYPE *b0, const DTYPE *b1, DTYPE alpha, int nR)
{
  register int k;
  register __m256 A, B0, B1, C0, C1, Z, M;
  __m256 P0, Q0, S0, R0, P1, Q1, S1, R1;

  // sum registers
  S0 = _mm256_set1_ps(0.0);
  S1 = _mm256_set1_ps(0.0);
  C0 = _mm256_set1_ps(0.0);
  C1 = _mm256_set1_ps(0.0);
  
  for (k = 0; k < nR-7; k += 8) {
    A   = _mm256_load_ps(&a[k]);
    B0  = _mm256_load_ps(&b0[k]);
    B1  = _mm256_load_ps(&b1[k]);

    twoprod_f32x8(&P0, &Q0, A, B0);
    twosum_f32x8(&S0, &R0, P0, S0);
    //R0 += Q0;
    C0 += (R0 + Q0);

    twoprod_f32x8(&P1, &Q1, A, B1);
    twosum_f32x8(&S1, &R1, P1, S1);
    //R1 += Q1;
    C1 += (R1 + Q1);
  }
  if (k == nR)
    goto update;
  
  Z  = _mm256_set1_ps(0.0);
  A  = _mm256_load_ps(&a[k]);
  B0 = _mm256_load_ps(&b0[k]);
  B1 = _mm256_load_ps(&b1[k]);
  M  = _mm256_load_ps((float *)&__masks_ps[nR-k][0]);
  A = _mm256_blendv_ps(A, Z, M);
  B0 = _mm256_blendv_ps(B0, Z, M);
  B1 = _mm256_blendv_ps(B1, Z, M);
#if 0
  switch(nR-k) {
  case 7:
    A  = _mm256_blend_ps(A, Z, 0x80);
    B0 = _mm256_blend_ps(B0, Z, 0x80);
    B1 = _mm256_blend_ps(B1, Z, 0x80);
    break;
  case 6:
    A  = _mm256_blend_ps(A, Z, 0xC0);
    B0 = _mm256_blend_ps(B0, Z, 0xC0);
    B1 = _mm256_blend_ps(B1, Z, 0xC0);
    break;
  case 5:
    A  = _mm256_blend_ps(A, Z, 0xE0);
    B0 = _mm256_blend_ps(B0, Z, 0xE0);
    B1 = _mm256_blend_ps(B1, Z, 0xE0);
    break;
  case 4:
    A  = _mm256_blend_ps(A, Z, 0xF0);
    B0 = _mm256_blend_ps(B0, Z, 0xF0);
    B1 = _mm256_blend_ps(B1, Z, 0xF0);
    break;
  case 3:
    A  = _mm256_blend_ps(A, Z, 0xF8);
    B0 = _mm256_blend_ps(B0, Z, 0xF8);
    B1 = _mm256_blend_ps(B1, Z, 0xF8);
    break;
  case 2:
    A  = _mm256_blend_ps(A, Z, 0xFC);
    B0 = _mm256_blend_ps(B0, Z, 0xFC);
    B1 = _mm256_blend_ps(B1, Z, 0xFC);
    break;
  case 1:
    A  = _mm256_blend_ps(A, Z, 0xFE);
    B0 = _mm256_blend_ps(B0, Z, 0xFE);
    B1 = _mm256_blend_ps(B1, Z, 0xFE);
    break;
  }
#endif
  twoprod_f32x8(&P0, &Q0, A, B0);
  twosum_f32x8(&S0, &R0, P0, S0);
  //R0 += Q0; 
  C0 += (R0 + Q0); 

  twoprod_f32x8(&P1, &Q1, A, B1);
  twosum_f32x8(&S1, &R1, P1, S1);
  //R1 += Q1;
  C1 += (R1 + Q1);
  
 update:
  A = _mm256_set1_ps(alpha);
  twoprod_f32x8(&P0, &Q0, A, S0);
  // here P + Q = alpha*sum a[:]*b[:] 
  C0 *= A;
  C0 += Q0;
  // sum horizontally
  hsum_m256(c0, d0, P0, C0);

  twoprod_f32x8(&P1, &Q1, A, S1);
  // here P + Q = alpha*sum a[:]*b[:] 
  C1 *= A;
  C1 += Q1;
  // sum horizontally
  hsum_m256(c1, d1, P1, C1);
}


#endif  // _MULT_EXT_AVX_F64_H

// Local Variables:
// indent-tabs-mode: nil
// End:

