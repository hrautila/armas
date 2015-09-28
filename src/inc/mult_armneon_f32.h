
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _MULT_ARMNEON_F32_H
#define _MULT_ARMNEON_F32_H 1

#include <limits.h>
#include <stdint.h>
#include <arm_neon.h>

#include "simd.h"
#include "debug.h"

const uint32_t __hb32 = (1 << 31);

#define __t32 0xffffff

static uint32_t __masks_f32[4][4] __attribute__((aligned(64))) = {
  {__t32, __t32, __t32, __t32},
  {__t32, __t32, __t32,     0},
  {__t32, __t32,     0,     0},
  {__t32,     0,     0,     0}};

// update 1x4 block of C; one row, four columns (mult4x1x1)
static inline
void __mult1c4(float *c0, float *c1, float *c2, float *c3,
               const float *a, const float *b0, const float *b1,
               const float *b2, const float *b3, float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, y2, y3, A0, Z;
  register float32x4_t B0, B1, B2, B3;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y2 = y3 = y0;

  for (k = 0; k < nR-3; k += 4) {
    A0 = load_f32x4(&a[k]);
    B0 = load_f32x4(&b0[k]);
    B1 = load_f32x4(&b1[k]);
    B2 = load_f32x4(&b2[k]);
    B3 = load_f32x4(&b3[k]);
    y0 = vmlaq_f32(y0, A0, B0);
    y1 = vmlaq_f32(y1, A0, B1);
    y2 = vmlaq_f32(y2, A0, B2); 
    y3 = vmlaq_f32(y3, A0, B3);
  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  A0 = load_f32x4(&a[k]);
  A0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A0, (uint32x4_t)Z);

  B0 = load_f32x4(&b0[k]);
  B1 = load_f32x4(&b1[k]);
  B2 = load_f32x4(&b2[k]);
  B3 = load_f32x4(&b3[k]);

  y0 = vmlaq_f32(y0, A0, B0);
  y1 = vmlaq_f32(y1, A0, B1);
  y2 = vmlaq_f32(y2, A0, B2);
  y3 = vmlaq_f32(y3, A0, B3);

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c1[0] += alpha*hsum_f32x4(y1);
  c2[0] += alpha*hsum_f32x4(y2);
  c3[0] += alpha*hsum_f32x4(y3);
}


// update 2x4 block of C; two rows, four columns (mult4x2x1)
static inline
void __mult2c4(float *c0, float *c1, float *c2, float *c3,
               const float *a0, const float *a1, const float *b0,
               const float *b1, const float *b2, const float *b3,
               float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, y2, y3, y4, y5, y6, y7, A0, A1, Z;
  register float32x4_t B0, B1, B2, B3;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y2 = y3 = y4 = y5 = y6 = y7 = y0;

  for (k = 0; k < nR-3; k += 4) {
    A0 = load_f32x4(&a0[k]);
    A1 = load_f32x4(&a1[k]);
    B0 = load_f32x4(&b0[k]);
    B1 = load_f32x4(&b1[k]);
    B2 = load_f32x4(&b2[k]);
    B3 = load_f32x4(&b3[k]);
    y0 = vmlaq_f32(y0, A0, B0);
    y1 = vmlaq_f32(y1, A0, B1);
    y2 = vmlaq_f32(y2, A0, B2);
    y3 = vmlaq_f32(y3, A0, B3);

    y4 = vmlaq_f32(y4, A1, B0);
    y5 = vmlaq_f32(y5, A1, B1);
    y6 = vmlaq_f32(y6, A1, B2);
    y7 = vmlaq_f32(y7, A1, B3); 

  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  A0 = load_f32x4(&a0[k]);
  A1 = load_f32x4(&a1[k]);
  A0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A0, (uint32x4_t)Z);
  A1 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A1, (uint32x4_t)Z);

  B0 = load_f32x4(&b0[k]);
  B1 = load_f32x4(&b1[k]);
  B2 = load_f32x4(&b2[k]);
  B3 = load_f32x4(&b3[k]);
  y0 = vmlaq_f32(y0, A0, B0);
  y1 = vmlaq_f32(y1, A0, B1);
  y2 = vmlaq_f32(y2, A0, B2);
  y3 = vmlaq_f32(y3, A0, B3);
  
  y4 = vmlaq_f32(y4, A1, B0);
  y5 = vmlaq_f32(y5, A1, B1);
  y6 = vmlaq_f32(y6, A1, B2);
  y7 = vmlaq_f32(y7, A1, B3); 

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c1[0] += alpha*hsum_f32x4(y1);
  c2[0] += alpha*hsum_f32x4(y2);
  c3[0] += alpha*hsum_f32x4(y3);
  c0[1] += alpha*hsum_f32x4(y4);
  c1[1] += alpha*hsum_f32x4(y5);
  c2[1] += alpha*hsum_f32x4(y6);
  c3[1] += alpha*hsum_f32x4(y7);
}


// update 1x2 block of C; one row, two columns (mult2x1x1)
static inline
void __mult1c2(float *c0, float *c1,
               const float *a, const float *b0, const float *b1,
               float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, A0, B0, B1, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    A0 = load_f32x4(&a[k]);
    B0 = load_f32x4(&b0[k]);
    B1 = load_f32x4(&b1[k]);
    y0 = vmlaq_f32(y0, A0, B0);
    y1 = vmlaq_f32(y1, A0, B1);
  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  A0 = load_f32x4(&a[k]);
  A0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A0, (uint32x4_t)Z);

  B0 = load_f32x4(&b0[k]);
  B1 = load_f32x4(&b1[k]);
  y0 = vmlaq_f32(y0, A0, B0);
  y1 = vmlaq_f32(y1, A0, B1);

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c1[0] += alpha*hsum_f32x4(y1);
}


// update 2x2 block of C; (mult2x2x1)
static inline
void __mult2c2(float *c0, float *c1,
               const float *a0, const float *a1,
               const float *b0, const float *b1,
               float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, y2, y3, A0, A1, B0, B1, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y0; 
  y2 = y0; 
  y3 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    A0 = load_f32x4(&a0[k]);
    A1 = load_f32x4(&a1[k]);
    B0 = load_f32x4(&b0[k]);
    B1 = load_f32x4(&b1[k]);
    y0 = vmlaq_f32(y0, A0, B0);
    y1 = vmlaq_f32(y1, A0, B1);
    y2 = vmlaq_f32(y2, A1, B0);
    y3 = vmlaq_f32(y3, A1, B1);  
  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  A0 = load_f32x4(&a0[k]);
  A1 = load_f32x4(&a1[k]);
  A0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A0, (uint32x4_t)Z);
  A1 = (float32x4_t)vbslq_u32(M, (uint32x4_t)A1, (uint32x4_t)Z);

  B0 = load_f32x4(&b0[k]);
  B1 = load_f32x4(&b1[k]);
  y0 = vmlaq_f32(y0, A0, B0);
  y1 = vmlaq_f32(y1, A0, B1);
  y2 = vmlaq_f32(y2, A1, B0);
  y3 = vmlaq_f32(y3, A1, B1);  

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c1[0] += alpha*hsum_f32x4(y1);
  c0[1] += alpha*hsum_f32x4(y2);
  c1[1] += alpha*hsum_f32x4(y3);
}

// update single element of C; with inner product of A row and B column
static inline
void __mult1c1(float *c, const float *a, const float *b, float alpha, int nR)
{
  register int k;
  register float32x4_t y0, A, B, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  for (k = 0; k < nR-3; k += 4) {
    A  = load_f32x4(&a[k]);
    B  = load_f32x4(&b[k]);
    y0 = vmlaq_f32(y0, A, B);
  }
  if (k == nR)
    goto update;

  Z = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  A = load_f32x4(&a[k]);
  A = (float32x4_t)vbslq_u32(M, (uint32x4_t)A, (uint32x4_t)Z);
  B  = load_f32x4(&b[k]);
  y0 = vmlaq_f32(y0, A, B);

update:
  c[0] += alpha*hsum_f32x4(y0);
}


// version for breadth-first update of C; update C row-wise
// 1 to 4 rows of A, 1 or 2 rows of B, update atmost 4x2 block of C 

// update 4x1 block of C; four rows, one column (dmult4x1x1)
static inline
void __mult4c1(float *c0, 
               const float *a0, const float *a1,
               const float *a2, const float *a3,
               const float *b0, float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, y2, y3, B0, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y2 = y3 = y0;

  for (k = 0; k < nR-3; k += 4) {
    B0 = load_f32x4(&b0[k]);
    y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
    y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));
    y2 = vmlaq_f32(y2, B0, load_f32x4(&a2[k]));
    y3 = vmlaq_f32(y3, B0, load_f32x4(&a3[k]));
  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  B0 = load_f32x4(&b0[k]);
  B0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)B0, (uint32x4_t)Z);

  y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
  y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));
  y2 = vmlaq_f32(y2, B0, load_f32x4(&a2[k]));
  y3 = vmlaq_f32(y3, B0, load_f32x4(&a3[k]));

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c0[1] += alpha*hsum_f32x4(y1);
  c0[2] += alpha*hsum_f32x4(y2);
  c0[3] += alpha*hsum_f32x4(y3);
}


// update 4x2 block of C; four rows, one column (dmult4x2x1)
static inline
void __mult4c2(float *c0, float *c1,
               const float *a0, const float *a1,
               const float *a2, const float *a3,
               const float *b0, const float *b1, float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, y2, y3, y4, y5, y6, y7, B0, B1, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y2 = y3 = y0;
  y4 = y5 = y6 = y7 = y0;

  for (k = 0; k < nR-3; k += 4) {
    B0 = load_f32x4(&b0[k]);
    B1 = load_f32x4(&b1[k]);
    y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
    y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));
    y2 = vmlaq_f32(y2, B0, load_f32x4(&a2[k]));
    y3 = vmlaq_f32(y3, B0, load_f32x4(&a3[k]));
    y4 = vmlaq_f32(y4, B1, load_f32x4(&a0[k]));
    y5 = vmlaq_f32(y5, B1, load_f32x4(&a1[k]));
    y6 = vmlaq_f32(y6, B1, load_f32x4(&a2[k]));
    y7 = vmlaq_f32(y7, B1, load_f32x4(&a3[k]));

  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  B0 = load_f32x4(&b0[k]);
  B1 = load_f32x4(&b1[k]);
  B0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)B0, (uint32x4_t)Z);
  B1 = (float32x4_t)vbslq_u32(M, (uint32x4_t)B1, (uint32x4_t)Z);

  y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
  y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));
  y2 = vmlaq_f32(y2, B0, load_f32x4(&a2[k]));
  y3 = vmlaq_f32(y3, B0, load_f32x4(&a3[k]));
  y4 = vmlaq_f32(y4, B1, load_f32x4(&a0[k]));
  y5 = vmlaq_f32(y5, B1, load_f32x4(&a1[k]));
  y6 = vmlaq_f32(y6, B1, load_f32x4(&a2[k]));
  y7 = vmlaq_f32(y7, B1, load_f32x4(&a3[k]));

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c0[1] += alpha*hsum_f32x4(y1);
  c0[2] += alpha*hsum_f32x4(y2);
  c0[3] += alpha*hsum_f32x4(y3);
  c1[0] += alpha*hsum_f32x4(y4);
  c1[1] += alpha*hsum_f32x4(y5);
  c1[2] += alpha*hsum_f32x4(y6);
  c1[3] += alpha*hsum_f32x4(y7);
}



// update 2x1 block of C; two rows, one column; (dmult2x1x1)
static inline
void __mult2c1(float *c0, 
               const float *a0, const float *a1,
               const float *b0, float alpha, int nR)
{
  register int k;
  register float32x4_t y0, y1, B0, Z;
  register uint32x4_t M;

  y0 = zero_f32x4();
  y1 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    B0 = load_f32x4(&b0[k]);
    y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
    y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));
  }
  if (k == nR)
    goto update;

  Z  = zero_f32x4();
  M  = vld1q_u32(&__masks_f32[nR-k][0]);
  B0 = load_f32x4(&b0[k]);
  B0 = (float32x4_t)vbslq_u32(M, (uint32x4_t)B0, (uint32x4_t)Z);

  y0 = vmlaq_f32(y0, B0, load_f32x4(&a0[k]));
  y1 = vmlaq_f32(y1, B0, load_f32x4(&a1[k]));

update:
  c0[0] += alpha*hsum_f32x4(y0);
  c0[1] += alpha*hsum_f32x4(y1);
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

