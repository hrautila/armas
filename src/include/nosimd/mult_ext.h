
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef _MULT_EXT_NOSIMD_H
#define _MULT_EXT_NOSIMD_H 1

#include "eft.h"

// update single element of C; (mult1x1x4)
static inline
void __mult1c1_ext(DTYPE *c, DTYPE *d, const DTYPE *a, const DTYPE *b, DTYPE alpha, int nR)
{
  register int k;
  DTYPE y0, q0, p0, r0, e0;

  twoprod(&y0, &e0, a[0], b[0]);
  for (k = 1; k < nR; k++) {
    twoprod(&p0, &q0, a[k+0], b[k+0]);
    twosum(&y0, &r0, y0, p0);
    e0 += q0 + r0;
  }
  // p0 + q0 = alpha * sum a[:]*b[:]
  twoprod(&p0, &q0, alpha, y0);
  e0 = alpha*e0 + q0;
  // c[0] + q0 = p0 + c[0]
  twosum(c, &q0, p0, c[0]);
  // add error terms 
  d[0] += q0 + e0;
}


// update 1x2 block of C (mult2x1x4)
static inline
void __mult1c2_ext(DTYPE *c0, DTYPE *c1, DTYPE *d0, DTYPE *d1, 
                   const DTYPE *a,
                   const DTYPE *b0, const DTYPE *b1, DTYPE alpha, int nR)
{
  register int k;
  DTYPE y0, y1, e0, e1, p0, p1, q0, q1, r0, r1;

  twoprod(&y0, &e0, a[0], b0[0]);
  twoprod(&y1, &e1, a[0], b1[0]);
  for (k = 1; k < nR; k++) {
    twoprod(&p0, &q0, a[k+0], b0[k+0]);
    twosum(&y0, &r0, y0, p0);
    e0 += q0 + r0;
    twoprod(&p1, &q1, a[k+0], b1[k+0]);
    twosum(&y1, &r1, y1, p1);
    e1 += q1 + r1;
  }
  // p0 + q0 = alpha * sum a[:]*b[:]
  twoprod(&p0, &q0, alpha, y0);
  e0 = alpha*e0 + q0;
  twoprod(&p1, &q1, alpha, y1);
  e1 = alpha*e1 + q1;
  // c[0] + q0 = p0 + c[0]
  twosum(c0, &q0, p0, c0[0]);
  twosum(c1, &q1, p1, c1[0]);
  // add error terms 
  d0[0] += q0 + e0;
  d1[0] += q1 + e1;
}



#if 0
// Disabled for time being

// update 1x4 block of C; (mult4x1x1)
static inline
void __mult1c4_ext(DTYPE *c0, DTYPE *c1, DTYPE *c2, DTYPE *c3,
                   const DTYPE *a,
                   const DTYPE *b0, const DTYPE *b1,
                   const DTYPE *b2, const DTYPE *b3, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += a[k]*b0[k];
    y1 += a[k]*b1[k];
    y2 += a[k]*b2[k];
    y3 += a[k]*b3[k];
  }
update:
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
  c2[0] += y2*alpha;
  c3[0] += y3*alpha;
}
#endif  // 0


#endif  // _MULT_EXT_NOSIMD_H

// Local Variables:
// indent-tabs-mode: nil
// End:

