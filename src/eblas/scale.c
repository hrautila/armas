
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_ext_scale_unsafe)
#define __ARMAS_PROVIDES 1
#endif

// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "matcpy.h"
#include "eft.h"

// scale block with constant (C + dC = beta*A)
void armas_x_ext_scale_unsafe(
  armas_x_dense_t *C0, armas_x_dense_t *dC,
  DTYPE beta,
  const armas_x_dense_t *A)
{
  int i, j;
  if (beta == __ZERO) {
    for (j = 0; j < C0->cols; j++) {
      for (i = 0; i < C0->rows; i++) {
          C0->elems[i+j*C0->step] = __ZERO;
          dC->elems[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  if (beta == __ONE) {
    for (j = 0; j < C0->cols; j++) {
      for (i = 0; i < C0->rows; i++) {
        C0->elems[i+j*C0->step] = A->elems[i+j*A->step];
        dC->elems[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  for (j = 0; j < C0->cols; j++) {
    for (i = 0; i < C0->rows-1; i += 2) {
      twoprod(&C0->elems[(i+0)+j*C0->step],
              &dC->elems[(i+0)+j*dC->step], beta, A->elems[(i+0)+j*A->step]);
      twoprod(&C0->elems[(i+1)+j*C0->step],
              &dC->elems[(i+1)+j*dC->step], beta, A->elems[(i+1)+j*A->step]);
    }
    if (i != C0->rows) {
      twoprod(&C0->elems[(i+0)+j*C0->step],
              &dC->elems[(i+0)+j*dC->step], beta, A->elems[(i+0)+j*A->step]);
    }
  }
}

#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 2
// End:
