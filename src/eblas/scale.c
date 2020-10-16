
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_ext_scale_unsafe)
#define ARMAS_PROVIDES 1
#endif

// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "matcpy.h"
#include "eft.h"

// scale block with constant (C + dC = beta*A)
void armas_ext_scale_unsafe(
  armas_dense_t *C0, armas_dense_t *dC,
  DTYPE beta,
  const armas_dense_t *A)
{
    int i, j;
    if (beta == ZERO) {
        for (j = 0; j < C0->cols; j++) {
            for (i = 0; i < C0->rows; i++) {
                C0->elems[i+j*C0->step] = ZERO;
                dC->elems[i+j*dC->step] = ZERO;
            }
        }
        return;
    }
    if (beta == ONE) {
        for (j = 0; j < C0->cols; j++) {
            for (i = 0; i < C0->rows; i++) {
                C0->elems[i+j*C0->step] = A->elems[i+j*A->step];
                dC->elems[i+j*dC->step] = ZERO;
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
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
