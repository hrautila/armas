
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix element wise addition

//! \cond
#include <stdio.h>

#include "dtype.h"
//! \endcond
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_add_elems)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(armas_x_madd)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

static inline
void __madd_lower(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top triangle
    for (k = j; k < j+3; k++) {
      for (i = k; i < j+3 && i < nR; i++) {
        a[i+k*lda] = a[i+k*lda] + b[i+k*ldb];
      }
    }
    // rest of the column block
    for (i = j+3; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] + b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] + b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] + b[i+(j+3)*ldb];
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = j; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
    }
  }
}

static inline
void __madd_lower_abs(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top triangle
    for (k = j; k < j+3; k++) {
      for (i = k; i < j+3 && i < nR; i++) {
        a[i+k*lda] = __ABS(a[i+k*lda]) + __ABS(b[i+k*ldb]);
      }
    }
    // rest of the column block
    for (i = j+3; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) + __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) + __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) + __ABS(b[i+(j+3)*ldb]);
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = j; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
    }
  }
}

static inline
void __madd_upper(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] + b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] + b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] + b[i+(j+3)*ldb];
    }
    // bottom triangle
    for (i = j+1; i < j+4 && i < nR; i++) {
      for (k = i; k < j+4; k++) {
        a[i+k*lda] = a[i+k*lda] + b[i+k*ldb];
      }
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
    }
  }
}

static inline
void __madd_upper_abs(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) + __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) + __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) + __ABS(b[i+(j+3)*ldb]);
    }
    // bottom triangle
    for (i = j+1; i < j+4 && i < nR; i++) {
      for (k = i; k < j+4; k++) {
        a[i+k*lda] = __ABS(a[i+k*lda]) + __ABS(b[i+k*ldb]);
      }
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
    }
  }
}


static inline
void __madd(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC, int flags)
{
  register int i, j, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
    for (j = 0; j < nC-3; j += 4) {
      // top column block
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[(j+0)+i*ldb];
        a[i+(j+1)*lda] = a[i+(j+1)*lda] + b[(j+1)+i*ldb];
        a[i+(j+2)*lda] = a[i+(j+2)*lda] + b[(j+2)+i*ldb];
        a[i+(j+3)*lda] = a[i+(j+3)*lda] + b[(j+3)+i*ldb];
      }
    }
    if (j == nC)
      return;
    for (; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[(j+0)+i*ldb];
      }
    }
    return;
  }

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] + b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] + b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] + b[i+(j+3)*ldb];
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] + b[i+(j+0)*ldb];
    }
  }
}

static inline
void __madd_abs(armas_x_dense_t *A, const armas_x_dense_t *B, int nR, int nC, int flags)
{
  register int i, j, lda, ldb;
  DTYPE *a, *b;

  a = armas_x_data(A);
  b = armas_x_data(B);
  lda = A->step; ldb = B->step;

  if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
    for (j = 0; j < nC-3; j += 4) {
      // top column block
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[(j+0)+i*ldb]);
        a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) + __ABS(b[(j+1)+i*ldb]);
        a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) + __ABS(b[(j+2)+i*ldb]);
        a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) + __ABS(b[(j+3)+i*ldb]);
      }
    }
    if (j == nC)
      return;
    for (; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[(j+0)+i*ldb]);
      }
    }
    return;
  }

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) + __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) + __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) + __ABS(b[i+(j+3)*ldb]);
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) + __ABS(b[i+(j+0)*ldb]);
    }
  }
}


/**
 * @brief Element wise addition of \f$A = A + B\f$
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] B
 *    Second input matrix. If B is 1x1 then operation equals to adding constant
 *    to first matrix.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER, ARMAS_LOWER or ARMAS_TRANS
 *
 * \ingroup matrix
 */
int armas_x_add_elems(armas_x_dense_t *A, const armas_x_dense_t *B, int flags)
{
  if (armas_x_size(A) == 0 || armas_x_size(B) == 0)
    return 0;
  if (armas_x_size(B) == 1)
    return armas_x_madd(A, armas_x_get_unsafe(B, 0, 0), flags);

  if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
    if (A->rows != B->cols || A->cols != B->rows)
      return -1;
  } else {
    if (A->rows != B->rows || A->cols != B->cols)
      return -1;
  }

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    if (flags & ARMAS_ABS) {
      __madd_lower_abs(A, B, A->rows, A->cols);
    } else {
      __madd_lower(A, B, A->rows, A->cols);
    }
    break;
  case ARMAS_UPPER:
    if (flags & ARMAS_ABS) {
      __madd_upper_abs(A, B, A->rows, A->cols);
    } else {
      __madd_upper(A, B, A->rows, A->cols);
    }
    break;
  default:
    if (flags & ARMAS_ABS) {
      __madd_abs(A, B, A->rows, A->cols, flags);
    } else {
      __madd(A, B, A->rows, A->cols, flags);
    }
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
