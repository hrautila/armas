
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix element wise multiplication

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_mul_elems)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(__armas_mscale)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

static inline
void __mmul_lower(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top triangle
    for (k = j; k < j+3; k++) {
      for (i = k; i < j+3 && i < nR; i++) {
        a[i+k*lda] = a[i+k*lda] * b[i+k*ldb];
      }
    }
    // rest of the column block
    for (i = j+3; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] * b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] * b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] * b[i+(j+3)*ldb];
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = j; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
    }
  }
}

static inline
void __mmul_lower_abs(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top triangle
    for (k = j; k < j+3; k++) {
      for (i = k; i < j+3 && i < nR; i++) {
        a[i+k*lda] = __ABS(a[i+k*lda]) * __ABS(b[i+k*ldb]);
      }
    }
    // rest of the column block
    for (i = j+3; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) * __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) * __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) * __ABS(b[i+(j+3)*ldb]);
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = j; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
    }
  }
}

static inline
void __mmul_upper(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] * b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] * b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] * b[i+(j+3)*ldb];
    }
    // bottom triangle
    for (i = j+1; i < j+4 && i < nR; i++) {
      for (k = i; k < j+4; k++) {
        a[i+k*lda] = a[i+k*lda] * b[i+k*ldb];
      }
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
    }
  }
}

static inline
void __mmul_upper_abs(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC)
{
  register int i, j, k, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) * __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) * __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) * __ABS(b[i+(j+3)*ldb]);
    }
    // bottom triangle
    for (i = j+1; i < j+4 && i < nR; i++) {
      for (k = i; k < j+4; k++) {
        a[i+k*lda] = __ABS(a[i+k*lda]) * __ABS(b[i+k*ldb]);
      }
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i <= j && i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
    }
  }
}


static inline
void __mmul(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC, int flags)
{
  register int i, j, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
    for (j = 0; j < nC-3; j += 4) {
      // top column block
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[(j+0)+i*ldb];
        a[i+(j+1)*lda] = a[i+(j+1)*lda] * b[(j+1)+i*ldb];
        a[i+(j+2)*lda] = a[i+(j+2)*lda] * b[(j+2)+i*ldb];
        a[i+(j+3)*lda] = a[i+(j+3)*lda] * b[(j+3)+i*ldb];
      }
    }
    if (j == nC)
      return;
    for (; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[(j+0)+i*ldb];
      }
    }
    return;
  }

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
      a[i+(j+1)*lda] = a[i+(j+1)*lda] * b[i+(j+1)*ldb];
      a[i+(j+2)*lda] = a[i+(j+2)*lda] * b[i+(j+2)*ldb];
      a[i+(j+3)*lda] = a[i+(j+3)*lda] * b[i+(j+3)*ldb];
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = a[i+(j+0)*lda] * b[i+(j+0)*ldb];
    }
  }
}

static inline
void __mmul_abs(__armas_dense_t *A, const __armas_dense_t *B, int nR, int nC, int flags)
{
  register int i, j, lda, ldb;
  DTYPE *a, *b;

  a = __armas_data(A);
  b = __armas_data(B);
  lda = A->step; ldb = B->step;

  if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
    for (j = 0; j < nC-3; j += 4) {
      // top column block
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[(j+0)+i*ldb]);
        a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) * __ABS(b[(j+1)+i*ldb]);
        a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) * __ABS(b[(j+2)+i*ldb]);
        a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) * __ABS(b[(j+3)+i*ldb]);
      }
    }
    if (j == nC)
      return;
    for (; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[(j+0)+i*ldb]);
      }
    }
    return;
  }

  for (j = 0; j < nC-3; j += 4) {
    // top column block
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
      a[i+(j+1)*lda] = __ABS(a[i+(j+1)*lda]) * __ABS(b[i+(j+1)*ldb]);
      a[i+(j+2)*lda] = __ABS(a[i+(j+2)*lda]) * __ABS(b[i+(j+2)*ldb]);
      a[i+(j+3)*lda] = __ABS(a[i+(j+3)*lda]) * __ABS(b[i+(j+3)*ldb]);
    }
  }
  if (j == nC)
    return;
  for (; j < nC; j++) {
    for (i = 0; i < nR; i++) {
      a[i+(j+0)*lda] = __ABS(a[i+(j+0)*lda]) * __ABS(b[i+(j+0)*ldb]);
    }
  }
}

/**
 * @brief Element wise multiplication of \f$A_{i,j} = A_{i,j} * B_{i,j}\f$
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] B
 *    Second input matrix. If B is 1x1 then operation equals scaling of first matrix
 *    by constant.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @return 
 *    0 for success, -1 for error
 */
int __armas_mul_elems(__armas_dense_t *A, const __armas_dense_t *B, int flags)
{

  if (__armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;
  if (__armas_size(B) == 1)
    return __armas_mscale(A, __armas_get_unsafe(B, 0, 0), flags);

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
      __mmul_lower_abs(A, B, A->rows, A->cols);
    } else {
      __mmul_lower(A, B, A->rows, A->cols);
    }
    break;
  case ARMAS_UPPER:
    if (flags & ARMAS_ABS) {
      __mmul_upper_abs(A, B, A->rows, A->cols);
    } else {
      __mmul_upper(A, B, A->rows, A->cols);
    }
    break;
  default:
    if (flags & ARMAS_ABS) {
      __mmul_abs(A, B, A->rows, A->cols, flags);
    } else {
      __mmul(A, B, A->rows, A->cols, flags);
    }
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
