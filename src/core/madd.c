
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_add_elem)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(__armas_madd)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

/**
 * @brief Element wise addition of A = A + B
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] B
 *    Second input matrix. If B is 1x1 then operation equals to adding constant
 *    to first matrix.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 */
int __armas_add_elem(__armas_dense_t *A, const __armas_dense_t *B, int flags)
{
  int i, j;
  DTYPE a, b;
  if (__armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;
  if (__armas_size(B) == 1)
    return __armas_madd(A, __armas_get_unsafe(B, 0, 0), flags);
  if (A->rows != B->rows || A->cols != B->cols)
    return -1;

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        a = __armas_get_unsafe(A, i, j);
        b = __armas_get_unsafe(B, i, j);
        __armas_set_unsafe(A, i, j, a+b);
      }
    }
    break;
  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j; i++) {
        a = __armas_get_unsafe(A, i, j);
        b = __armas_get_unsafe(B, i, j);
        __armas_set_unsafe(A, i, j, a+b);
      }
    }
    break;
  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        a = __armas_get_unsafe(A, i, j);
        b = __armas_get_unsafe(B, i, j);
        __armas_set_unsafe(A, i, j, a+b);
      }
    }
    break;
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
