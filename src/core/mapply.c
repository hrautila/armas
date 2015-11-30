
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_apply) && defined(__armas_apply2)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

/**
 * @brief Apply function element wise to matrix A.
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] oper
 *    Operator function.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @return 
 *    0 for success, -1 for error
 */
int __armas_apply(__armas_dense_t *A, __armas_operator_t oper, int flags)
{
  int i, j;

  if (__armas_size(A) == 0)
    return 0;

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j)));
      }
    }
    break;
  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j)));
      }
    }
    break;
  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j)));
      }
    }
    break;
  }
  return 0;
}

/**
 * @brief Apply function element wise to matrix A.
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] oper
 *    Operator function.
 * @param[in] val
 *    Operator function constant parameter
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @return 
 *    0 for success, -1 for error
 */
int __armas_apply2(__armas_dense_t *A, __armas_operator2_t oper, DTYPE val, int flags)
{
  int i, j;

  if (__armas_size(A) == 0)
    return 0;

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j), val));
      }
    }
    break;
  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j), val));
      }
    }
    break;
  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        __armas_set_unsafe(A, i, j, oper(__armas_get_unsafe(A, i, j), val));
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
