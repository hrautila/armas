
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix operators

//! \cond
#include <stdio.h>

#include "dtype.h"
//! \endcond
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_apply) && defined(armas_x_apply2)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

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
 * \ingroup matrix
 */
int armas_x_apply(armas_x_dense_t *A, armas_x_operator_t oper, int flags)
{
  int i, j;

  if (armas_x_size(A) == 0)
    return 0;

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j)));
      }
    }
    break;
  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j)));
      }
    }
    break;
  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j)));
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
 * \ingroup matrix
 */
int armas_x_apply2(armas_x_dense_t *A, armas_x_operator2_t oper, DTYPE val, int flags)
{
  int i, j;

  if (armas_x_size(A) == 0)
    return 0;

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j), val));
      }
    }
    break;
  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j), val));
      }
    }
    break;
  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        armas_x_set_unsafe(A, i, j, oper(armas_x_get_unsafe(A, i, j), val));
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
