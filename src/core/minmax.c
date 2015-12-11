
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__armas_aminmax)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "simd.h"

#if defined(FLOAT32)
#define __MINF min_f32
#define __MAXF max_f32
#else
#define __MINF min_f64
#define __MAXF max_f64
#endif

/**
 * @brief Find 
 *
 * \param[out] min
 *    Minimum value 
 * \param[out] max
 *    Maximum value 
 * \param[in] A
 *    Input matrix
 * \param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @return 
 *    0 for success, -1 for error
 */
int __armas_aminmax(DTYPE *min, DTYPE *max, const __armas_dense_t *A, int flags)
{
  int i, j;
  DTYPE _min, _max, aval;

  if (__armas_size(A) == 0)
    return 0;

  _min = _max = __ABS(__armas_get_unsafe(A, 0, 0));

  switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
  case ARMAS_LOWER:
    for (j = 0; j < A->cols; j++) {
      for (i = j; i < A->rows; i++) {
        aval = __ABS(__armas_get_unsafe(A, i, j));
        _min = __MINF(_min, aval);
        _max = __MAXF(_max, aval);
      }
    }
    break;

  case ARMAS_UPPER:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i <= j && i < A->rows; i++) {
        aval = __ABS(__armas_get_unsafe(A, i, j));
        _min = __MINF(_min, aval);
        _max = __MAXF(_max, aval);
      }
    }
    break;

  default:
    for (j = 0; j < A->cols; j++) {
      for (i = 0; i < A->rows; i++) {
        aval = __ABS(__armas_get_unsafe(A, i, j));
        _min = __MINF(_min, aval);
        _max = __MAXF(_max, aval);
      }
    }
    break;
  }
  if (*min)
    *min = _min;
  if (*max)
    *max = _max;
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
