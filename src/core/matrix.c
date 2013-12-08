
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "internal.h"
#include "matcpy.h"
#include "matrix.h"
  
// non-inline functions

__armas_dense_t *__armas_transpose(__armas_dense_t *d, __armas_dense_t *s)
{
  if (d->rows != s->cols || d->cols != s->rows)
    return (__armas_dense_t *)0;
  
  __CPTRANS(d->elems, d->step, s->elems, s->step, s->rows, s->cols);
  return d;
}

__armas_dense_t *__armas_mcopy(__armas_dense_t *d, __armas_dense_t *s)
{
  if (d->rows != s->rows || d->cols != s->cols)
    return (__armas_dense_t *)0;
  
  __CP(d->elems, d->step, s->elems, s->step, s->rows, s->cols);
  return d;
}

// Return true if A is element-wise equal with a tolerance to B. The tolerance
// values are positive, typically very small numbers.
// Elements are equal within tolerance if
//     abs(a[i,j]-b[i,j]) <= atol + rtol*abs(b[i,j])
int __armas_intolerance(__armas_dense_t *a, __armas_dense_t *b, ABSTYPE atol, ABSTYPE rtol)
{
  register int i, j;
  ABSTYPE df, ref;

  if (a->rows != b->rows || a->cols != b->cols)
    return 0;

  for (j = 0; j < a->cols; j++) {
    for (i = 0; i < a->rows; i++) {
      df = __ABS(a->elems[i+j*a->step] - b->elems[i+j*b->step]);
      ref = atol + rtol * __ABS(b->elems[i+j*b->step]);
      if (df > ref)
        return 0;
    }
  }
  return 1;
}

// Default relative tolerance, these are spyed from Python NumPy
static const ABSTYPE RTOL = 1.0000000000000001e-05;

// Default absolute tolerance
static const ABSTYPE ATOL = 1e-8;

// Return true if A is element-wise equal with a tolerance to B.
int __armas_allclose(__armas_dense_t *a, __armas_dense_t *b)
{
  return __armas_intolerance(a, b, ATOL, RTOL);
}


void __armas_print(const __armas_dense_t *m, FILE *out)
{
  __armas_printf(out, "%8.1", m);
}

void __armas_printf(FILE *out, const char *efmt, const __armas_dense_t *m)
{
  int i, j;
  if (!m)
    return;
  if (!efmt)
    efmt = "%8.1e";

  int rowpartial = m->rows > 18;
  int colpartial = m->cols > 9;
  for (i = 0; i < m->rows; i++ ) {
    printf("[");
    for (j = 0; j < m->cols; j++ ) {
      if (j > 0) {
	printf(", ");
      }
      printf(efmt, m->elems[j*m->step+i]);
      if (colpartial && j == 3) {
        j = m->cols - 5;
        printf(", ...");
      }
    }
    printf("]\n");
    if (rowpartial && i == 8) {
      printf(" ....\n");
      i = m->rows - 10;
    }
  }
}

int __armas_set_values(__armas_dense_t *m, VALUEFUNC value, int flags)
{
  int i, j;
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_SYMM)) {
  case ARMAS_UPPER:
    for (j = 0; j < m->cols; j++) {
      for (i = 0; i < j && i < m->rows; i++) {
        m->elems[j*m->step+i] = value(i, j);
      }
      // don't set diagonal on upper trapezoidal matrix (cols > rows)
      if (j < m->rows)
        m->elems[j*m->step + j] = flags & ARMAS_UNIT ? __ONE : value(j, j);
    }
    break;
  case ARMAS_LOWER:
    for (j = 0; j < m->cols; j++) {
      if (j < m->rows)
        m->elems[j*m->step + j] = flags & ARMAS_UNIT ? __ONE : value(j, j);
      for (i = j+1; i < m->rows; i++) {
        m->elems[j*m->step+i] = value(i, j);
      }
    }
    break;
  case ARMAS_SYMM:
    if (m->rows != m->cols)
      return -1;
    for (j = 0; j < m->cols; j++) {
      m->elems[j*m->step + j] = flags & ARMAS_UNIT ? __ONE : value(j, j);
      for (i = j+1; i < m->rows; i++) {
        m->elems[j*m->step+i] = value(i, j);
        m->elems[i*m->step+j] = m->elems[j*m->step+i];
      }
    }
    break;
  default:
    for (j = 0; j < m->cols; j++) {
      for (i = 0; i < m->rows; i++) {
        m->elems[j*m->step+i] = value(i, j);
      }
    }
  }
  return 0;
}

// make matrix triangular with unit or non-unit diagonal.
void __armas_mk_trm(__armas_dense_t *m, int flags) {
  int i, j;
  if (flags & ARMAS_UPPER) {
    // clear lower triangular/trapezoidial part
    for (j = 0; j < m->cols; j++) {
      if (flags & ARMAS_UNIT)
        m->elems[j + j*m->step] = __ONE;
      for (i = j+1; i < m->rows; i++) {
        m->elems[j*m->step+i] = __ZERO;
      }
    }
  }
  else if (flags & ARMAS_LOWER) {
    // clear upper triangular/trapezoidial part
    for (j = 0; j < m->cols; j++) {
      for (i = 0; i < m->rows & i < j; i++) {
        m->elems[i + j*m->step] = __ZERO;
      }
      if (flags & ARMAS_UNIT && j < m->rows)
        m->elems[j + j*m->step] = __ONE;
    }
  }
}

int __armas_mscale(__armas_dense_t *m, const DTYPE alpha, int flags)
{
  int c, n;
  __armas_dense_t C;
  switch (flags & (ARMAS_SYMM|ARMAS_UPPER|ARMAS_LOWER)) {
  case ARMAS_UPPER:
    // scale strictly upper triangular part, if UNIT set, don't touch diagonal
    n = flags & ARMAS_UNIT ? 1 : 0;
    for (c = n; c < m->rows; c++) {
      __armas_submatrix(&C, m, c, c+n, 1, m->rows-c-n);
      __armas_scale(&C, alpha, (armas_conf_t *)0);
    }
    break;

  case ARMAS_LOWER:
    // scale strictly lower triangular part. if UNIT set, don't touch diagonal
    n = flags & ARMAS_UNIT ? 1 : 0;
    for (c = 0; c < m->cols-n; c++) {
      __armas_submatrix(&C, m, c+n, c, m->rows-c-n, 1);
      __armas_scale(&C, alpha, (armas_conf_t *)0);
    }
    break;

  case ARMAS_SYMM:
    if (m->rows != m->cols)
      return -1;
    // fall through to do it.
  default:
    __blk_scale((mdata_t *)m, alpha, m->rows, m->cols);
    break;
  }
  return 0;
}

int __armas_madd(__armas_dense_t *m, DTYPE alpha, int flags)
{
  int c, n;
  __armas_dense_t C;
  switch (flags & (ARMAS_SYMM|ARMAS_UPPER|ARMAS_LOWER)) {
  case ARMAS_UPPER:
    // scale strictly upper triangular part, if UNIT set, don't touch diagonal
    // (works for upper trapezoidal matrix too)
    n = flags & ARMAS_UNIT ? 1 : 0;
    for (c = n; c < m->rows; c++) {
      __armas_submatrix(&C, m, c, c+n, 1, m->rows-c-n);
      __blk_add((mdata_t *)&C, alpha, C.rows, C.cols);
    }
    break;

  case ARMAS_LOWER:
    // scale strictly lower triangular part. if UNIT set, don't touch diagonal
    // (works for lower trapezoidal matrix too)
    n = flags & ARMAS_UNIT ? 1 : 0;
    for (c = 0; c < m->cols-n; c++) {
      __armas_submatrix(&C, m, c+n, c, m->rows-c-n, 1);
      __blk_add((mdata_t *)&C, alpha, C.rows, C.cols);
    }
    break;

  case ARMAS_SYMM:
    if (m->rows != m->cols)
      return -1;
    // fall through to do it.
  default:
    __blk_add((mdata_t *)m, alpha, m->rows, m->cols);
    break;
  }
  return 0;
}


// Local Variables:
// indent-tabs-mode: nil
// End:
