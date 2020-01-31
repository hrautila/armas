
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Matrix basic operators
 */

//! \cond
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "matrix.h"
#include "internal.h"
#include "matcpy.h"

// CPU cache line size in bytes (CPU spesific)
#define CACHELINE 64
#define CLMASK (CACHELINE-1)

// return adjusted size to allow alignement to cacheline
#define ALIGNSIZE(n, T) (n + (CACHELINE/sizeof(T)-1))

// return byte offset to first CPU cacheline aligned byte.
#define ALIGNOFFSET(ptr) \
  (((unsigned long)(ptr) & CLMASK) ? CACHELINE-((unsigned long)(ptr) & CLMASK) : 0)
//! \endcond

// non-inline functions

/**
 * Initialize matrix structure and allocate space for elements.
 *
 * @param [in,out] m
 *    On entry uninitialized matrix. On exit initialized matrix with space allocated
 *    for elements.
 * @param [in] r
 *    Number of rows
 * @param [in] c
 *    Number of columns
 *
 * If number of rows or columns is zero, no space is allocated but matrix
 * is properly initialized to zero size.
 *
 * @return Pointer to initialized matrix.
 * \ingroup matrix
 */
armas_x_dense_t *armas_x_init(armas_x_dense_t *m, int r, int c)
{
  int doff;

  if (r <= 0 || c <= 0) {
    m->rows = 0; m->cols = 0;
    m->step = 0;
    m->elems = (DTYPE *)0;
    m->__data = (void *)0;
    m->__nbytes = 0;
    return m;
  }
  // set first to adjusted element count
  m->__nbytes = ALIGNSIZE(r*c, DTYPE);
  m->__data = calloc(m->__nbytes, sizeof(DTYPE));
  if ( !m->__data ) {
    m->__nbytes = 0;
    return (armas_x_dense_t *)0;
  }
  // convert to number of bytes
  m->__nbytes *= sizeof(DTYPE);
  m->rows = r;
  m->cols = c;
  m->step = r;
  doff = ALIGNOFFSET(m->__data);
  m->elems = (DTYPE *)&((unsigned char *)m->__data)[doff];
  return m;
}

/**
 * @brief New copy of matrix
 *
 * Allocate space and copy matrix, A = newcopy(B)
 *
 * @param [in] A
 *   The source matrix
 *
 * @retval NOT NULL Success, pointer to new matrix
 * @retval NULL Failed
 * \ingroup matrix
 */
armas_x_dense_t *armas_x_newcopy(const armas_x_dense_t *A)
{
  armas_x_dense_t *Anew = armas_x_alloc(A->rows, A->cols);
  if (Anew) {
    CP(Anew->elems, Anew->step, A->elems, A->step, A->rows, A->cols);
  }
  return Anew;
}

/**
 * @brief Element-wise equality with in tolerances
 *
 * Test if \f$A == B\f$ within given tolerances. Elements are considered equal if
 *
 *  \f$|A_{i,j} - B_{i,j}| <= atol + rtol*B_{i,j}\f$
 *
 * @param [in] A, B
 *     Matrices to compare element wise
 * @param [in] atol
 *     Absolute tolerance
 * @param [in] rtol
 *     Relative tolerance
 *
 * @retval 0 not equal
 * @retval 1 equal
 * \ingroup matrix
 */
int armas_x_intolerance(const armas_x_dense_t *A, const armas_x_dense_t *B, ABSTYPE atol, ABSTYPE rtol)
{
  register int i, j;
  ABSTYPE df, ref;

  if (A->rows != B->rows || A->cols != B->cols)
    return 0;

  for (j = 0; j < A->cols; j++) {
    for (i = 0; i < A->rows; i++) {
      df = __ABS(A->elems[i+j*A->step] - B->elems[i+j*B->step]);
      ref = atol + rtol * __ABS(B->elems[i+j*B->step]);
      if (df > ref)
        return 0;
    }
  }
  return 1;
}

/**
 * @brief Default relative tolerance.
 */
static const ABSTYPE RTOL = 1.0000000000000001e-05;

/**
 * @brief Default absolute tolerance.
 */
static const ABSTYPE ATOL = 1e-8;

/**
 * @brief Element-wise equality with predefined tolerances
 *
 * @param [in] A, B 
 *    Matrices to compare
 *
 * @retval 0 not equal
 * @retval 1 equal
 * \ingroup matrix
 */
int armas_x_allclose(const armas_x_dense_t *A, const armas_x_dense_t *B)
{
  return armas_x_intolerance(A, B, ATOL, RTOL);
}


void armas_x_printf(FILE *out, const char *efmt, const armas_x_dense_t *m)
{
  unsigned int i, j;
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

void armas_x_print(const armas_x_dense_t *m, FILE *out)
{
  armas_x_printf(out, "%8.1", m);
}




// Local Variables:
// indent-tabs-mode: nil
// End:
