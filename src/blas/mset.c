// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Matrix set values.
 */

//! \cond
#include <stdio.h>

#include "dtype.h"
//! \endcond
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_set_values) && defined(armas_x_make_trm)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"

/**
 * @brief Set matrix values
 *
 * Set matrix A values using parameter value function.
 *
 *  \f$A_{i,j} = value(i, j)\f$
 *
 * Elements affected are selected with flag bits. If flag bit
 * ARMAS_UPPER (ARMAS_LOWER) is set the only upper (lower) triangular
 * or trapezoidal elements are set. If bit ARMAS_UNIT is set then
 * diagonal elements are not touched. If ARMAS_SYMM is set then
 * target matrix must be square matrix and upper triangular part
 * is transpose of lower triangular part. If bit ARMAS_UNIT is set then
 * diagonal entry is set to one.
 *
 * @param [out] A
 *      On exit, matrix with selected elements set.
 * @param [in] value
 *      the element value function
 * @param [in] flags
 *      flag bits (ARMAS_UPPER,ARMAS_LOWER,ARMAS_UNIT,ARMAS_SYMM)
 *
 * @returns 0  Succes
 * @returns -1 Failure
 * \ingroup matrix
 */
int armas_x_set_values(armas_x_dense_t *A, armas_x_valuefunc_t value, int flags)
{
    int i, j;
    switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_SYMM)) {
    case ARMAS_UPPER:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < j && i < A->rows; i++) {
                armas_x_set_unsafe(A, i, j, value(i, j));
            }
            // don't set diagonal on upper trapezoidal matrix (cols > rows)
            if (j < A->rows && !(flags & ARMAS_UNIT))
                armas_x_set_unsafe(A, j, j, value(j, j));
        }
        break;
    case ARMAS_LOWER:
        for (j = 0; j < A->cols; j++) {
            if (j < A->rows && !(flags & ARMAS_UNIT))
                armas_x_set_unsafe(A, j, j, value(j, j));
            for (i = j+1; i < A->rows; i++) {
                armas_x_set_unsafe(A, i, j, value(i, j));
            }
        }
        break;
    case ARMAS_SYMM:
        if (A->rows != A->cols)
            return -1;
        for (j = 0; j < A->cols; j++) {
            A->elems[j*A->step + j] = flags & ARMAS_UNIT ? __ONE : value(j, j);
            for (i = j+1; i < A->rows; i++) {
                armas_x_set_unsafe(A, i, j, value(i, j));
                armas_x_set_unsafe(A, j, i, armas_x_get_unsafe(A, i, j));
            }
        }
        break;
    default:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < A->rows; i++) {
                armas_x_set_unsafe(A, i, j, value(i, j));
            }
        }
    }
    return 0;
}

/**
 * @brief Make matrix triangular or trapezoidal
 *
 * Set matrix m elements not part of upper (lower) triangular
 * part to zero. If bit ARMAS_UNIT is set then diagonal element
 * is set to one.
 *
 * @param [in,out] m
 *      On entry, input matrix. On exit triangular matrix.
 * @param [in] flags
 *      flag bits (ARMAS_UPPER,ARMAS_LOWER,ARMAS_UNIT)
 *
 * \ingroup matrix
 */
void armas_x_make_trm(armas_x_dense_t *m, int flags)
{
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
            for (i = 0; i < m->rows && i < j; i++) {
                m->elems[i + j*m->step] = __ZERO;
            }
            if (flags & ARMAS_UNIT && j < m->rows)
                m->elems[j + j*m->step] = __ONE;
        }
    }
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
