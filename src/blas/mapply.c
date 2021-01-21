
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_apply) && defined(armas_apply2)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

/**
 * @brief Apply function element wise to matrix A.
 *
 * Updates elements of A such that \f$ A_{i,j} = oper( A_{i,j} ) \f$
 *
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] oper
 *    Operator function.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @retval 0  Success
 * @retval <0 Error
 * @ingroup matrix
 */
int armas_apply(armas_dense_t *A, armas_operator_t oper, int flags)
{
    int i, j;

    if (armas_size(A) == 0)
        return 0;

    switch (flags & (ARMAS_LOWER | ARMAS_UPPER)) {
    case ARMAS_LOWER:
        for (j = 0; j < A->cols; j++) {
            for (i = j; i < A->rows; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j)));
            }
        }
        break;
    case ARMAS_UPPER:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i <= j; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j)));
            }
        }
        break;
    default:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < A->rows; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j)));
            }
        }
        break;
    }
    return 0;
}

/**
 * @brief Apply function element wise to matrix A.
 *
 * Updates elements of A such that \f$ A_{i,j} = oper( A_{i,j}, \alpha ) \f$
 * 
 * @param[in,out] A
 *    On entry, first input matrix. On exit result matrix.
 * @param[in] oper
 *    Operator function.
 * @param[in] alpha
 *    Operator function constant parameter
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @retval 0  Success
 * @retval <0 Error
 * 
 * @ingroup matrix
 */
int armas_apply2(armas_dense_t *A, armas_operator2_t oper, DTYPE alpha, int flags)
{
    int i, j;

    if (armas_size(A) == 0)
        return 0;

    switch (flags & (ARMAS_LOWER | ARMAS_UPPER)) {
    case ARMAS_LOWER:
        for (j = 0; j < A->cols; j++) {
            for (i = j; i < A->rows; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j), alpha));
            }
        }
        break;
    case ARMAS_UPPER:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i <= j; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j), alpha));
            }
        }
        break;
    default:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < A->rows; i++) {
                armas_set_unsafe(A, i, j, oper(armas_get_unsafe(A, i, j), alpha));
            }
        }
        break;
    }
    return 0;
}


/**
 * @brief Iterate through elements of A.
 *
 * @param[in,out] A
 *    Input matrix;
 * @param[in] oper
 *    Iterator function.
 * @param[in] alpha
 *    Iterator function private parameters.
 * @param[in] flags
 *    Indicator flags for matrix shape, ARMAS_UPPER or ARMAS_LOWER
 *
 * @retval 0  Success
 * @retval <0 Error
 *
 * @ingroup matrix
 */
int armas_iterate(const armas_dense_t *A, armas_iterator_t oper, void *p, int flags)
{
    int i, j;

    if (armas_size(A) == 0)
        return 0;

    switch (flags & (ARMAS_LOWER | ARMAS_UPPER)) {
    case ARMAS_LOWER:
        for (j = 0; j < A->cols; j++) {
            for (i = j; i < A->rows; i++) {
                oper(armas_get_unsafe(A, i, j), p);
            }
        }
        break;
    case ARMAS_UPPER:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i <= j; i++) {
                oper(armas_get_unsafe(A, i, j), p);
            }
        }
        break;
    default:
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < A->rows; i++) {
                oper(armas_get_unsafe(A, i, j), p);
            }
        }
        break;
    }
    return 0;
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
