
#include <armas/armas.h>
#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__lapack_pivots) 
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(__armas_swap) && defined(__armas_iamax)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "pivot.h"

/*
 * \brief Swap rows of matrix
 *
 * \ingroup lapackaux internal
 */
void __swap_rows(__armas_dense_t *A, int src, int dst, armas_conf_t *conf) {
  __armas_dense_t r0, r1;
  if (src == dst || A->cols <= 0)
    return;
  if (src >= A->rows || dst >= A->rows)
    return;

  __armas_submatrix(&r0, A, src, 0, 1, A->cols);
  __armas_submatrix(&r1, A, dst, 0, 1, A->cols);
  __armas_swap(&r0, &r1, (armas_conf_t *)0);
}

/*
 * \brief Swap columns of matrix
 *
 * \ingroup lapackaux internal
 */
void __swap_cols(__armas_dense_t *A, int src, int dst, armas_conf_t *conf) {
  __armas_dense_t r0, r1;
  if (src == dst || A->rows <= 0)
    return;
  if (src >= A->cols || dst >= A->cols)
    return;

  __armas_submatrix(&r0, A, 0, src, A->rows, 1);
  __armas_submatrix(&r1, A, 0, dst, A->rows, 1);
  __armas_swap(&r0, &r1, (armas_conf_t *)0);
}


void __apply_pivots(__armas_dense_t *A, armas_pivot_t *P, armas_conf_t *conf)
{
  int k, n;

  if (A->cols == 0)
    return;

  for (k = 0; k < P->npivots; k++) {
    n = P->indexes[k];
    if (n > 0 && n-1 != k) {
      __swap_rows(A, n-1, k, conf);
    }
  }
}

/*
 * \brief Apply row pivots forward or backward.
 *
 * \ingroup lapackaux internal
 */
void __apply_row_pivots(__armas_dense_t *A, armas_pivot_t *P,
                        int dir, armas_conf_t *conf)
{
  int k, n;

  if (A->cols == 0)
    return;

  if (dir == PIVOT_FORWARD) {
    for (k = 0; k < P->npivots; k++) {
      n = P->indexes[k];
      if (n > 0 && n-1 != k) {
        __swap_rows(A, n-1, k, conf);
      }
    }
  } else {
    for (k = P->npivots-1; k >= 0; k--) {
      n = P->indexes[k];
      if (n > 0 && n-1 != k) {
        __swap_rows(A, n-1, k, conf);
      }
    }
  }
}

#if 0
void __apply_col_pivots(__armas_dense_t *A, armas_pivot_t *P,
                        int dir, armas_conf_t *conf)
{
}
#endif

/*
 * \brief Find index to largest absolute of vector.
 *
 * Find largest absolute value on a vector. Assumes A is vector. Returns non-zero
 * n and largest value at index n-1. On error returns zero.
 *
 * \ingroup lapackaux internal
 */
int __pivot_index(__armas_dense_t *A, armas_conf_t *conf)
{
  return __armas_iamax(A, conf) + 1;
}

/*
 * \brief Sort vector elements
 *
 * \param D [in]
 *      Vector to sort
 * \param P [in,out]
 *      Pivot table, on exit contains sorted order of vector
 * \param direction [in]
 *      Sort direction, >0 ascending, <0 descending
 * \return
 *      Non-zero if D not a vector, zero otherwise
 *
 * \ingroup lapackaux internal
 *
 * (This is maybe not so usefull at all, tricky to order based on this.)
 */
int __pivot_sort(__armas_dense_t *D, armas_pivot_t *P, int direction)
{
    int k, j, pk, pj;
    DTYPE cval, tmpval;

    if (! __armas_isvector(D)) {
        return -1;
    }

    // initialize the array
    for (k = 0; k < __armas_size(D); k++) {
        armas_pivot_set(P, k, k+1);
    }
    // simple indirect insertion sort
    for (k = 1; k < __armas_size(D); k++) {
        pk = armas_pivot_get(P, k) - 1;
        cval = __ABS(__armas_get_at_unsafe(D, pk));
        for (j = k; j > 0; j--) {
            pj = armas_pivot_get(P, j-1) - 1;
            tmpval = __ABS(__armas_get_at_unsafe(D, pj));
            if (direction < 0 && tmpval >= cval) {
                break;
            }
            if (direction > 0 && tmpval <= cval) {
                break;
            }
            armas_pivot_set(P, j, pj+1);
        }
        armas_pivot_set(P, j, pk+1);
    }
    return 0;
}


#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// End:
